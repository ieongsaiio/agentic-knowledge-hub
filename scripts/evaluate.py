#!/usr/bin/env python
"""Run legacy golden-set evaluation or config-driven benchmark ablations.

Examples:
    python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json
    python scripts/evaluate.py --config config/settings.yaml --dry-run
    python scripts/evaluate.py --experiments baseline,api_reranker
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TEST_SET = "tests/fixtures/golden_test_set.json"
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|passwd|token|authorization|credential|"
    r"private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(r"(?i)(bearer\s+)[^\s,;]+|(?:sk-[A-Za-z0-9_-]{8,})")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run golden-set evaluation or config-driven benchmark ablations."
    )
    parser.add_argument(
        "--test-set",
        default=None,
        help=(
            "Use legacy golden-set mode with this JSON file. When omitted, "
            "evaluation.benchmark is used if configured."
        ),
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name for legacy golden-set mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary. Answers and secret values are omitted.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip retrieval, primarily for evaluator smoke testing.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Settings YAML path (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated benchmark experiment names to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan benchmark experiments without downloads, ingestion, or API calls.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Clear and rebuild benchmark indexes for selected fingerprints.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override evaluation.output.directory for benchmark reports.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")
    return args


def main() -> int:
    """Load settings and dispatch to benchmark or legacy evaluation mode."""
    args = parse_args()
    try:
        from src.core.settings import load_settings

        settings = load_settings(args.config)
    except Exception as exc:
        print(f"Configuration error: {_safe_error(exc)}", file=sys.stderr)
        return 2

    benchmark = getattr(settings.evaluation, "benchmark", None)
    if args.test_set is None and benchmark is not None:
        return _run_benchmark(args, settings)

    test_set = args.test_set or DEFAULT_TEST_SET
    return _run_legacy(args, settings, test_set)


def _run_benchmark(args: argparse.Namespace, settings: Any) -> int:
    """Plan and execute the configured benchmark experiment suite."""
    try:
        from src.observability.evaluation.experiment_runner import ExperimentRunner

        selected = _parse_experiments(args.experiments)
        planner = ExperimentRunner(settings, lambda plan: None)
        plans = planner.plan(selected)
    except Exception as exc:
        print(f"Benchmark planning failed: {_safe_error(exc, settings)}", file=sys.stderr)
        return 2

    if args.dry_run:
        payload = {
            "mode": "benchmark_dry_run",
            "benchmark": settings.evaluation.benchmark.provider,
            "experiments": [_plan_dict(plan) for plan in plans],
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("BENCHMARK DRY RUN")
            for plan in plans:
                reuse = plan.reuses_index_of or "no"
                print(
                    f"- {plan.name}: fingerprint={plan.index_fingerprint} "
                    f"collection={plan.collection_name} reuse={reuse}"
                )
        return 0

    try:
        from src.libs.benchmark.benchmark_factory import BenchmarkFactory

        benchmark = BenchmarkFactory.create(settings)
        cases = benchmark.prepare()
        pdf_paths = _resolve_referenced_pdfs(benchmark, cases)
    except Exception as exc:
        print(
            f"Benchmark preparation failed: {_safe_error(exc, settings)}",
            file=sys.stderr,
        )
        return 1

    built_fingerprints: set[str] = set()

    def execute(plan: Any) -> Any:
        if not args.no_search:
            _ensure_index(
                plan,
                pdf_paths,
                built_fingerprints,
                force_reindex=args.force_reindex,
            )
        return _evaluate_plan(
            plan,
            cases,
            args.top_k,
            no_search=args.no_search,
            checkpoint_path=_benchmark_checkpoint_path(
                plan,
                cases,
                settings,
                args.output_dir,
            ),
        )

    runner = ExperimentRunner(settings, execute)
    suite = runner.run(selected)
    try:
        run_dir = _persist_suite(
            suite=suite,
            cases=cases,
            settings=settings,
            output_override=args.output_dir,
        )
    except Exception as exc:
        print(f"Failed to persist report: {_safe_error(exc, settings)}", file=sys.stderr)
        return 1

    _print_suite_summary(suite, run_dir, settings=settings, as_json=args.json)
    return 1 if not suite.experiments else 0


def _ensure_index(
    plan: Any,
    pdf_paths: Sequence[Path],
    built_fingerprints: set[str],
    *,
    force_reindex: bool,
) -> None:
    """Build one collection per unique fingerprint and reuse it thereafter."""
    if plan.index_fingerprint in built_fingerprints:
        return

    from src.core.settings import resolve_path
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    vector_store = VectorStoreFactory.create(
        plan.settings,
        collection_name=plan.collection_name,
    )
    stats = vector_store.get_collection_stats()
    vector_count = int(stats.get("count", 0))
    bm25_dir = resolve_path(f"data/db/bm25/{plan.collection_name}")
    bm25_path = bm25_dir / f"{plan.collection_name}_bm25.json"
    expected_sources = {path.name.casefold() for path in pdf_paths}
    indexed_sources = _indexed_source_names(vector_store)
    has_bm25 = bm25_path.is_file()
    complete_index = vector_count > 0 and has_bm25 and indexed_sources == expected_sources

    if complete_index and not force_reindex:
        built_fingerprints.add(plan.index_fingerprint)
        return

    stale_sources = indexed_sources - expected_sources
    needs_full_rebuild = (
        force_reindex
        or bool(stale_sources)
        or (vector_count > 0 and not has_bm25)
    )

    if needs_full_rebuild and vector_count:
        vector_store.clear()
        indexed_sources = set()
    if needs_full_rebuild and bm25_path.exists():
        bm25_path.unlink()

    from src.ingestion.pipeline import IngestionPipeline

    missing_paths = [
        path for path in pdf_paths if path.name.casefold() not in indexed_sources
    ]
    if not missing_paths and not has_bm25:
        missing_paths = list(pdf_paths)

    print(
        "[benchmark:index] "
        f"collection={plan.collection_name} "
        f"indexed_sources={len(indexed_sources)}/{len(expected_sources)} "
        f"missing={len(missing_paths)} "
        f"rebuild={needs_full_rebuild}",
        flush=True,
    )

    del vector_store

    pipeline = IngestionPipeline(
        plan.settings,
        collection=plan.collection_name,
        force=True,
    )
    try:
        for pdf_path in missing_paths:
            print(f"[benchmark:index] ingest {pdf_path.name}", flush=True)
            result = pipeline.run(str(pdf_path))
            if not result.success:
                raise RuntimeError(
                    f"Ingestion failed for {pdf_path.name}: {result.error or 'unknown error'}"
                )
    finally:
        pipeline.close()

    built_fingerprints.add(plan.index_fingerprint)


def _indexed_source_names(vector_store: Any) -> set[str]:
    """Return source file names represented by the current collection."""
    collection = getattr(vector_store, "collection", None)
    if collection is None or not hasattr(collection, "get"):
        return set()
    try:
        payload = collection.get(include=["metadatas"])
    except Exception:
        return set()

    sources: set[str] = set()
    for metadata in payload.get("metadatas") or []:
        if not isinstance(metadata, Mapping):
            continue
        source = metadata.get("source_path", metadata.get("source"))
        if isinstance(source, str) and source.strip():
            sources.add(Path(source).name.casefold())
    return sources


def _evaluate_plan(
    plan: Any,
    cases: list[Any],
    top_k: int,
    *,
    no_search: bool,
    checkpoint_path: Path | None = None,
) -> Any:
    """Construct configured providers and evaluate one experiment."""
    from src.core.query_engine.reranker import CoreReranker
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.answer_generator import (
        EvaluationAnswerGenerator,
    )
    from src.observability.evaluation.eval_runner import (
        EvalRunner,
        requires_generated_answer,
    )

    hybrid_search = (
        None
        if no_search
        else _create_hybrid_search(
            plan.settings,
            plan.collection_name,
        )
    )
    answer_generator = (
        EvaluationAnswerGenerator(plan.settings)
        if requires_generated_answer(plan.settings)
        else None
    )
    runner = EvalRunner(
        settings=plan.settings,
        hybrid_search=hybrid_search,
        reranker=CoreReranker(plan.settings),
        evaluator=EvaluatorFactory.create(plan.settings),
        answer_generator=answer_generator,
    )
    report = runner.run_cases(
        cases,
        top_k=top_k,
        checkpoint_path=checkpoint_path,
    )
    report.test_set_path = (
        f"{plan.settings.evaluation.benchmark.provider}:{plan.settings.evaluation.benchmark.split}"
    )
    return report


def _benchmark_checkpoint_path(
    plan: Any,
    cases: Sequence[Any],
    settings: Any,
    output_override: str | None,
) -> Path:
    """Return a stable checkpoint path for one experiment and case set."""
    base = Path(output_override or settings.evaluation.output.directory).expanduser()
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    identity = {
        "experiment": plan.name,
        "index_fingerprint": plan.index_fingerprint,
        "case_ids": [getattr(case, "case_id", None) for case in cases],
        "retrieval": repr(plan.settings.retrieval),
        "rerank_provider": plan.settings.rerank.provider,
        "rerank_model": (
            plan.settings.rerank.api_args.get("model")
            or plan.settings.rerank.model
        ),
        "rerank_top_k": plan.settings.rerank.top_k,
        "metrics": list(plan.settings.evaluation.metrics),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(plan.name)).strip("._")
    return base.resolve() / ".checkpoints" / f"{safe_name}_{digest}.jsonl"


def _create_hybrid_search(settings: Any, collection: str) -> Any:
    """Create HybridSearch from the same factories used by the query pipeline."""
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.settings import resolve_path
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding = EmbeddingFactory.create(settings)
    dense = create_dense_retriever(
        settings=settings,
        embedding_client=embedding,
        vector_store=vector_store,
    )
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25,
        vector_store=vector_store,
    )
    sparse.default_collection = collection
    return create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
    )


def _resolve_referenced_pdfs(benchmark: Any, cases: Iterable[Any]) -> list[Path]:
    """Resolve only documents referenced by the sampled benchmark cases."""
    pdf_root = Path(getattr(benchmark, "pdfs_dir", benchmark.pdf_dir))
    available = sorted(path for path in pdf_root.rglob("*.pdf") if path.is_file())
    by_key: dict[str, list[Path]] = {}
    for path in available:
        for key in {path.name.casefold(), path.stem.casefold()}:
            by_key.setdefault(key, []).append(path)

    document_names = sorted(
        {
            str(name).strip()
            for case in cases
            for name in getattr(case, "expected_documents", ())
            if str(name).strip()
        }
    )
    resolved: list[Path] = []
    missing: list[str] = []
    for document_name in document_names:
        raw = Path(document_name).name
        keys = [raw.casefold(), Path(raw).stem.casefold()]
        candidates = {candidate.resolve() for key in keys for candidate in by_key.get(key, [])}
        if not candidates:
            missing.append(document_name)
            continue
        if len(candidates) > 1:
            rendered = ", ".join(sorted(str(path) for path in candidates))
            raise ValueError(f"Ambiguous FinanceBench PDF reference {document_name!r}: {rendered}")
        resolved.append(next(iter(candidates)))

    if missing:
        raise FileNotFoundError(
            f"Missing referenced FinanceBench PDF(s) under {pdf_root}: {', '.join(missing)}"
        )
    if not resolved:
        raise ValueError("Sampled benchmark cases do not reference any PDFs")
    return sorted(set(resolved))


def _persist_suite(
    *,
    suite: Any,
    cases: Sequence[Any],
    settings: Any,
    output_override: str | None,
) -> Path:
    """Persist run metadata, per-experiment results, and a comparison table."""
    base = Path(output_override or settings.evaluation.output.directory).expanduser()
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = base.resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    plans_by_name = {plan.name: plan for plan in suite.plans}
    run_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": _redact(asdict(settings.evaluation.benchmark)),
        "case_count": len(cases),
        "elapsed_ms": suite.elapsed_ms,
        "plans": [
            {
                **_plan_dict(plan),
                "effective_settings": _redact(asdict(plan.settings)),
            }
            for plan in suite.plans
        ],
        "successful_experiments": list(suite.experiments),
        "errors": {name: _safe_error(message, settings) for name, message in suite.errors.items()},
    }
    _write_json(run_dir / "run.json", run_payload)

    for name, report in suite.experiments.items():
        plan = plans_by_name[name]
        report_dict = report.to_dict()
        safe_name = _safe_filename(name)
        summary = {
            "experiment": name,
            "collection": plan.collection_name,
            "index_fingerprint": plan.index_fingerprint,
            "reuses_index_of": plan.reuses_index_of,
            "query_count": report_dict["query_count"],
            "total_elapsed_ms": report_dict["total_elapsed_ms"],
            "evaluator_name": report_dict["evaluator_name"],
            "aggregate_metrics": report_dict["aggregate_metrics"],
        }
        _write_json(run_dir / f"{safe_name}.summary.json", summary)
        with (run_dir / f"{safe_name}.cases.jsonl").open("w", encoding="utf-8") as handle:
            for result in report_dict["query_results"]:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    _write_comparison(run_dir / "comparison.csv", suite, settings)
    return run_dir


def _write_comparison(path: Path, suite: Any, settings: Any) -> None:
    metric_names = sorted(
        {metric for report in suite.experiments.values() for metric in report.aggregate_metrics}
    )
    plans = {plan.name: plan for plan in suite.plans}
    fields = [
        "experiment",
        "status",
        "collection",
        "index_fingerprint",
        "reuses_index_of",
        "query_count",
        "total_elapsed_ms",
        *metric_names,
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for plan in suite.plans:
            report = suite.experiments.get(plan.name)
            row: dict[str, Any] = {
                "experiment": plan.name,
                "status": "success" if report is not None else "failed",
                "collection": plans[plan.name].collection_name,
                "index_fingerprint": plan.index_fingerprint,
                "reuses_index_of": plan.reuses_index_of or "",
                "query_count": len(report.query_results) if report else 0,
                "total_elapsed_ms": report.total_elapsed_ms if report else "",
                "error": _safe_error(suite.errors.get(plan.name, ""), settings),
            }
            if report:
                row.update(report.aggregate_metrics)
            writer.writerow(row)


def _run_legacy(args: argparse.Namespace, settings: Any, test_set: str) -> int:
    """Preserve the original explicit ``--test-set`` evaluation workflow."""
    try:
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.eval_runner import EvalRunner

        evaluator = EvaluatorFactory.create(settings)
    except Exception as exc:
        print(f"Failed to create evaluator: {_safe_error(exc, settings)}", file=sys.stderr)
        return 2

    hybrid_search = None
    if not args.no_search:
        try:
            collection = args.collection or "default"
            hybrid_search = _create_hybrid_search(settings, collection)
            print(f"HybridSearch initialized for collection: {collection}")
        except Exception as exc:
            print(
                f"Search initialization failed; continuing without retrieval: "
                f"{_safe_error(exc, settings)}",
                file=sys.stderr,
            )

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )
    try:
        report = runner.run(
            test_set_path=test_set,
            top_k=args.top_k,
            collection=None,
        )
    except Exception as exc:
        print(f"Evaluation failed: {_safe_error(exc, settings)}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                _console_report_dict(report),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        _print_report(report)
    return 0


def _print_suite_summary(
    suite: Any,
    run_dir: Path,
    *,
    settings: Any,
    as_json: bool,
) -> None:
    payload = {
        "run_directory": str(run_dir),
        "elapsed_ms": round(suite.elapsed_ms, 1),
        "experiments": {
            name: {
                "query_count": len(report.query_results),
                "aggregate_metrics": report.aggregate_metrics,
            }
            for name, report in suite.experiments.items()
        },
        "errors": {name: _safe_error(error, settings) for name, error in suite.errors.items()},
    }
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("BENCHMARK EVALUATION")
    for name, report in suite.experiments.items():
        metrics = ", ".join(
            f"{key}={value:.4f}" for key, value in sorted(report.aggregate_metrics.items())
        )
        print(f"- {name}: {metrics or 'no metrics'}")
    for name, error in suite.errors.items():
        print(f"- {name}: failed ({_safe_error(error, settings)})")
    print(f"Reports: {run_dir}")


def _print_report(report: Any) -> None:
    """Print the legacy report without generated or reference answers."""
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluator: {report.evaluator_name}")
    print(f"Test Set: {report.test_set_path}")
    print(f"Queries: {len(report.query_results)}")
    print(f"Time: {report.total_elapsed_ms:.0f} ms")
    print("AGGREGATE METRICS")
    for metric, value in sorted(report.aggregate_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    if not report.aggregate_metrics:
        print("  (no metrics computed)")
    print("PER-QUERY RESULTS")
    for index, result in enumerate(report.query_results, 1):
        print(
            f"  [{index}] retrieved={len(result.retrieved_chunk_ids)} "
            f"time={result.elapsed_ms:.0f} ms"
        )
        for metric, value in sorted(result.metrics.items()):
            print(f"      {metric}: {value:.4f}")


def _console_report_dict(report: Any) -> dict[str, Any]:
    payload = report.to_dict()
    for result in payload.get("query_results", []):
        result.pop("generated_answer", None)
        result.pop("reference_answer", None)
        for retrieved in result.get("retrieved_results", []):
            retrieved.pop("text", None)
    return payload


def _parse_experiments(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("--experiments must contain at least one name")
    return names


def _plan_dict(plan: Any) -> dict[str, Any]:
    return {
        "name": plan.name,
        "index_fingerprint": plan.index_fingerprint,
        "collection_name": plan.collection_name,
        "reuses_index_of": plan.reuses_index_of,
    }


def _redact(value: Any) -> Any:
    """Recursively redact values stored under secret-like keys."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): ("[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _secret_values(settings: Any) -> set[str]:
    values: set[str] = set()

    def visit(value: Any, secret_context: bool = False) -> None:
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, bool(_SECRET_KEY.search(str(key))))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, secret_context)
        elif secret_context and isinstance(value, str) and value:
            values.add(value)

    visit(settings)
    return values


def _safe_error(error: Any, settings: Any = None) -> str:
    if error is None or error == "":
        return ""
    text = str(error) or type(error).__name__
    if settings is not None:
        for secret in sorted(_secret_values(settings), key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
    return _SECRET_TEXT.sub(
        lambda match: f"{match.group(1)}[REDACTED]" if match.group(1) else "[REDACTED]",
        text,
    )


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "experiment"


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    sys.exit(main())
