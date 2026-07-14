"""Evaluation Panel page – run evaluations and view metrics.

Layout:
1. Configuration section: select evaluator backend, golden test set, top_k
2. Run button with progress indicator
3. Results section: aggregate metrics, per-query detail table
4. Optional: historical evaluation results comparison
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# Default golden test set location
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
# Evaluation results history file
EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")
DEFAULT_BENCHMARK_OUTPUT_DIR = Path("data/evaluation")

_COMPARISON_BASE_FIELDS = {
    "experiment",
    "status",
    "collection",
    "index_fingerprint",
    "reuses_index_of",
    "query_count",
    "total_elapsed_ms",
    "error",
}


def render() -> None:
    """Render the Evaluation Panel page."""
    st.header("📏 Evaluation Panel")
    st.markdown(
        "Run evaluation against a **golden test set** to measure retrieval "
        "and generation quality. Results include per-query details and "
        "aggregate metrics."
    )

    # ── Configuration Section ──────────────────────────────────────
    st.subheader("⚙️ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        backend = st.selectbox(
            "Evaluator Backend",
            options=["custom", "ragas", "composite"],
            index=0,
            key="eval_backend",
            help="Select which evaluator backend to use.",
        )

    # Show info/warning based on selected backend
    if backend in ("custom", "composite"):
        st.info(
            "ℹ️ **Custom Evaluator** 尚未完成数据集准备，当前仅为预留接口。"
            "Custom Evaluator 需要在 Golden Test Set 中填写 `expected_chunk_ids` "
            "作为 ground truth 才能计算 hit_rate / MRR 指标。"
            "目前建议使用 **ragas** 后端进行评估。",
            icon="🚧",
        )

    with col2:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=50,
            value=10,
            key="eval_top_k",
            help="Number of chunks to retrieve per query.",
        )

    with col3:
        collection = st.text_input(
            "Collection (optional)",
            value="",
            key="eval_collection",
            help="Limit retrieval to a specific collection.",
        )

    # Golden test set file selection
    golden_path_str = st.text_input(
        "Golden Test Set Path",
        value=str(DEFAULT_GOLDEN_SET),
        key="eval_golden_path",
        help="Path to the golden_test_set.json file.",
    )
    golden_path = Path(golden_path_str)

    # Validate golden set exists
    if not golden_path.exists():
        st.warning(
            f"⚠️ **Golden test set not found:** `{golden_path}`. "
            "Create a JSON file with test queries and expected results. "
            "See `tests/fixtures/golden_test_set.json` for the format."
        )

    # ── Answer Input Section (for Ragas) ───────────────────────────
    user_answers: dict[int, str] = {}
    if backend == "ragas" and golden_path.exists():
        st.divider()
        st.subheader("✏️ Provide Answers (回答输入)")
        st.caption(
            "**RAGAS 需要 Query + Context + Answer 三要素来评估。**"
            "日志中仅包含 Query 和检索到的上下文（Context），"
            "请为每个测试用例填写实际的系统回答（Answer），"
            "以便获得有意义的 faithfulness 和 answer_relevancy 评分。"
        )
        try:
            _test_cases = _load_golden_queries(golden_path)
            for tc_idx, tc in enumerate(_test_cases):
                ans_key = f"eval_answer_tc_{tc_idx}"
                default_val = tc.get("reference_answer", "")
                q_preview = tc["query"][:60] + ("…" if len(tc["query"]) > 60 else "")
                user_ans = st.text_area(
                    f"Q{tc_idx + 1}: {q_preview}",
                    value=st.session_state.get(ans_key, default_val),
                    height=80,
                    key=ans_key,
                    placeholder="请输入该问题对应的系统回答…",
                    help=(
                        f"Query: {tc['query']}\n\n"
                        "填写 LLM 生成的回答或期望的回答文本。"
                        "Ragas 会基于此评估 faithfulness（忠实度）和 answer_relevancy（相关性）。"
                    ),
                )
                if user_ans.strip():
                    user_answers[tc_idx] = user_ans.strip()

            # Show fill status
            filled = len(user_answers)
            total = len(_test_cases)
            if filled < total:
                st.warning(f"⚠️ 已填写 {filled}/{total} 个回答。未填写的用例将使用检索片段拼接作为回答（评估结果可能不准确）。")
            else:
                st.success(f"✅ 所有 {total} 个回答已填写。")
        except Exception as exc:
            st.warning(f"无法加载测试用例预览: {exc}")

    # ── Run Evaluation ─────────────────────────────────────────────
    st.divider()

    run_clicked = st.button(
        "▶️  Run Evaluation",
        type="primary",
        key="eval_run_btn",
        disabled=not golden_path.exists(),
    )

    if run_clicked:
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
            user_answers=user_answers if user_answers else None,
        )

    # ── Historical Results ─────────────────────────────────────────
    st.divider()
    _render_benchmark_experiment_results()

    st.divider()
    _render_history()


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: str | None,
    user_answers: dict[int, str] | None = None,
) -> None:
    """Execute an evaluation run and display results.

    Attempts to load the evaluator, run the golden test set, and
    display aggregate + per-query metrics.  Falls back to a graceful
    error message on failure.
    """
    with st.spinner("Loading evaluator and running evaluation…"):
        try:
            report_dict = _execute_evaluation(
                backend=backend,
                golden_path=golden_path,
                top_k=top_k,
                collection=collection,
                user_answers=user_answers,
            )
        except Exception as exc:
            st.error(f"❌ Evaluation failed: {exc}")
            logger.exception("Evaluation failed")
            return

    # ── Display results ────────────────────────────────────────────
    st.success("✅ Evaluation complete!")

    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)

    # Save to history
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: str | None,
    user_answers: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Run the evaluation pipeline and return the report dict.

    This function imports heavy dependencies lazily to keep the
    dashboard responsive when the page is not used.
    """
    from dataclasses import replace as dc_replace

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner

    settings = load_settings()

    # Override evaluator provider from UI selection — build a new full
    # Settings object so that RagasEvaluator can still access .llm / .embedding.
    eval_settings = settings.evaluation
    overridden_eval = type(eval_settings)(
        enabled=True,
        provider=backend,
        metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
    )
    # Replace only the evaluation sub-config in the full settings
    settings_with_override = dc_replace(settings, evaluation=overridden_eval)

    evaluator = EvaluatorFactory.create(settings_with_override)

    # Try to create HybridSearch (optional – works without if not configured)
    target_collection = collection or "default"
    hybrid_search = _try_create_hybrid_search(settings, target_collection)

    # Create reranker if enabled
    reranker = None
    try:
        from src.core.query_engine.reranker import create_core_reranker
        reranker = create_core_reranker(settings=settings)
        if not reranker.is_enabled:
            reranker = None
    except Exception as exc:
        logger.warning("Could not create reranker: %s", exc)

    # Build answer_override map: index → user-provided answer text
    # EvalRunner will use these instead of auto-generating from chunks.
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_overrides=user_answers,
        reranker=reranker,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=collection,
    )

    return report.to_dict()


def _try_create_hybrid_search(settings: Any, collection: str = "default") -> Any:
    """Attempt to create a HybridSearch instance.

    Returns None if required dependencies are not available
    (e.g., no indexed data).
    """
    try:
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        return create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None


def _render_aggregate_metrics(report: dict[str, Any]) -> None:
    """Display aggregate metrics as metric cards."""
    st.subheader("📊 Aggregate Metrics")

    agg = report.get("aggregate_metrics", {})

    if not agg:
        st.info("No aggregate metrics available.")
        return

    cols = st.columns(min(len(agg), 4))
    for idx, (name, value) in enumerate(sorted(agg.items())):
        with cols[idx % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )

    st.caption(
        f"Evaluator: **{report.get('evaluator_name', '—')}** · "
        f"Queries: **{report.get('query_count', 0)}** · "
        f"Total time: **{report.get('total_elapsed_ms', 0):.0f} ms**"
    )


def _render_query_details(report: dict[str, Any]) -> None:
    """Display per-query evaluation results in an expandable table."""
    st.subheader("🔍 Per-Query Details")

    query_results = report.get("query_results", [])
    if not query_results:
        st.info("No per-query results available.")
        return

    for idx, qr in enumerate(query_results):
        query = qr.get("query", "—")
        elapsed = qr.get("elapsed_ms", 0)
        metrics = qr.get("metrics", {})

        # Build metric summary for the expander label
        metric_summary = " · ".join(
            f"{k}: {v:.3f}" for k, v in sorted(metrics.items())
        )
        if not metric_summary:
            metric_summary = "no metrics"

        with st.expander(
            f"**Q{idx + 1}**: {query[:80]} — {elapsed:.0f} ms — {metric_summary}",
            expanded=False,
        ):
            # Metrics
            if metrics:
                mcols = st.columns(min(len(metrics), 4))
                for midx, (mname, mval) in enumerate(sorted(metrics.items())):
                    with mcols[midx % len(mcols)]:
                        st.metric(mname, f"{mval:.4f}")

            # Retrieved chunks
            chunks = qr.get("retrieved_chunk_ids", [])
            if chunks:
                st.markdown(f"**Retrieved Chunks** ({len(chunks)}):")
                st.code(", ".join(chunks[:20]), language=None)

            # Generated answer
            answer = qr.get("generated_answer")
            if answer:
                st.markdown("**Generated Answer:**")
                st.text(answer[:500])


def _render_history() -> None:
    """Display historical evaluation results for comparison."""
    st.subheader("📈 Evaluation History")

    history = _load_history()
    if not history:
        st.info(
            "**No evaluation history yet.** "
            "Configure the evaluator above and click \"Run Evaluation\" to start. "
            "Results will be saved here for comparison across runs."
        )
        return

    # Show recent runs as a table
    rows = []
    for entry in history[-10:]:  # last 10 runs
        rows.append(
            {
                "Timestamp": entry.get("timestamp", "—"),
                "Evaluator": entry.get("evaluator_name", "—"),
                "Queries": entry.get("query_count", 0),
                "Time (ms)": round(entry.get("total_elapsed_ms", 0)),
                **{
                    k: round(v, 4)
                    for k, v in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    st.dataframe(rows, use_container_width=True)


def _render_benchmark_experiment_results() -> None:
    """Render persisted config-driven benchmark experiment results."""
    st.subheader("Benchmark Experiment Results")

    output_dir = _resolve_benchmark_output_dir()
    st.caption(f"Reading benchmark reports from `{output_dir}`")

    runs = _discover_benchmark_runs(output_dir)
    if not runs:
        st.info(
            "No benchmark experiment reports found yet. Run "
            "`python scripts/evaluate.py --config config/settings.yaml "
            "--experiments baseline --json` to generate reports."
        )
        return

    labels = [_benchmark_run_label(run) for run in runs]
    selected_label = st.selectbox(
        "Benchmark Run",
        options=labels,
        index=0,
        key="benchmark_eval_run",
        help="Select one persisted benchmark evaluation run.",
    )
    selected = runs[labels.index(selected_label)]
    run_dir = Path(selected["path"])

    run_payload = selected.get("run", {})
    exp_rows = _build_benchmark_experiment_rows(run_dir, run_payload)
    if not exp_rows:
        st.warning("This benchmark run has no experiment summaries.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Cases", run_payload.get("case_count", len(exp_rows)))
    with c2:
        st.metric("Experiments", len(exp_rows))
    with c3:
        st.metric(
            "Succeeded",
            len([row for row in exp_rows if row.get("Status") == "success"]),
        )
    with c4:
        elapsed_ms = _to_float(run_payload.get("elapsed_ms"))
        st.metric("Suite Latency", _format_ms(elapsed_ms))

    metric_tab, latency_tab, metadata_tab = st.tabs(
        ["Experiment Metrics", "Latency Details", "Run Metadata"]
    )

    with metric_tab:
        st.dataframe(exp_rows, use_container_width=True, hide_index=True)

    with latency_tab:
        names = [str(row.get("Experiment", "")) for row in exp_rows]
        selected_experiment = st.selectbox(
            "Experiment",
            options=names,
            index=0,
            key="benchmark_latency_experiment",
        )
        case_rows = _build_benchmark_case_rows(run_dir, selected_experiment)
        summary = next(
            (row for row in exp_rows if row.get("Experiment") == selected_experiment),
            {},
        )
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.metric(
                "Avg Query",
                _format_ms(_to_float(summary.get("Avg Query Latency (ms)"))),
            )
        with lc2:
            st.metric(
                "P50 Query",
                _format_ms(_to_float(summary.get("P50 Query Latency (ms)"))),
            )
        with lc3:
            st.metric(
                "P95 Query",
                _format_ms(_to_float(summary.get("P95 Query Latency (ms)"))),
            )
        with lc4:
            st.metric(
                "Max Query",
                _format_ms(_to_float(summary.get("Max Query Latency (ms)"))),
            )

        if case_rows:
            st.dataframe(case_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No per-case JSONL output found for this experiment.")

    with metadata_tab:
        st.json(_compact_run_metadata(run_payload))


def _resolve_benchmark_output_dir() -> Path:
    """Resolve evaluation.output.directory from settings, with fallback."""
    try:
        from src.core.settings import load_settings, resolve_path

        settings = load_settings()
        output = getattr(getattr(settings, "evaluation", None), "output", None)
        directory = getattr(output, "directory", str(DEFAULT_BENCHMARK_OUTPUT_DIR))
        return resolve_path(directory)
    except Exception as exc:
        logger.warning("Could not resolve benchmark output directory: %s", exc)
        return DEFAULT_BENCHMARK_OUTPUT_DIR.resolve()


def _discover_benchmark_runs(output_dir: Path) -> list[dict[str, Any]]:
    """Return persisted benchmark runs, newest first."""
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_json = run_dir / "run.json"
        comparison = run_dir / "comparison.csv"
        if not run_json.exists() and not comparison.exists():
            continue
        payload = _read_json(run_json) if run_json.exists() else {}
        runs.append(
            {
                "path": run_dir,
                "name": run_dir.name,
                "created_at": payload.get("created_at", ""),
                "elapsed_ms": payload.get("elapsed_ms"),
                "case_count": payload.get("case_count"),
                "run": payload,
                "modified": run_dir.stat().st_mtime,
            }
        )
    return sorted(runs, key=lambda item: item["modified"], reverse=True)


def _benchmark_run_label(run: dict[str, Any]) -> str:
    created = run.get("created_at") or "unknown time"
    cases = run.get("case_count")
    elapsed = _format_ms(_to_float(run.get("elapsed_ms")))
    case_text = f", {cases} cases" if cases is not None else ""
    return f"{run.get('name', 'run')} ({created}{case_text}, {elapsed})"


def _build_benchmark_experiment_rows(
    run_dir: Path,
    run_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one dashboard row per configured benchmark experiment."""
    comparison_rows = _read_csv_rows(run_dir / "comparison.csv")
    summaries = _load_experiment_summaries(run_dir)
    plans_by_name = {
        str(plan.get("name", "")): plan
        for plan in run_payload.get("plans", [])
        if isinstance(plan, dict)
    }

    source_rows = comparison_rows or [
        {
            "experiment": summary.get("experiment", safe_name),
            "status": "success",
            "collection": summary.get("collection", ""),
            "index_fingerprint": summary.get("index_fingerprint", ""),
            "reuses_index_of": summary.get("reuses_index_of", ""),
            "query_count": summary.get("query_count", ""),
            "total_elapsed_ms": summary.get("total_elapsed_ms", ""),
            **summary.get("aggregate_metrics", {}),
        }
        for safe_name, summary in summaries.items()
    ]

    metric_names = _configured_metric_names(run_payload, source_rows, summaries)
    rows: list[dict[str, Any]] = []
    for raw in source_rows:
        experiment = str(raw.get("experiment", ""))
        safe_name = _safe_filename(experiment)
        summary = summaries.get(safe_name, {})
        plan = plans_by_name.get(experiment, {})
        configured_metrics = _plan_metrics(plan) or metric_names
        latencies = _case_latency_stats(run_dir / f"{safe_name}.cases.jsonl")
        total_elapsed = _to_float(
            raw.get("total_elapsed_ms", summary.get("total_elapsed_ms"))
        )
        query_count = _to_int(raw.get("query_count", summary.get("query_count")))
        row: dict[str, Any] = {
            "Experiment": experiment,
            "Status": raw.get("status", "success"),
            "Configured Metrics": ", ".join(configured_metrics),
            "Collection": raw.get("collection", summary.get("collection", "")),
            "Reuses Index Of": raw.get(
                "reuses_index_of",
                summary.get("reuses_index_of", ""),
            ),
            "Queries": query_count,
            "Total Latency (ms)": round(total_elapsed, 1),
            "Avg Query Latency (ms)": round(latencies.get("avg_ms", 0.0), 1),
            "P50 Query Latency (ms)": round(latencies.get("p50_ms", 0.0), 1),
            "P95 Query Latency (ms)": round(latencies.get("p95_ms", 0.0), 1),
            "Max Query Latency (ms)": round(latencies.get("max_ms", 0.0), 1),
        }
        for metric in metric_names:
            row[metric] = _to_display_number(
                raw.get(metric, summary.get("aggregate_metrics", {}).get(metric))
            )
        error = raw.get("error")
        if error:
            row["Error"] = error
        rows.append(row)
    return rows


def _build_benchmark_case_rows(run_dir: Path, experiment: str) -> list[dict[str, Any]]:
    """Build per-query rows for one benchmark experiment."""
    path = run_dir / f"{_safe_filename(experiment)}.cases.jsonl"
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(_read_jsonl(path), start=1):
        metrics = case.get("metrics", {}) if isinstance(case, dict) else {}
        row: dict[str, Any] = {
            "Case": index,
            "Case ID": case.get("case_id", ""),
            "Query": _truncate(str(case.get("query", "")), 120),
            "Latency (ms)": _to_display_number(case.get("elapsed_ms")),
            "Retrieved Chunks": len(case.get("retrieved_chunk_ids", []) or []),
        }
        if isinstance(metrics, dict):
            for name, value in sorted(metrics.items()):
                row[name] = _to_display_number(value)
        rows.append(row)
    return rows


def _load_experiment_summaries(run_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in run_dir.glob("*.summary.json"):
        safe_name = path.name[: -len(".summary.json")]
        value = _read_json(path)
        if isinstance(value, dict):
            summaries[safe_name] = value
    return summaries


def _configured_metric_names(
    run_payload: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    names: list[str] = []
    for plan in run_payload.get("plans", []):
        if isinstance(plan, dict):
            names.extend(_plan_metrics(plan))
    for summary in summaries.values():
        metrics = summary.get("aggregate_metrics", {})
        if isinstance(metrics, dict):
            names.extend(str(name) for name in metrics)
    for row in comparison_rows:
        names.extend(
            str(name)
            for name, value in row.items()
            if name not in _COMPARISON_BASE_FIELDS and str(value).strip() != ""
        )
    return sorted(dict.fromkeys(names))


def _plan_metrics(plan: dict[str, Any]) -> list[str]:
    effective = plan.get("effective_settings", {})
    evaluation = effective.get("evaluation", {}) if isinstance(effective, dict) else {}
    metrics = evaluation.get("metrics", []) if isinstance(evaluation, dict) else []
    if not isinstance(metrics, list):
        return []
    return [str(metric) for metric in metrics]


def _case_latency_stats(path: Path) -> dict[str, float]:
    values = [
        _to_float(item.get("elapsed_ms"))
        for item in _read_jsonl(path)
        if isinstance(item, dict) and item.get("elapsed_ms") is not None
    ]
    values = [value for value in values if value >= 0]
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    return {
        "avg_ms": sum(ordered) / len(ordered),
        "p50_ms": _percentile(ordered, 50),
        "p95_ms": _percentile(ordered, 95),
        "max_ms": ordered[-1],
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _compact_run_metadata(run_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": run_payload.get("created_at"),
        "benchmark": run_payload.get("benchmark"),
        "case_count": run_payload.get("case_count"),
        "elapsed_ms": run_payload.get("elapsed_ms"),
        "successful_experiments": run_payload.get("successful_experiments", []),
        "errors": run_payload.get("errors", {}),
        "plans": [
            {
                "name": plan.get("name"),
                "collection_name": plan.get("collection_name"),
                "index_fingerprint": plan.get("index_fingerprint"),
                "reuses_index_of": plan.get("reuses_index_of"),
                "configured_metrics": _plan_metrics(plan),
            }
            for plan in run_payload.get("plans", [])
            if isinstance(plan, dict)
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read JSON file %s: %s", path, exc)
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError as exc:
        logger.warning("Failed to read JSONL file %s: %s", path, exc)
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        logger.warning("Failed to read CSV file %s: %s", path, exc)
        return []


def _safe_filename(name: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "experiment"


def _to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_display_number(value: Any) -> Any:
    if value in (None, ""):
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return value


def _format_ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.1f} ms"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _save_to_history(report: dict[str, Any]) -> None:
    """Append an evaluation report to the history file."""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> list[dict[str, Any]]:
    """Load evaluation history from JSONL file."""
    if not EVAL_HISTORY_PATH.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries


def _load_golden_queries(golden_path: Path) -> list[dict[str, Any]]:
    """Load test cases from golden test set for display in the UI.

    Returns list of dicts with at least 'query' and optionally
    'reference_answer' keys.
    """
    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])
