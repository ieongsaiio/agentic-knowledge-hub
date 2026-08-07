#!/usr/bin/env python
"""OCR the documents referenced by a deterministic benchmark sample."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "analysis" / "financebench_ocr_sample.json"
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR only the unique PDFs referenced by a benchmark sample."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pdf_page_count(path: Path) -> int:
    import fitz

    with fitz.open(path) as pdf:
        return pdf.page_count


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# FinanceBench OCR Sample Report",
        "",
        "本次只执行 PDF OCR 与 parsed cache 持久化，不执行 Chunking、Embedding、"
        "BM25、Chroma、Query 或 Evaluation。",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Benchmark cases | {summary['benchmark_cases']} |",
        f"| Unique PDFs | {summary['unique_pdfs']} |",
        f"| Successful PDFs | {summary['successful_pdfs']} |",
        f"| Failed PDFs | {summary['failed_pdfs']} |",
        f"| PDF pages | {summary['pdf_pages']} |",
        f"| OCR pages | {summary['ocr_pages']} |",
        f"| Markdown characters | {summary['markdown_characters']} |",
        f"| Total wall time (s) | {summary['wall_seconds']:.3f} |",
        f"| Sum of API OCR time (s) | {summary['sum_api_elapsed_seconds']:.3f} |",
        f"| Sum of per-PDF wall time (s) | {summary['sum_pdf_wall_seconds']:.3f} |",
        f"| Mean wall time per PDF (s) | {summary['mean_pdf_wall_seconds']:.3f} |",
        f"| Mean wall time per page (s) | {summary['mean_page_wall_seconds']:.3f} |",
        "",
        "## Per PDF",
        "",
        "| PDF | Cases | Pages | OCR pages | API time (s) | Client time (s) | s/page | Chars | Cache hit | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["documents"]:
        lines.append(
            "| {name} | {case_count} | {pdf_pages} | {ocr_pages} | "
            "{api_elapsed_seconds:.3f} | {wall_seconds:.3f} | "
            "{seconds_per_page:.3f} | {markdown_characters} | "
            "{cache_hit} | {status} |".format(**item)
        )
    failures = [item for item in report["documents"] if item["status"] != "ok"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.append(f"- `{item['name']}`: {item.get('error', 'unknown error')}")
    return "\n".join(lines) + "\n"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text(_markdown_report(report), encoding="utf-8")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than zero")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay cannot be negative")

    from src.core.settings import load_settings, resolve_path
    from src.libs.benchmark.benchmark_factory import BenchmarkFactory
    from src.libs.loader.loader_factory import LoaderFactory
    from src.libs.loader.parsed_document_cache import ParsedDocumentCache

    settings = load_settings(args.config)
    benchmark_settings = settings.evaluation.benchmark
    if benchmark_settings is None:
        raise ValueError("evaluation.benchmark is not configured")
    effective_settings = replace(
        settings,
        evaluation=replace(
            settings.evaluation,
            benchmark=replace(benchmark_settings, sample_size=args.sample_size),
        ),
    )
    benchmark = BenchmarkFactory.create(effective_settings)
    cases = benchmark.load_cases()

    case_counts = Counter(
        evidence.document_name
        for case in cases
        for evidence in case.evidences
    )
    requested_names = list(dict.fromkeys(case_counts))
    pdf_lookup = {
        path.stem.casefold(): path
        for path in benchmark.pdf_dir.rglob("*.pdf")
        if path.is_file()
    }
    missing = [name for name in requested_names if name.casefold() not in pdf_lookup]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark PDFs: " + ", ".join(sorted(missing))
        )

    loader = LoaderFactory.create(effective_settings)
    if getattr(loader, "backend", None) != "api":
        raise ValueError("This batch runner requires the Paddle OCR API backend")
    parsed_dir = resolve_path(effective_settings.ingestion.loader.parsed_dir)
    cache = ParsedDocumentCache(parsed_dir, loader.cache_config())
    report_path = Path(args.report).resolve()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    documents: list[dict[str, Any] | None] = [None] * len(requested_names)

    async def process(index: int, name: str) -> None:
        path = pdf_lookup[name.casefold()]
        pdf_pages = await asyncio.to_thread(_pdf_page_count, path)
        file_hash = await asyncio.to_thread(_sha256, path)
        async with semaphore:
            item_started = time.perf_counter()
            print(
                f"[{index + 1}/{len(requested_names)}] OCR start: "
                f"{path.name} ({pdf_pages} pages)",
                flush=True,
            )
            try:
                document = None
                cache_hit = False
                document = await asyncio.to_thread(cache.get, file_hash, path)
                if document is not None:
                    cache_hit = True
                    print(
                        f"[{index + 1}/{len(requested_names)}] cache hit: {path.name}",
                        flush=True,
                    )
                else:
                    for attempt in range(args.max_retries + 1):
                        try:
                            document = await loader.aload(path)
                            break
                        except Exception as exc:
                            message = str(exc)
                            retryable = (
                                "10010" in message
                                or "status 500" in message
                                or "status code 500" in message
                                or "status 504" in message
                                or "status code 504" in message
                                or "timeout" in type(exc).__name__.lower()
                                or "timed out" in message.lower()
                            )
                            if not retryable or attempt >= args.max_retries:
                                raise
                            delay = args.retry_delay * min(attempt + 1, 4)
                            print(
                                f"[{index + 1}/{len(requested_names)}] retry "
                                f"{attempt + 1}/{args.max_retries} in {delay:.0f}s: "
                                f"{path.name}",
                                flush=True,
                            )
                            await asyncio.sleep(delay)
                if document is None:
                    raise RuntimeError("PaddleOCR returned no document")
                cache_path = (
                    cache.cache_path(file_hash)
                    if cache_hit
                    else await asyncio.to_thread(cache.put, file_hash, document)
                )
                wall_seconds = time.perf_counter() - item_started
                artifact = document.metadata.get("parsed_artifact", {})
                ocr_pages = int(document.metadata.get("page_count", 0))
                api_elapsed_seconds = float(artifact.get("elapsed_seconds", 0.0))
                item = {
                    "name": name,
                    "file": str(path),
                    "case_count": case_counts[name],
                    "file_bytes": path.stat().st_size,
                    "pdf_pages": pdf_pages,
                    "ocr_pages": ocr_pages,
                    "wall_seconds": wall_seconds,
                    "api_elapsed_seconds": api_elapsed_seconds,
                    "seconds_per_page": (
                        api_elapsed_seconds / pdf_pages if pdf_pages else 0.0
                    ),
                    "markdown_characters": len(document.text),
                    "page_spans": len(document.metadata.get("page_spans", [])),
                    "cache_file": str(cache_path),
                    "cache_hit": cache_hit,
                    "status": "ok",
                }
                print(
                    f"[{index + 1}/{len(requested_names)}] OCR done: "
                    f"{path.name}, {wall_seconds:.1f}s, {len(document.text)} chars",
                    flush=True,
                )
            except Exception as exc:
                wall_seconds = time.perf_counter() - item_started
                item = {
                    "name": name,
                    "file": str(path),
                    "case_count": case_counts[name],
                    "file_bytes": path.stat().st_size,
                    "pdf_pages": pdf_pages,
                    "ocr_pages": 0,
                    "wall_seconds": wall_seconds,
                    "api_elapsed_seconds": 0.0,
                    "seconds_per_page": wall_seconds / pdf_pages if pdf_pages else 0.0,
                    "markdown_characters": 0,
                    "page_spans": 0,
                    "cache_file": "",
                    "cache_hit": False,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"[{index + 1}/{len(requested_names)}] OCR failed: "
                    f"{path.name}: {item['error']}",
                    flush=True,
                )
            documents[index] = item

    await asyncio.gather(
        *(process(index, name) for index, name in enumerate(requested_names))
    )
    finished_at = datetime.now(timezone.utc)
    complete_documents = [item for item in documents if item is not None]
    successes = [item for item in complete_documents if item["status"] == "ok"]
    total_pages = sum(item["pdf_pages"] for item in complete_documents)
    sum_pdf_wall = sum(item["wall_seconds"] for item in complete_documents)
    report = {
        "configuration": {
            "provider": benchmark_settings.provider,
            "split": benchmark_settings.split,
            "seed": benchmark_settings.seed,
            "sample_size": args.sample_size,
            "loader": effective_settings.ingestion.loader.provider,
            "backend": getattr(loader, "backend", None),
            "model": getattr(loader, "api_config", {}).get("model"),
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "retry_delay_seconds": args.retry_delay,
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "sample_case_ids": [case.case_id for case in cases],
        "summary": {
            "benchmark_cases": len(cases),
            "unique_pdfs": len(complete_documents),
            "successful_pdfs": len(successes),
            "failed_pdfs": len(complete_documents) - len(successes),
            "pdf_pages": total_pages,
            "ocr_pages": sum(item["ocr_pages"] for item in successes),
            "markdown_characters": sum(
                item["markdown_characters"] for item in successes
            ),
            "sum_api_elapsed_seconds": sum(
                item["api_elapsed_seconds"] for item in successes
            ),
            "wall_seconds": time.perf_counter() - started,
            "sum_pdf_wall_seconds": sum_pdf_wall,
            "mean_pdf_wall_seconds": (
                sum_pdf_wall / len(complete_documents) if complete_documents else 0.0
            ),
            "mean_page_wall_seconds": (
                sum_pdf_wall / total_pages if total_pages else 0.0
            ),
        },
        "documents": complete_documents,
    }
    _write_report(report_path, report)
    print(f"JSON report: {report_path}", flush=True)
    print(f"Markdown report: {report_path.with_suffix('.md')}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        print(f"OCR benchmark sample failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0 if report["summary"]["failed_pdfs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
