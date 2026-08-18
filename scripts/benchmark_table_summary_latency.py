"""Benchmark repeated table-summary calls against the largest exported table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.settings import load_settings, resolve_path
from src.libs.splitter.structured_markdown_splitter import _LLMTableSummarizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--chunks",
        default="tmp/financebench_30_chunking_current/chunks.jsonl",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--chunk-id",
        help="Benchmark this exact exported chunk instead of the largest table.",
    )
    parser.add_argument(
        "--output",
        default="data/analysis/table_summary_latency_largest_10.json",
    )
    return parser.parse_args()


def _largest_table(path: Path) -> dict[str, Any]:
    largest: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("chunk_type") != "table_group":
                continue
            if largest is None or int(record["char_count"]) > int(
                largest["char_count"]
            ):
                largest = record
    if largest is None:
        raise RuntimeError(f"No table_group chunks found in {path}")
    return largest


def _find_table(path: Path, chunk_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("chunk_id") == chunk_id:
                if record.get("chunk_type") != "table_group":
                    raise ValueError(f"Chunk {chunk_id} is not a table_group")
                return record
    raise ValueError(f"Chunk not found: {chunk_id}")


def _table_source(text: str) -> str:
    start = text.casefold().find("<table")
    end = text.casefold().rfind("</table>")
    if start < 0 or end < start:
        return text
    return text[start : end + len("</table>")]


def _nested_int(data: Any, *paths: tuple[str, ...]) -> int | None:
    for path in paths:
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if isinstance(current, int) and not isinstance(current, bool):
                return current
    return None


def _usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None) or {}
    raw = getattr(response, "raw_response", None) or {}
    raw_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    sources = [usage, raw_usage]

    def find(*paths: tuple[str, ...]) -> int | None:
        for source in sources:
            value = _nested_int(source, *paths)
            if value is not None:
                return value
        return None

    input_tokens = find(("prompt_tokens",), ("input_tokens",))
    output_tokens = find(("completion_tokens",), ("output_tokens",))
    total_tokens = find(("total_tokens",))
    cached_tokens = find(
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("cached_input_tokens",),
        ("cached_tokens",),
    )
    reasoning_tokens = find(
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("reasoning_output_tokens",),
    )
    uncached_tokens = (
        max(0, input_tokens - cached_tokens)
        if input_tokens is not None and cached_tokens is not None
        else None
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        "reasoning_output_tokens": reasoning_tokens,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stats(values: list[float | int | None]) -> dict[str, float | int | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"count": 0, "min": None, "avg": None, "p50": None, "p90": None,
                "p95": None, "p99": None, "max": None, "stddev": None}
    return {
        "count": len(present),
        "min": round(min(present), 3),
        "avg": round(statistics.fmean(present), 3),
        "p50": round(_percentile(present, 0.50), 3),
        "p90": round(_percentile(present, 0.90), 3),
        "p95": round(_percentile(present, 0.95), 3),
        "p99": round(_percentile(present, 0.99), 3),
        "max": round(max(present), 3),
        "stddev": round(statistics.pstdev(present), 3),
    }


def main() -> int:
    args = _parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    settings = load_settings(resolve_path(args.config))
    source_path = resolve_path(args.chunks)
    record = (
        _find_table(source_path, args.chunk_id)
        if args.chunk_id
        else _largest_table(source_path)
    )
    metadata = record.get("metadata") or {}
    table_text = _table_source(str(record["text"]))
    summarizer = _LLMTableSummarizer(settings)
    header_path = metadata.get("header_path") or []
    section_path = (
        " > ".join(str(item) for item in header_path)
        if isinstance(header_path, list)
        else str(header_path)
    )
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    page_range = (
        str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    )
    units = [item for item in metadata.get("units", []) if isinstance(item, dict)]
    unit_count = max(1, int(metadata.get("unit_count") or 1))

    runs: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        started = time.perf_counter()
        summary, response = summarizer.summarize_with_response(
            table_text,
            table_title=str(metadata.get("table_title") or "") or None,
            footnotes=[
                item
                for item in metadata.get("vision_footnotes", [])
                if isinstance(item, str)
            ],
            previous_context=str(metadata.get("previous_context") or "") or None,
            next_context=str(metadata.get("next_context") or "") or None,
            document_name=str(record.get("document_name") or "") or None,
            section_path=section_path or None,
            page_range=page_range if page_start is not None else None,
            table_units=units,
            table_unit_count=unit_count,
        )
        latency = time.perf_counter() - started
        run = {
            "run": run_index,
            "latency_seconds": round(latency, 6),
            **_usage(response),
            "summary_chars": len(summary),
            "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            "summary": summary,
            "response_model": getattr(response, "model", None),
        }
        runs.append(run)
        print(
            f"run={run_index}/{args.runs} latency={latency:.3f}s "
            f"input={run['input_tokens']} output={run['output_tokens']} "
            f"cached={run['cached_input_tokens']}"
        )

    metric_names = (
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "reasoning_output_tokens",
        "summary_chars",
    )
    summary_stats = {
        metric: _stats([run.get(metric) for run in runs]) for metric in metric_names
    }
    unique_summaries = len({run["summary_sha256"] for run in runs})
    table_summary_llm = settings.ingestion.structured_chunking["table_summary"]["llm"]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(runs),
        "percentile_method": "linear interpolation, position=(n-1)*q",
        "warning": "P95 and P99 from 10 samples are exploratory, not stable tail estimates.",
        "model": table_summary_llm.get("model", settings.llm.model),
        "provider": table_summary_llm.get("provider", settings.llm.provider),
        "api_mode": table_summary_llm.get("api_mode", settings.llm.api_mode),
        "source": {
            "chunks_file": str(source_path),
            "document_name": record.get("document_name"),
            "chunk_id": record.get("chunk_id"),
            "chunk_chars": record.get("char_count"),
            "table_source_chars": len(table_text),
            "page_start": page_start,
            "page_end": page_end,
            "table_unit_count": unit_count,
        },
        "summary_variants": unique_summaries,
        "statistics": summary_stats,
        "runs": runs,
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
