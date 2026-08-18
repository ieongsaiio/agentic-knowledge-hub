"""Judge summaries produced by the repeated largest-table latency benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.benchmark_table_summary_latency import (
    _find_table,
    _stats,
    _table_source,
    _usage,
)
from src.core.settings import load_settings, resolve_path
from src.observability.evaluation.table_summary_judge import LLMTableSummaryJudge


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--summaries",
        default="data/analysis/table_summary_latency_largest_10.json",
    )
    parser.add_argument(
        "--output",
        default="data/analysis/table_summary_judgements_largest_10.json",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_settings(resolve_path(args.config))
    summaries_path = resolve_path(args.summaries)
    benchmark = json.loads(summaries_path.read_text(encoding="utf-8"))
    chunks_path = Path(benchmark["source"]["chunks_file"])
    source_record = _find_table(chunks_path, str(benchmark["source"]["chunk_id"]))
    table_source = _table_source(str(source_record["text"]))
    source = benchmark["source"]
    metadata = source_record.get("metadata") or {}
    raw_header_path = metadata.get("header_path") or []
    section_path = (
        " > ".join(str(item) for item in raw_header_path)
        if isinstance(raw_header_path, list)
        else str(raw_header_path)
    )
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    judge = LLMTableSummaryJudge(
        settings,
        max_output_tokens=args.max_output_tokens,
    )

    records: list[dict[str, Any]] = []
    for run in benchmark["runs"]:
        attempts = 0
        total_latency = 0.0
        while True:
            attempts += 1
            started = time.perf_counter()
            try:
                judgement, response = judge.judge_with_response(
                    table_source,
                    str(run["summary"]),
                    document_name=str(source["document_name"]),
                    page_range=f"{source['page_start']}-{source['page_end']}",
                    section_path=section_path or None,
                    table_title=str(metadata.get("table_title") or "") or None,
                    previous_context=str(metadata.get("previous_context") or "") or None,
                    footnotes=[
                        item
                        for item in metadata.get("vision_footnotes", [])
                        if isinstance(item, str)
                    ],
                    next_context=str(metadata.get("next_context") or "") or None,
                    table_unit_count=int(source["table_unit_count"]),
                )
                total_latency += time.perf_counter() - started
                break
            except Exception as exc:
                total_latency += time.perf_counter() - started
                if attempts >= args.max_attempts:
                    raise
                print(
                    f"summary={run['run']} attempt={attempts} failed: "
                    f"{type(exc).__name__}: {exc}; retrying"
                )
        latency = total_latency
        result = {
            "summary_run": int(run["run"]),
            "summary_sha256": run["summary_sha256"],
            "summary_chars": run["summary_chars"],
            "judge_latency_seconds": round(latency, 6),
            "judge_attempts": attempts,
            "judge_model": getattr(response, "model", None),
            "judge_usage": _usage(response),
            **asdict(judgement),
            "raw_score": judgement.raw_score,
            "overall_score": judgement.overall_score,
            "quality_level": judgement.quality_level,
        }
        records.append(result)
        print(
            f"summary={run['run']}/{len(benchmark['runs'])} "
            f"score={judgement.overall_score:.2f} "
            f"level={judgement.quality_level} latency={latency:.3f}s"
        )

    score_fields = (
        "faithfulness",
        "key_information_coverage",
        "retrieval_utility",
        "trend_relationship_quality",
        "conciseness_clarity",
        "raw_score",
        "overall_score",
        "judge_latency_seconds",
    )
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": settings.llm.model,
        "judge_provider": settings.llm.provider,
        "judge_api_mode": settings.llm.api_mode,
        "rubric": "config/prompts/table_summary_judge.txt",
        "summaries_file": str(summaries_path),
        "source": source,
        "sample_size": len(records),
        "statistics": {
            field: _stats([record[field] for record in records])
            for field in score_fields
        },
        "quality_level_counts": {
            level: sum(record["quality_level"] == level for record in records)
            for level in ("excellent", "good", "usable", "poor", "reject")
        },
        "records": records,
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
