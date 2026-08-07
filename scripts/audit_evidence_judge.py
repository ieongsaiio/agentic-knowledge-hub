#!/usr/bin/env python
"""Re-run only the evidence judge against persisted benchmark retrieval results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import load_settings  # noqa: E402
from src.libs.benchmark.benchmark_factory import BenchmarkFactory  # noqa: E402
from src.observability.evaluation.benchmark_metrics import (  # noqa: E402
    evidence_candidate_ranks,
    normalize_document_name,
)
from src.observability.evaluation.evidence_judge import (  # noqa: E402
    LLMEvidenceJudge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit atomic-fact evidence judgements without re-running retrieval."
    )
    parser.add_argument("--cases", required=True, help="Persisted benchmark cases JSONL.")
    parser.add_argument("--output", required=True, help="Audit output JSONL path.")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--eligibility",
        choices=("page", "document", "all"),
        default="page",
        help="Candidate gate applied before judging (default: page).",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case IDs to audit.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        results.append(value)
    return results


def _document_candidate_ranks(case: Any, chunks: list[Any]) -> tuple[tuple[int, ...], ...]:
    ranks: list[tuple[int, ...]] = []
    for evidence in case.evidences:
        expected = normalize_document_name(evidence.document_name)
        matched: list[int] = []
        for rank, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            source = metadata.get("source_path", metadata.get("source", ""))
            if normalize_document_name(source) == expected:
                matched.append(rank)
        ranks.append(tuple(matched))
    return tuple(ranks)


def _eligible_ranks(case: Any, chunks: list[Any], mode: str) -> tuple[tuple[int, ...], ...]:
    if mode == "page":
        return evidence_candidate_ranks(case, chunks)
    if mode == "document":
        return _document_candidate_ranks(case, chunks)
    all_ranks = tuple(range(1, len(chunks) + 1))
    return tuple(all_ranks for _ in case.evidences)


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    benchmark = BenchmarkFactory.create(settings)
    cases = {case.case_id: case for case in benchmark.prepare()}
    persisted = _load_results(Path(args.cases))
    selected_ids = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    if selected_ids:
        persisted = [item for item in persisted if str(item.get("case_id", "")) in selected_ids]
    judge = LLMEvidenceJudge(settings, max_output_tokens=args.max_output_tokens)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = {
        str(item.get("case_id", ""))
        for item in (_load_results(output_path) if output_path.is_file() else [])
    }

    if args.retries < 1:
        raise ValueError("retries must be at least one")

    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, result in enumerate(persisted, 1):
            case_id = str(result.get("case_id", ""))
            if case_id in completed_ids:
                print(f"[{index}/{len(persisted)}] {case_id} cached", flush=True)
                continue
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"unknown benchmark case: {case_id}")
            chunks = list(result.get("retrieved_results") or [])
            eligible_ranks = _eligible_ranks(case, chunks, args.eligibility)
            judgement = None
            final_error: Exception | None = None
            for attempt in range(1, args.retries + 1):
                try:
                    judgement = judge.judge(case, chunks, eligible_ranks=eligible_ranks)
                    break
                except Exception as exc:
                    final_error = exc
                    if attempt == args.retries:
                        break
                    print(
                        f"[{index}/{len(persisted)}] {case_id} retry {attempt}: {exc}",
                        flush=True,
                    )
                    time.sleep(float(attempt))
            if judgement is None:
                print(
                    f"[{index}/{len(persisted)}] {case_id} failed: {final_error}",
                    flush=True,
                )
                continue
            fact_groups = judge._fact_groups(case)  # Reproduce the evaluated facts.

            evidence_groups: list[dict[str, Any]] = []
            for evidence_index, match in enumerate(judgement.matches, 1):
                fact_text = dict(fact_groups[evidence_index - 1])
                facts = [
                    {
                        "fact_id": fact.fact_id,
                        "fact": fact_text[fact.fact_id],
                        "supported": fact.supported,
                        "supporting_ranks": list(fact.supporting_ranks),
                        "relevant_ranks": list(fact.relevant_ranks),
                    }
                    for fact in match.fact_matches
                ]
                supporting_ranks = sorted(
                    {
                        rank
                        for fact in match.fact_matches
                        for rank in fact.supporting_ranks
                    }
                )
                evidence_groups.append(
                    {
                        "evidence_index": evidence_index,
                        "eligible_ranks": list(eligible_ranks[evidence_index - 1]),
                        "complete": match.first_matching_rank is not None,
                        "completion_rank": match.first_matching_rank,
                        "supporting_ranks": supporting_ranks,
                        "uses_multiple_chunks": len(supporting_ranks) > 1,
                        "facts": facts,
                    }
                )

            audit = {
                "case_id": case_id,
                "query": case.query,
                "reference_answer": case.reference_answer,
                "context_recall": judgement.context_recall,
                "context_relevant_ranks": list(judgement.context_relevant_ranks),
                "evidence_hit_rate": judgement.evidence_hit_rate,
                "evidence_groups": evidence_groups,
            }
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(persisted)}] {case_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
