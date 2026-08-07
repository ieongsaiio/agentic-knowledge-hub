"""Render fixed FinanceBench atomic-fact annotations as a human-reviewable Markdown file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.libs.benchmark.financebench_benchmark import FinanceBenchBenchmark
from src.observability.evaluation.atomic_fact_annotations import (
    load_atomic_fact_annotations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "config" / "evaluation" / "financebench_30_atomic_facts.v1.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "analysis" / "financebench_30_atomic_facts_review.md"


def _blockquote(value: str) -> list[str]:
    lines = value.strip().splitlines() or [""]
    return [f"> {line}" if line else ">" for line in lines]


def render_report(annotations_path: Path, output_path: Path, data_dir: str) -> None:
    benchmark = FinanceBenchBenchmark(
        {
            "data_dir": data_dir,
            "auto_download": False,
            "sample_size": 30,
            "seed": 42,
        }
    )
    cases = benchmark.load_cases()
    annotations = load_atomic_fact_annotations(annotations_path, cases=cases)

    fact_count = sum(
        len(group.evidence_facts)
        for annotation in annotations
        for group in annotation.evidence_groups
    )
    lines = [
        "# FinanceBench 30 Atomic Supporting Facts",
        "",
        "This document is a human-review view of the fixed retrieval ground truth.",
        "The sample is selected with `sample_size=30` and `seed=42`.",
        "",
        f"- Questions: {len(cases)}",
        f"- Evidence groups: {sum(len(item.evidence_groups) for item in annotations)}",
        f"- Atomic facts: {fact_count}",
        "- Schema version: `1.0`",
        "",
        "Atomic facts contain only retrievable evidence. Formulas, calculations, comparisons,",
        "and business judgements are intentionally excluded from these facts.",
        "",
        "---",
        "",
    ]

    for display_index, (case, annotation) in enumerate(
        zip(cases, annotations, strict=True), start=1
    ):
        lines.extend(
            [
                f"## {display_index:02d}. {case.case_id}",
                "",
                "**Question**",
                "",
                *_blockquote(case.query),
                "",
                "**Reference Answer**",
                "",
                *_blockquote(case.reference_answer),
                "",
                "**Atomic Supporting Facts**",
                "",
            ]
        )
        for group, evidence in zip(
            annotation.evidence_groups, case.evidences, strict=True
        ):
            lines.extend(
                [
                    f"### {group.evidence_id}",
                    "",
                    f"Source: `{evidence.document_name}`, PDF page `{evidence.page_number}`",
                    "",
                ]
            )
            for fact in group.evidence_facts:
                lines.append(f"- `{fact.fact_id}`: {fact.fact}")
            lines.append("")

        if annotation.reasoning_requirements:
            lines.extend(["**Excluded Reasoning Requirements**", ""])
            for requirement in annotation.reasoning_requirements:
                lines.append(
                    f"- `{requirement.reasoning_id}`: {requirement.requirement}"
                )
            lines.append("")
        if annotation.annotation_notes:
            lines.extend(["**Annotation Notes**", ""])
            lines.extend(f"- {note}" for note in annotation.annotation_notes)
            lines.append("")
        lines.extend(["---", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", default="./data/benchmarks/financebench")
    args = parser.parse_args()
    render_report(args.annotations, args.output, args.data_dir)
    print(f"atomic-fact review report: {args.output}")


if __name__ == "__main__":
    main()
