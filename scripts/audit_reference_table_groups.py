"""Audit FinanceBench reference tables against current MinerU table grouping."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.libs.loader.mineru_artifact_normalizer import MinerUArtifactNormalizer
from src.libs.loader.table_continuation_merger import TableContinuationMerger
from src.libs.splitter.html_table_parser import HTMLTableParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "financebench"
    / "data"
    / "financebench_open_source.jsonl"
)
DEFAULT_TARGETS = (
    PROJECT_ROOT
    / "config"
    / "evaluation"
    / "financebench_30_table_targets.v1.json"
)
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "config"
    / "evaluation"
    / "financebench_30_atomic_facts.v1.jsonl"
)
DEFAULT_PARSED = PROJECT_ROOT / "data" / "parsed"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "analysis" / "reference_table_group_audit_current.md"
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&'’-]*")
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?%?"
    r"(?![A-Za-z0-9])"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _words(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(html.unescape(text))
        if len(token) > 2
    }


def _numbers(text: str) -> set[str]:
    values: set[str] = set()
    normalized_text = re.sub(r"\bFY(?=\d)", "", html.unescape(text), flags=re.I)
    for match in _NUMBER.findall(normalized_text):
        normalized = match.replace(",", "").replace(" ", "")
        suffix = "%" if normalized.endswith("%") else ""
        numeric = normalized[:-1] if suffix else normalized
        if numeric.startswith("(") and numeric.endswith(")"):
            numeric = numeric[1:-1]
        numeric = numeric.removeprefix("-")
        normalized = numeric + suffix
        values.add(normalized.casefold())
    return values


def _coverage(required: set[str], candidate: set[str]) -> float:
    return len(required & candidate) / len(required) if required else 1.0


def _facts(annotation: dict[str, Any], evidence_index: int) -> list[str]:
    groups = annotation.get("evidence_groups", [])
    if not 0 < evidence_index <= len(groups):
        return []
    return [
        str(item.get("fact", ""))
        for item in groups[evidence_index - 1].get("evidence_facts", [])
        if item.get("fact")
    ]


def _cache_by_document(parsed_dir: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for path in parsed_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = payload.get("document", {})
        metadata = document.get("metadata", {})
        source_path = metadata.get("source_path")
        artifact = metadata.get("parsed_source_artifact")
        if source_path and isinstance(artifact, dict):
            cache[Path(str(source_path)).stem] = artifact
    return cache


def _table_blocks(page: Any) -> list[Any]:
    return [block for block in page.blocks if block.type == "table"]


def build_report(
    *,
    cases_path: Path,
    targets_path: Path,
    annotations_path: Path,
    parsed_dir: Path,
) -> str:
    cases = {row["financebench_id"]: row for row in _jsonl(cases_path)}
    annotations = {
        row["case_id"]: row for row in _jsonl(annotations_path)
    }
    targets = json.loads(targets_path.read_text(encoding="utf-8"))["targets"]
    cache = _cache_by_document(parsed_dir)
    normalizer = MinerUArtifactNormalizer()
    merger = TableContinuationMerger()
    parser = HTMLTableParser()
    document_cache: dict[str, tuple[Any, Any]] = {}
    results: list[dict[str, Any]] = []

    for target in targets:
        case_id = str(target["case_id"])
        evidence_index = int(target["evidence_index"])
        document_name = str(target["document_name"])
        page_number = int(target["page_number"])
        case = cases[case_id]
        evidence = case["evidence"][evidence_index - 1]
        facts = _facts(annotations[case_id], evidence_index)
        reference_text = "\n".join([str(evidence["evidence_text"]), *facts])

        if document_name not in document_cache:
            parsed = normalizer.normalize(cache[document_name])
            document_cache[document_name] = (parsed, merger.process(parsed))
        parsed, grouped = document_cache[document_name]
        before_page = next(
            page for page in parsed.pages if page.page_index == page_number - 1
        )
        after_page = next(
            page for page in grouped.pages if page.page_index == page_number - 1
        )
        before_tables = _table_blocks(before_page)
        after_tables = _table_blocks(after_page)

        required_words = _words(reference_text)
        required_numbers = _numbers("\n".join(facts)) or _numbers(reference_text)
        ranked: list[tuple[float, float, Any, str]] = []
        for block in after_tables:
            visible = parser.visible_text(block.content)
            candidate_text = "\n".join(
                [*block.caption, visible, *block.footnotes]
            )
            number_coverage = _coverage(required_numbers, _numbers(candidate_text))
            word_coverage = _coverage(required_words, _words(candidate_text))
            ranked.append((number_coverage, word_coverage, block, candidate_text))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        number_coverage, word_coverage, best, visible = ranked[0]
        unit_count = int(best.metadata.get("unit_count", 1))
        results.append(
            {
                "case_id": case_id,
                "evidence_index": evidence_index,
                "document_name": document_name,
                "page_number": page_number,
                "before_tables": len(before_tables),
                "after_tables": len(after_tables),
                "unit_count": unit_count,
                "group_id": best.metadata.get("table_group_id", best.block_id),
                "captions": [
                    str(unit.get("caption") or "")
                    for unit in best.metadata.get("units", [])
                    if isinstance(unit, dict)
                ] or list(best.caption),
                "fact_number_coverage": number_coverage,
                "reference_word_coverage": word_coverage,
                "required_numbers": sorted(required_numbers),
                "missing_numbers": sorted(required_numbers - _numbers(visible)),
                "grouped": unit_count > 1,
            }
        )

    full_numeric = sum(item["fact_number_coverage"] == 1 for item in results)
    grouped = sum(item["grouped"] for item in results)
    flagged = [item for item in results if item["fact_number_coverage"] < 1]
    lines = [
        "# Current FinanceBench Reference Table Group Audit",
        "",
        "## Summary",
        "",
        f"- Direct-table evidence records: **{len(results)}**",
        f"- Best groups with complete Atomic-Fact numeric coverage: **{full_numeric}/{len(results)}**",
        f"- Evidence records whose best result is a multi-unit Table Group: **{grouped}/{len(results)}**",
        f"- Automatically flagged records: **{len(flagged)}**",
        "",
        "Numeric coverage is based on fixed Atomic Facts. Word coverage is diagnostic only and is not treated as OCR correctness.",
        "",
        "## Results",
        "",
        "| Case | Evidence | Document | Page | Tables before→after | Units | Numeric coverage | Word coverage | Missing numbers |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        missing = ", ".join(item["missing_numbers"]) or "—"
        lines.append(
            (
                "| `{case_id}` | {evidence_index} | {document_name} | "
                "{page_number} | {before_tables}→{after_tables} | {unit_count} | "
                "{fact_number_coverage:.1%} | {reference_word_coverage:.1%} | "
                "{missing} |"
            ).format(**item, missing=missing)
        )

    lines.extend(["", "## Multi-unit Groups", ""])
    for item in results:
        if not item["grouped"]:
            continue
        captions = [caption for caption in item["captions"] if caption]
        lines.extend(
            [
                f"### {item['case_id']} / evidence {item['evidence_index']}",
                "",
                f"- Document/page: `{item['document_name']}` / {item['page_number']}",
                f"- Tables: {item['before_tables']} → {item['after_tables']}",
                f"- Unit count: {item['unit_count']}",
                f"- Captions: {json.dumps(captions, ensure_ascii=False)}",
                f"- Required numbers: {', '.join(item['required_numbers']) or 'none'}",
                f"- Missing numbers: {', '.join(item['missing_numbers']) or 'none'}",
                "",
            ]
        )

    lines.extend(["## Flagged For Manual Review", ""])
    if not flagged:
        lines.append("No records were automatically flagged.")
    for item in flagged:
        lines.append(
            f"- `{item['case_id']}` evidence {item['evidence_index']}: "
            f"numeric={item['fact_number_coverage']:.1%}, "
            f"word={item['reference_word_coverage']:.1%}, "
            f"missing={item['missing_numbers']}"
        )
    lines.extend(
        [
            "",
            "## Manual Review Notes",
            "",
            "- `financebench_id_00941`: the table contains all three debt-security "
            "names, symbols, maturities, and exchanges. The missing `2023` is the "
            "reporting period supplied by the question/document, not a table cell.",
            "- `financebench_id_01902`: the category percentages are prose on the "
            "reference page, not contents of that page's financial table. This record "
            "was misclassified by the historical table-target list and should be "
            "audited as text evidence.",
            "- The former major split-table cases are repaired by grouping: "
            "`financebench_id_00799` is 3→1 units, `financebench_id_04660` is "
            "2→1, and `financebench_id_04171` is 2→1; all required Atomic-Fact "
            "numbers are present in each resulting group.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_report(
        cases_path=args.cases,
        targets_path=args.targets,
        annotations_path=args.annotations,
        parsed_dir=args.parsed_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
