"""Audit table-continuation merging against the complete parsed cache."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage
from src.libs.loader.table_continuation_merger import TableContinuationMerger

DEFAULT_PARSED_DIR = Path("data/parsed")
DEFAULT_OUTPUT_DIR = Path("data/analysis/table_continuation_cache_audit")


def _parsed_document(cache: dict[str, Any]) -> tuple[ParsedDocument, str]:
    document = cache["document"]
    metadata = document["metadata"]
    structure = metadata["parsed_structure"]
    pages: list[ParsedPage] = []
    for raw_page in structure["pages"]:
        blocks = []
        for raw_block in raw_page["blocks"]:
            blocks.append(
                ParsedBlock.from_dict(
                    {
                        key: raw_block[key]
                        for key in (
                            "block_id",
                            "type",
                            "content",
                            "page_index",
                            "bbox",
                            "order",
                            "level",
                            "caption",
                            "footnotes",
                            "images",
                            "metadata",
                        )
                        if key in raw_block
                    }
                )
            )
        pages.append(
            ParsedPage(
                page_index=raw_page["page_index"],
                width=raw_page.get("width"),
                height=raw_page.get("height"),
                blocks=blocks,
            )
        )
    return (
        ParsedDocument(
            schema_version=structure["schema_version"],
            provider=structure["provider"],
            parser_version=structure.get("parser_version"),
            pages=pages,
        ),
        str(metadata.get("source_path", "unknown")),
    )


def _table_rows(table_html: str, merger: TableContinuationMerger) -> list[str]:
    return merger._extract_rows(table_html)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _write_preview(
    path: Path,
    *,
    title: str,
    source_tables: list[ParsedBlock],
    merged: ParsedBlock,
) -> None:
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;margin:16px 0}td,th{border:1px solid #888;padding:4px 7px}.table-section-caption{background:#dbeafe}.table-section-footnote{background:#f3f4f6;font-weight:normal}h2{margin-top:32px}</style>",
        f"<title>{html.escape(title)}</title></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for index, block in enumerate(source_tables, start=1):
        sections.extend(
            [
                f"<h2>Source table {index}</h2>",
                f"<p>Caption: {html.escape(repr(block.caption))}</p>",
                block.content,
            ]
        )
    sections.extend(["<h2>Merged table</h2>", merged.content, "</body></html>"])
    path.write_text("".join(sections), encoding="utf-8")


def run(parsed_dir: Path, output_dir: Path, sample_limit: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    merger = TableContinuationMerger()
    total_tables = 0
    adjacent_pairs = 0
    accepted_pairs = 0
    rejection_reasons: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []

    for cache_path in sorted(parsed_dir.glob("*.json")):
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        document, source_path = _parsed_document(cache)
        for page in document.pages:
            total_tables += sum(block.type == "table" for block in page.blocks)
            ordered = sorted(page.blocks, key=merger._visual_sort_key)
            index = 0
            while index < len(ordered):
                if ordered[index].type != "table":
                    index += 1
                    continue
                group = [ordered[index]]
                assessments = []
                cursor = index
                while cursor + 1 < len(ordered) and ordered[cursor + 1].type == "table":
                    adjacent_pairs += 1
                    assessment = merger.assess(
                        ordered[cursor],
                        ordered[cursor + 1],
                        page_width=page.width,
                        page_height=page.height,
                    )
                    if not assessment.accepted:
                        rejection_reasons.update(assessment.reasons)
                        break
                    accepted_pairs += 1
                    assessments.append(assessment)
                    group.append(ordered[cursor + 1])
                    cursor += 1
                if len(group) > 1:
                    merged = merger._merge_group(group, assessments)
                    source_row_count = sum(
                        len(_table_rows(block.content, merger)) for block in group
                    )
                    inserted_rows = sum(
                        len(block.caption) + len(block.footnotes)
                        for block in group[1:]
                    ) + len(group[0].footnotes)
                    merged_row_count = len(_table_rows(merged.content, merger))
                    candidates.append(
                        {
                            "source_path": source_path,
                            "page": page.page_index + 1,
                            "page_width": page.width,
                            "page_height": page.height,
                            "blocks": group,
                            "merged": merged,
                            "logical_width": assessments[0].left_width,
                            "horizontal_overlap": min(
                                item.horizontal_overlap_ratio for item in assessments
                            ),
                            "vertical_gap": max(
                                item.vertical_gap_ratio for item in assessments
                            ),
                            "source_rows": source_row_count,
                            "inserted_context_rows": inserted_rows,
                            "merged_rows": merged_row_count,
                            "row_count_valid": merged_row_count
                            == source_row_count + inserted_rows,
                        }
                    )
                    index += len(group)
                else:
                    index += 1

    # Prefer diverse documents, widths, and tables that exercise inserted captions.
    candidates.sort(
        key=lambda item: (
            not any(block.caption for block in item["blocks"][1:]),
            -len(item["blocks"]),
            str(item["source_path"]),
            item["page"],
        )
    )
    selected: list[dict[str, Any]] = []
    seen_documents: Counter[str] = Counter()
    seen_widths: Counter[int | None] = Counter()
    for candidate in candidates:
        source = str(candidate["source_path"])
        width = candidate["logical_width"]
        if seen_documents[source] >= 2 or seen_widths[width] >= 4:
            continue
        selected.append(candidate)
        seen_documents[source] += 1
        seen_widths[width] += 1
        if len(selected) >= sample_limit:
            break

    report = [
        "# Parsed Cache Table Continuation Audit",
        "",
        "This audit is offline and uses the existing MinerU parsed cache.",
        "No OCR API or table-child chunking is involved.",
        "",
        "## Corpus statistics",
        "",
        f"- Parsed documents: {len(list(parsed_dir.glob('*.json')))}",
        f"- Table blocks: {total_tables}",
        f"- Visually adjacent table pairs: {adjacent_pairs}",
        f"- Accepted continuation pairs: {accepted_pairs}",
        f"- Accepted logical groups: {len(candidates)}",
        f"- Sampled groups: {len(selected)}",
        "",
        "## Rejection reasons",
        "",
    ]
    for reason, count in rejection_reasons.most_common():
        report.append(f"- `{reason}`: {count}")
    report.extend(["", "## Samples", ""])

    for index, candidate in enumerate(selected, start=1):
        source_name = Path(candidate["source_path"]).name
        preview_name = (
            f"{index:02d}_{_safe_name(Path(source_name).stem)}_"
            f"p{candidate['page']}.html"
        )
        preview_path = output_dir / preview_name
        _write_preview(
            preview_path,
            title=f"{source_name} page {candidate['page']}",
            source_tables=candidate["blocks"],
            merged=candidate["merged"],
        )
        captions = [block.caption for block in candidate["blocks"]]
        report.extend(
            [
                f"### Sample {index}: {source_name}, page {candidate['page']}",
                "",
                f"- Parts: {len(candidate['blocks'])}",
                f"- Logical width: {candidate['logical_width']}",
                f"- Minimum horizontal overlap: {candidate['horizontal_overlap']:.3f}",
                f"- Maximum vertical gap ratio: {candidate['vertical_gap']:.3f}",
                f"- Captions: `{captions!r}`",
                f"- Rows: {candidate['source_rows']} source + "
                f"{candidate['inserted_context_rows']} context = "
                f"{candidate['merged_rows']} merged",
                f"- Row preservation check: `{candidate['row_count_valid']}`",
                f"- Preview: [{preview_name}]({preview_name})",
                "",
            ]
        )

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=16)
    args = parser.parse_args()
    print(run(args.parsed_dir, args.output_dir, args.sample_limit))


if __name__ == "__main__":
    main()
