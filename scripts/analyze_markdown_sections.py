"""Analyze heading-based parent sections from parsed Markdown cache files.

This diagnostic intentionally does not change the ingestion pipeline. It compares
several Markdown heading boundary depths so a parent-section strategy can be
selected from measured document structure before it becomes a splitter provider.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
HTML_TABLE_PATTERN = re.compile(r"(?is)<table\b.*?</table>")
PIPE_TABLE_DELIMITER_PATTERN = re.compile(
    r"(?m)^[ \t]*\|?(?:[ \t]*:?-{3,}:?[ \t]*\|){1,}"
    r"[ \t]*:?-{3,}:?[ \t]*\|?[ \t]*$"
)


@dataclass(frozen=True)
class Section:
    text: str
    start_offset: int
    end_offset: int
    heading_level: int | None
    heading: str | None


@dataclass(frozen=True)
class StrategyStats:
    boundary_max_level: int
    section_count: int
    char_min: int
    char_p25: int
    char_median: int
    char_p75: int
    char_p95: int
    char_max: int
    char_mean: float
    sections_under_100_chars: int
    sections_under_300_chars: int
    heading_coverage: float
    table_count: int
    split_table_count: int
    table_integrity_rate: float


def split_by_heading_depth(text: str, boundary_max_level: int) -> list[Section]:
    """Split Markdown at headings whose level is within the selected depth."""
    if not 1 <= boundary_max_level <= 6:
        raise ValueError("boundary_max_level must be between 1 and 6")
    if not text or not text.strip():
        return []

    headings = [
        match
        for match in HEADING_PATTERN.finditer(text)
        if len(match.group(1)) <= boundary_max_level
    ]
    boundaries = [match.start() for match in headings]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))

    headings_by_start = {match.start(): match for match in headings}
    sections: list[Section] = []
    for start, end in zip(boundaries, boundaries[1:]):
        trimmed_start, trimmed_end = _trim_range(text, start, end)
        if trimmed_start >= trimmed_end:
            continue
        heading_match = headings_by_start.get(start)
        sections.append(
            Section(
                text=text[trimmed_start:trimmed_end],
                start_offset=trimmed_start,
                end_offset=trimmed_end,
                heading_level=(
                    len(heading_match.group(1)) if heading_match else None
                ),
                heading=(
                    heading_match.group(2).strip() if heading_match else None
                ),
            )
        )
    return sections


def analyze_strategy(
    text: str,
    boundary_max_level: int,
) -> tuple[StrategyStats, list[Section]]:
    """Return section-size and structural-integrity statistics."""
    sections = split_by_heading_depth(text, boundary_max_level)
    lengths = [len(section.text) for section in sections]
    table_spans = find_table_spans(text)
    split_table_count = sum(
        1 for table_start, table_end in table_spans
        if not any(
            section.start_offset <= table_start
            and section.end_offset >= table_end
            for section in sections
        )
    )
    headings_covered = sum(
        1 for section in sections if section.heading is not None
    )
    stats = StrategyStats(
        boundary_max_level=boundary_max_level,
        section_count=len(sections),
        char_min=min(lengths, default=0),
        char_p25=_percentile(lengths, 0.25),
        char_median=_percentile(lengths, 0.50),
        char_p75=_percentile(lengths, 0.75),
        char_p95=_percentile(lengths, 0.95),
        char_max=max(lengths, default=0),
        char_mean=round(statistics.fmean(lengths), 2) if lengths else 0.0,
        sections_under_100_chars=sum(length < 100 for length in lengths),
        sections_under_300_chars=sum(length < 300 for length in lengths),
        heading_coverage=round(
            headings_covered / len(sections), 4
        ) if sections else 0.0,
        table_count=len(table_spans),
        split_table_count=split_table_count,
        table_integrity_rate=round(
            (len(table_spans) - split_table_count) / len(table_spans),
            4,
        ) if table_spans else 1.0,
    )
    return stats, sections


def find_table_spans(text: str) -> list[tuple[int, int]]:
    """Find HTML and pipe-table spans without treating headings as table text."""
    spans = [(match.start(), match.end()) for match in HTML_TABLE_PATTERN.finditer(text)]
    lines = list(re.finditer(r"(?m)^.*(?:\n|$)", text))
    delimiter_lines = [
        index
        for index, line_match in enumerate(lines)
        if PIPE_TABLE_DELIMITER_PATTERN.fullmatch(
            line_match.group(0).rstrip("\r\n")
        )
    ]
    for delimiter_index in delimiter_lines:
        start_index = delimiter_index
        end_index = delimiter_index
        while start_index > 0 and "|" in lines[start_index - 1].group(0):
            start_index -= 1
        while end_index + 1 < len(lines) and "|" in lines[end_index + 1].group(0):
            end_index += 1
        spans.append((lines[start_index].start(), lines[end_index].end()))
    return _merge_spans(sorted(spans))


def load_parsed_document(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    document = payload.get("document", {})
    text = document.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{path} does not contain document.text")
    metadata = document.get("metadata", {})
    source = metadata.get("source_path") if isinstance(metadata, dict) else None
    return str(source or path.name), text


def analyze_files(
    paths: Iterable[Path],
    levels: Iterable[int],
) -> dict[str, object]:
    documents: list[dict[str, object]] = []
    for path in paths:
        source, text = load_parsed_document(path)
        strategies = []
        for level in levels:
            stats, sections = analyze_strategy(text, level)
            strategies.append(
                {
                    "stats": asdict(stats),
                    "sections": [
                        {
                            "index": index,
                            "heading": section.heading,
                            "heading_level": section.heading_level,
                            "start_offset": section.start_offset,
                            "end_offset": section.end_offset,
                            "chars": len(section.text),
                            "preview": " ".join(section.text.split())[:160],
                        }
                        for index, section in enumerate(sections)
                    ],
                }
            )
        documents.append(
            {
                "cache_file": str(path),
                "source": source,
                "document_chars": len(text),
                "strategies": strategies,
            }
        )
    return {"documents": documents}


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Markdown heading-based parent section depths."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory containing parsed cache JSON files.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Maximum Markdown heading depths used as split boundaries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/markdown_parent_sections.json"),
        help="JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = sorted(args.parsed_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No parsed JSON files found in {args.parsed_dir}")
    report = analyze_files(paths, args.levels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for document in report["documents"]:
        print(f"\n{document['source']} ({document['document_chars']} chars)")
        print(
            "level  count    min    p50    p95     max    mean  "
            "<100  <300  tables split"
        )
        for strategy in document["strategies"]:
            stats = strategy["stats"]
            print(
                f"H1-H{stats['boundary_max_level']:<2}"
                f"{stats['section_count']:>7}"
                f"{stats['char_min']:>7}"
                f"{stats['char_median']:>7}"
                f"{stats['char_p95']:>7}"
                f"{stats['char_max']:>8}"
                f"{stats['char_mean']:>8.1f}"
                f"{stats['sections_under_100_chars']:>6}"
                f"{stats['sections_under_300_chars']:>6}"
                f"{stats['table_count']:>8}"
                f"{stats['split_table_count']:>6}"
            )
    print(f"\nDetailed report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
