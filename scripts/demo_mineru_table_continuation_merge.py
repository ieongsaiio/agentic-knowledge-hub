"""Demonstrate conservative table continuation merging on saved MinerU artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.libs.loader.mineru_artifact_normalizer import MinerUArtifactNormalizer
from src.libs.loader.table_continuation_merger import TableContinuationMerger

DEFAULT_INPUT = Path(
    "data/analysis/mineru_problem_table_repeat/round_1/artifacts"
)
DEFAULT_OUTPUT = Path("data/analysis/mineru_table_continuation_demo")


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def run(input_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = MinerUArtifactNormalizer()
    merger = TableContinuationMerger()
    report: list[str] = [
        "# MinerU Split Table Continuation Demo",
        "",
        "This demo uses saved API artifacts only; it performs no OCR/API calls.",
        "",
    ]

    for artifact_path in sorted(input_dir.glob("*.json")):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        parsed = normalizer.normalize(artifact)
        merged = merger.process(parsed)
        case_dir = output_dir / _safe_name(artifact_path.stem)
        case_dir.mkdir(parents=True, exist_ok=True)

        report.extend([f"## {artifact_path.stem}", ""])
        for source_page, result_page in zip(parsed.pages, merged.pages):
            source_tables = [block for block in source_page.blocks if block.type == "table"]
            result_tables = [block for block in result_page.blocks if block.type == "table"]
            report.append(
                f"- Page {source_page.page_index + 1}: "
                f"{len(source_tables)} source table(s) -> "
                f"{len(result_tables)} result table(s)"
            )
            for index, block in enumerate(source_tables, start=1):
                source_path = case_dir / f"source_table_{index}.html"
                source_path.write_text(block.content, encoding="utf-8")
                report.append(
                    f"- Source {index}: `{source_path.as_posix()}`; "
                    f"caption={block.caption!r}; bbox={block.bbox!r}"
                )
            ordered = sorted(source_page.blocks, key=merger._visual_sort_key)
            for left, right in zip(ordered, ordered[1:]):
                if left.type != "table" or right.type != "table":
                    continue
                assessment = merger.assess(
                    left,
                    right,
                    page_width=source_page.width,
                    page_height=source_page.height,
                )
                report.append(
                    f"- Candidate `{left.block_id}` -> `{right.block_id}`: "
                    f"accepted={assessment.accepted}; "
                    f"width={assessment.left_width}/{assessment.right_width}; "
                    f"horizontal_overlap={assessment.horizontal_overlap_ratio:.3f}; "
                    f"vertical_gap={assessment.vertical_gap_ratio:.3f}; "
                    f"reasons={', '.join(assessment.reasons)}"
                )

            for index, block in enumerate(result_tables, start=1):
                html_path = case_dir / f"merged_table_{index}.html"
                html_path.write_text(block.content, encoding="utf-8")
                audit_path = case_dir / f"merged_table_{index}.json"
                audit_path.write_text(
                    json.dumps(block.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                report.extend(
                    [
                        f"- Result {index}: `{html_path.as_posix()}`",
                        f"  source blocks: `{block.metadata.get('source_block_ids', [block.block_id])}`",
                    ]
                )
        report.append("")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.input_dir, args.output_dir))


if __name__ == "__main__":
    main()
