#!/usr/bin/env python
"""Repeat MinerU API parsing for the three known problem table pages."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import fitz

from src.core.settings import load_settings
from src.libs.loader.mineru_pdf_loader import MineruPdfLoader

TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
CASES = (
    {
        "case_id": "financebench_id_00799",
        "pdf": "AMCOR_2023_10K.pdf",
        "page": 52,
        "required_rows": ("Total current assets", "Total current liabilities"),
        "required_headers": ("2023", "2022"),
        "required_context": ("in millions",),
    },
    {
        "case_id": "financebench_id_04660",
        "pdf": "BLOCK_2016_10K.pdf",
        "page": 68,
        "required_rows": ("Total current assets", "Total current liabilities"),
        "required_headers": ("2016", "2015"),
        "required_context": ("in thousands",),
    },
    {
        "case_id": "financebench_id_04171",
        "pdf": "MGMRESORTS_2018_10K.pdf",
        "page": 57,
        "required_rows": ("Accounts payable",),
        "required_headers": ("2018", "2017"),
        "required_context": ("in thousands",),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--pdf-dir", default="data/benchmarks/financebench/pdfs")
    parser.add_argument(
        "--output-dir",
        default="data/analysis/mineru_problem_table_repeat",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=3)
    return parser.parse_args()


def _normal(text: str) -> str:
    return " ".join(text.casefold().replace("&amp;", "&").split())


def _extract_page(source: Path, page_number: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(source) as pdf:
        if page_number < 1 or page_number > pdf.page_count:
            raise ValueError(f"Invalid page {page_number} for {source.name}")
        output = fitz.open()
        try:
            output.insert_pdf(pdf, from_page=page_number - 1, to_page=page_number - 1)
            output.save(destination, garbage=4, deflate=True)
        finally:
            output.close()


def _analyze(case: dict[str, Any], markdown: str) -> dict[str, Any]:
    tables = TABLE_RE.findall(markdown)
    normalized_tables = [_normal(table) for table in tables]
    required_rows = [_normal(value) for value in case["required_rows"]]
    required_headers = [_normal(value) for value in case["required_headers"]]
    required_context = [_normal(value) for value in case["required_context"]]
    scores = [sum(row in table for row in required_rows) for table in normalized_tables]
    best_index = max(range(len(tables)), key=scores.__getitem__) if tables else None
    best = normalized_tables[best_index] if best_index is not None else ""
    page = _normal(markdown)
    row_locations = {
        row: [index for index, table in enumerate(normalized_tables) if row in table]
        for row in required_rows
    }
    return {
        "table_count": len(tables),
        "selected_table_index": best_index,
        "required_rows_in_one_table": all(row in best for row in required_rows),
        "required_headers_in_selected_table": all(
            header in best for header in required_headers
        ),
        "required_context_on_page": all(value in page for value in required_context),
        "required_context_in_selected_table": all(
            value in best for value in required_context
        ),
        "row_table_indices": row_locations,
    }


def _loader(settings_path: str) -> MineruPdfLoader:
    settings = load_settings(settings_path)
    mineru_config = dict(settings.ingestion.loader.mineru)
    return MineruPdfLoader(mineru_config=mineru_config, extract_images=False)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds < 1 or args.concurrency < 1:
        raise ValueError("--rounds and --concurrency must be positive")
    pdf_dir = Path(args.pdf_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, Path] = {}
    for raw_case in CASES:
        case = dict(raw_case)
        source = pdf_dir / case["pdf"]
        if not source.is_file():
            raise FileNotFoundError(source)
        stem = f"{case['case_id']}_{source.stem}_p{case['page']}"
        slice_path = slices_dir / f"{stem}.pdf"
        _extract_page(source, case["page"], slice_path)
        prepared[case["case_id"]] = slice_path

    semaphore = asyncio.Semaphore(args.concurrency)

    async def process(round_number: int, case: dict[str, Any]) -> dict[str, Any]:
        source = pdf_dir / case["pdf"]
        slice_path = prepared[case["case_id"]]
        stem = f"{case['case_id']}_{source.stem}_p{case['page']}"
        round_dir = output_dir / f"round_{round_number}"
        markdown_dir = round_dir / "markdown"
        artifacts_dir = round_dir / "artifacts"
        tables_dir = round_dir / "tables"
        for directory in (markdown_dir, artifacts_dir, tables_dir):
            directory.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        async with semaphore:
            document = await _loader(args.settings).aload(slice_path)
        elapsed = time.perf_counter() - started
        markdown_path = markdown_dir / f"{stem}.md"
        markdown_path.write_text(document.text, encoding="utf-8")
        artifact = document.metadata["parsed_source_artifact"]
        artifact_path = artifacts_dir / f"{stem}.json"
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tables = TABLE_RE.findall(document.text)
        table_paths: list[str] = []
        for index, table in enumerate(tables, 1):
            table_path = tables_dir / f"{stem}_table_{index}.md"
            table_path.write_text(table + "\n", encoding="utf-8")
            table_paths.append(str(table_path))
        return {
            "round": round_number,
            "case_id": case["case_id"],
            "source_pdf": str(source),
            "physical_page": case["page"],
            "slice_pdf": str(slice_path),
            "elapsed_seconds": round(elapsed, 3),
            "markdown_characters": len(document.text),
            "markdown_path": str(markdown_path),
            "artifact_path": str(artifact_path),
            "table_paths": table_paths,
            "batch_id": artifact.get("batch_id"),
            "checks": _analyze(case, document.text),
        }

    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            process(round_number, dict(case))
            for round_number in range(1, args.rounds + 1)
            for case in CASES
        )
    )
    report = {
        "backend": "mineru_api",
        "model_version": "vlm",
        "rounds": args.rounds,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    report = asyncio.run(run(parse_args()))
    safe_report = json.loads(json.dumps(report))
    for result in safe_report["results"]:
        result.pop("batch_id", None)
    print(json.dumps(safe_report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
