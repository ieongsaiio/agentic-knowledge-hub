#!/usr/bin/env python
"""Validate PaddleOCR parsing, caching, and chunk page provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.core.settings import load_settings, resolve_path
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.parsed_document_cache import ParsedDocumentCache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,%()$-]*", text.lower()))


def _pdf_page_texts(path: Path) -> list[str]:
    try:
        import fitz
    except ImportError:
        return []
    with fitz.open(path) as pdf:
        return [page.get_text("text") for page in pdf]


def _expected_pages(page_spans: list[dict[str, int]], start: int, end: int) -> list[int]:
    pages: set[int] = set()
    for span in page_spans:
        if start >= span["end_offset"] or end <= span["start_offset"]:
            continue
        page = span.get("page")
        if isinstance(page, int):
            pages.add(page)
            continue
        page_start = span.get("page_start")
        page_end = span.get("page_end")
        if isinstance(page_start, int) and isinstance(page_end, int):
            pages.update(range(page_start, page_end + 1))
    return sorted(pages)


def _page_comparisons(document: Any, pdf_texts: list[str]) -> list[dict[str, Any]]:
    comparisons = []
    spans = document.metadata.get("page_spans", [])
    ranges_by_page: dict[int, set[tuple[int, int]]] = {}
    for span in spans:
        span_range = (span["start_offset"], span["end_offset"])
        page = span.get("page")
        if isinstance(page, int):
            ranges_by_page.setdefault(page, set()).add(span_range)
            continue
        page_start = span.get("page_start")
        page_end = span.get("page_end")
        if isinstance(page_start, int) and isinstance(page_end, int):
            for source_page in range(page_start, page_end + 1):
                ranges_by_page.setdefault(source_page, set()).add(span_range)

    for page in sorted(ranges_by_page):
        markdown = "\n".join(
            document.text[start:end]
            for start, end in sorted(ranges_by_page[page])
        )
        pdf_text = pdf_texts[page - 1] if page <= len(pdf_texts) else ""
        pdf_tokens = _tokens(pdf_text)
        markdown_tokens = _tokens(markdown)
        overlap = len(pdf_tokens & markdown_tokens)
        comparisons.append(
            {
                "page": page,
                "pdf_text_chars": len(pdf_text),
                "markdown_chars": len(markdown),
                "pdf_token_coverage": (overlap / len(pdf_tokens) if pdf_tokens else None),
                "token_jaccard": (
                    overlap / len(pdf_tokens | markdown_tokens)
                    if pdf_tokens or markdown_tokens
                    else None
                ),
                "pdf_excerpt": pdf_text[:800],
                "markdown_excerpt": markdown[:800],
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="PDF file to validate")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--force", action="store_true", help="Ignore parsed cache")
    parser.add_argument("--output", help="Optional validation report JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    settings = load_settings(args.config)
    if settings.ingestion is None:
        raise RuntimeError("ingestion settings are required")
    paddle_loader = replace(settings.ingestion.loader, provider="paddle")
    settings = replace(
        settings,
        ingestion=replace(settings.ingestion, loader=paddle_loader),
    )

    loader = LoaderFactory.create(
        settings,
        image_storage_dir=resolve_path("data/images/paddle_validation"),
    )
    cache = ParsedDocumentCache(
        cache_dir=settings.ingestion.loader.parsed_dir,
        loader_config=loader.cache_config(),
    )
    file_hash = _sha256(pdf_path)

    started = time.perf_counter()
    document = None if args.force else cache.get(file_hash, source_path=pdf_path)
    cache_hit = document is not None
    if document is None:
        document = loader.load(pdf_path)
        cache_path = cache.put(file_hash, document)
    else:
        cache_path = cache.cache_path(file_hash)
    parse_seconds = time.perf_counter() - started

    chunk_started = time.perf_counter()
    chunks = DocumentChunker(settings).split_document(document)
    chunk_seconds = time.perf_counter() - chunk_started
    page_spans = document.metadata.get("page_spans", [])
    errors: list[dict[str, Any]] = []
    for chunk in chunks:
        start = chunk.start_offset
        end = chunk.end_offset
        if start is None or end is None or document.text[start:end] != chunk.text:
            errors.append({"chunk_id": chunk.id, "error": "text_offset_mismatch"})
            continue
        pages = _expected_pages(page_spans, start, end)
        if pages and (
            chunk.metadata.get("page_start") != min(pages)
            or chunk.metadata.get("page_end") != max(pages)
        ):
            errors.append(
                {
                    "chunk_id": chunk.id,
                    "error": "page_range_mismatch",
                    "expected": [min(pages), max(pages)],
                    "actual": [
                        chunk.metadata.get("page_start"),
                        chunk.metadata.get("page_end"),
                    ],
                }
            )

    artifact = document.metadata.get("parsed_artifact", {})
    report = {
        "pdf": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "file_hash": file_hash,
        "cache_hit": cache_hit,
        "cache_path": str(cache_path),
        "parse_or_cache_seconds": parse_seconds,
        "paddle_inference_seconds": artifact.get("elapsed_seconds"),
        "chunk_seconds": chunk_seconds,
        "page_count": document.metadata.get("page_count"),
        "page_span_count": len(page_spans),
        "markdown_chars": len(document.text),
        "chunk_count": len(chunks),
        "metadata_validation": {
            "valid": not errors,
            "error_count": len(errors),
            "errors": errors[:20],
        },
        "pages": _page_comparisons(document, _pdf_page_texts(pdf_path)),
    }

    output = (
        Path(args.output)
        if args.output
        else resolve_path(f"data/parsed/{pdf_path.stem}_validation.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Validation report: {output}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
