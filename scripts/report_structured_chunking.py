#!/usr/bin/env python
"""Generate a readable per-chunk report from a parsed PDF cache."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.core.settings import load_settings
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.parsed_document_cache import ParsedDocumentCache


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_sections(node: dict[str, Any]) -> list[dict[str, Any]]:
    sections = [node]
    for child in node.get("subsections", []):
        sections.extend(_walk_sections(child))
    return sections


def _fenced(label: str, content: str) -> str:
    fence = "````"
    while fence in content:
        fence += "`"
    return f"{fence}{label}\n{content}\n{fence}"


def _stats(values: list[int]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return (
        f"min={ordered[0]}, avg={statistics.mean(values):.1f}, "
        f"median={statistics.median(values):.1f}, "
        f"p95={ordered[p95_index]}, max={ordered[-1]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).resolve()
    settings = load_settings(args.config)
    if settings.ingestion is None:
        raise RuntimeError("ingestion settings are required")

    loader = LoaderFactory.create(settings)
    cache = ParsedDocumentCache(
        cache_dir=settings.ingestion.loader.parsed_dir,
        loader_config=loader.cache_config(),
    )
    document = cache.get(_file_hash(pdf_path), source_path=pdf_path)
    if document is None:
        raise RuntimeError(
            "No matching parsed cache. Parse the PDF before generating this report."
        )

    started = time.perf_counter()
    chunker = DocumentChunker(settings)
    chunks = chunker.split_document(document)
    chunk_seconds = time.perf_counter() - started
    tree = document.metadata["section_tree"]
    sections = _walk_sections(tree["root"])
    page_spans = document.metadata.get("page_spans", [])

    errors: list[str] = []
    for index, chunk in enumerate(chunks):
        start = chunk.start_offset
        end = chunk.end_offset
        if start is None or end is None or document.text[start:end] != chunk.text:
            errors.append(f"Chunk {index}: source offset mismatch")
        expected_pages = [
            span["page"]
            for span in page_spans
            if isinstance(span, dict)
            and isinstance(span.get("page"), int)
            and start is not None
            and end is not None
            and start < span.get("end_offset", -1)
            and end > span.get("start_offset", -1)
        ]
        if expected_pages and (
            chunk.metadata.get("page_start") != min(expected_pages)
            or chunk.metadata.get("page_end") != max(expected_pages)
        ):
            errors.append(f"Chunk {index}: page range mismatch")
        if "table" in chunk.metadata.get("unit_types", []):
            raw = chunk.text.lower()
            if "<table" in raw and "</table>" not in raw:
                errors.append(f"Chunk {index}: incomplete HTML table")

    type_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    for chunk in chunks:
        unit_types = chunk.metadata.get("unit_types", [])
        type_counts.update(unit_types)
        composition_counts[" + ".join(unit_types)] += 1

    char_lengths = [len(chunk.text) for chunk in chunks]
    dense_lengths = [len(chunk.dense_index_text or chunk.text) for chunk in chunks]
    length_function = getattr(chunker._splitter, "_length", len)
    raw_token_lengths = [length_function(chunk.text) for chunk in chunks]
    dense_token_lengths = [
        length_function(chunk.dense_index_text or chunk.text)
        for chunk in chunks
    ]
    representation_counts = Counter(
        chunk.metadata.get("embedding_source_type", "raw_text")
        for chunk in chunks
    )
    oversized_counts = Counter(
        " + ".join(chunk.metadata.get("unit_types", []))
        for chunk, token_length in zip(chunks, raw_token_lengths)
        if token_length > settings.ingestion.chunk_size
    )
    artifact = document.metadata.get("parsed_artifact", {})
    elapsed = artifact.get("elapsed_seconds")
    loader_config = loader.cache_config()

    lines = [
        f"# {pdf_path.stem} Structured Chunking Report",
        "",
        "## 1. Input And Configuration",
        "",
        f"- PDF: `{pdf_path}`",
        f"- PDF bytes: `{pdf_path.stat().st_size}`",
        f"- Loader: `{loader_config.get('loader')}`",
        f"- Paddle backend: `{loader_config.get('backend')}`",
        f"- Pages: `{document.metadata.get('page_count')}`",
        f"- Parsed Markdown characters: `{len(document.text)}`",
        f"- Paddle inference: `{elapsed:.3f} s`" if isinstance(elapsed, (int, float)) else "- Paddle inference: `n/a`",
        f"- Splitter: `{settings.ingestion.splitter}`",
        f"- Length unit: `{settings.ingestion.length_unit}`",
        f"- Chunk target: `{settings.ingestion.chunk_size}`",
        f"- Chunk overlap: `{settings.ingestion.chunk_overlap}`",
        f"- Chunking time (cache hit): `{chunk_seconds:.3f} s`",
        "",
        "## 2. Result Summary",
        "",
        f"- Section nodes including root: `{len(sections)}`",
        f"- Retrieval chunks: `{len(chunks)}`",
        f"- Raw character distribution: `{_stats(char_lengths)}`",
        f"- Dense-index character distribution: `{_stats(dense_lengths)}`",
        f"- Raw token distribution: `{_stats(raw_token_lengths)}`",
        f"- Dense-index token distribution: `{_stats(dense_token_lengths)}`",
        f"- Unit occurrences: `{dict(type_counts)}`",
        f"- Chunk compositions: `{dict(composition_counts)}`",
        f"- Dense representation types: `{dict(representation_counts)}`",
        f"- Above-target protected chunks: `{dict(oversized_counts)}`",
        "",
        "## 3. Integrity Checks",
        "",
        f"- Exact `document.text[start_offset:end_offset] == chunk.text`: `{'PASS' if not any('offset' in error for error in errors) else 'FAIL'}`",
        f"- Page range metadata: `{'PASS' if not any('page' in error for error in errors) else 'FAIL'}`",
        f"- Complete HTML table boundaries: `{'PASS' if not any('table' in error for error in errors) else 'FAIL'}`",
        f"- Total validation errors: `{len(errors)}`",
    ]
    if errors:
        lines.extend(["", *[f"- {error}" for error in errors]])

    lines.extend(["", "## 4. Section Tree", ""])
    for section in sections:
        path = " > ".join(section.get("path") or []) or "(document preamble)"
        lines.append(
            f"- `{section['id']}` depth={section['heading_depth']} "
            f"direct_range={section['content_start_offset']}:{section['content_end_offset']} "
            f"path=`{path}`"
        )

    lines.extend(["", "## 5. Chunk Details", ""])
    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata
        path = " > ".join(metadata.get("header_path") or []) or "(document preamble)"
        lines.extend(
            [
                f"### Chunk {index:03d}",
                "",
                f"- ID: `{chunk.id}`",
                f"- Section path: `{path}`",
                f"- Unit types: `{metadata.get('unit_types')}`",
                f"- Page: `{metadata.get('page_start')}` to `{metadata.get('page_end')}`",
                f"- Source offsets: `{chunk.start_offset}:{chunk.end_offset}`",
                f"- Raw characters: `{len(chunk.text)}`",
                f"- Dense representation: `{metadata.get('embedding_source_type')}`",
            ]
        )
        if metadata.get("table_title"):
            lines.append(f"- Table title: `{metadata['table_title']}`")
        if metadata.get("vision_footnotes"):
            lines.append(f"- Table footnotes: `{metadata['vision_footnotes']}`")
        lines.extend(
            [
                "",
                "**Original content stored and returned**",
                "",
                _fenced("markdown", chunk.text),
                "",
                "**Dense embedding input**",
                "",
                _fenced("text", chunk.dense_index_text or chunk.text),
                "",
                "**Sparse/BM25 input**",
                "",
                _fenced("markdown", chunk.sparse_index_text or chunk.text),
                "",
            ]
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"report={output}\nchunks={len(chunks)}\n"
        f"sections={len(sections)}\nerrors={len(errors)}\n"
        f"chunk_seconds={chunk_seconds:.3f}"
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
