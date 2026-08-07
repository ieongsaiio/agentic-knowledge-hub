#!/usr/bin/env python
"""Split benchmark parsed documents and report chunk/unit statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.settings import load_settings
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.parsed_document_cache import ParsedDocumentCache
from src.libs.splitter.structured_markdown_splitter import (
    strip_table_child_section_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="data/analysis/evidence_judgements_30.jsonl",
        help="JSONL containing benchmark cases and their evidences.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/benchmarks/financebench/pdfs",
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--table-child-max-tokens",
        type=int,
        help="Override the configured table-child token budget for this report.",
    )
    parser.add_argument(
        "--output",
        default="data/analysis/financebench_30_chunking_statistics.md",
    )
    return parser.parse_args()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Case at line {line_number} must be an object")
            cases.append(value)
    return cases


def _case_documents(cases: Iterable[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for case in cases:
        for evidence in case.get("evidences", []):
            if isinstance(evidence, dict) and evidence.get("document_name"):
                names.add(str(evidence["document_name"]))
    return sorted(names)


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _stats(values: list[int]) -> str:
    if not values:
        return "count=0"
    return (
        f"count={len(values)}, min={min(values)}, "
        f"avg={statistics.mean(values):.1f}, "
        f"median={statistics.median(values):.1f}, "
        f"p90={_percentile(values, 0.90)}, "
        f"p95={_percentile(values, 0.95)}, max={max(values)}"
    )


def _unit_types(metadata: dict[str, Any]) -> list[str]:
    value = metadata.get("unit_types", [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _section_count(document: Any) -> int:
    root = document.metadata.get("section_tree", {}).get("root")
    if not isinstance(root, dict):
        return 0
    count = 0
    stack = [root]
    while stack:
        section = stack.pop()
        count += 1
        stack.extend(child for child in section.get("subsections", []) if isinstance(child, dict))
    return count


def _composition(types: list[str]) -> str:
    return " + ".join(types) if types else "unknown"


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    if settings.ingestion is None:
        raise RuntimeError("ingestion settings are required")
    if args.table_child_max_tokens is not None:
        if args.table_child_max_tokens < 1:
            raise ValueError("--table-child-max-tokens must be greater than zero")
        settings.ingestion.structured_chunking["table_child_chunking"][
            "max_tokens"
        ] = args.table_child_max_tokens

    cases_path = Path(args.cases).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    cases = _read_cases(cases_path)
    document_names = _case_documents(cases)
    pdfs = {path.stem.lower(): path for path in pdf_dir.glob("*.pdf")}

    loader = LoaderFactory.create(settings)
    cache = ParsedDocumentCache(
        cache_dir=settings.ingestion.loader.parsed_dir,
        loader_config=loader.cache_config(),
    )
    chunker = DocumentChunker(settings)
    token_length = getattr(chunker._splitter, "_length", len)

    total_started = time.perf_counter()
    document_rows: list[dict[str, Any]] = []
    all_token_lengths: list[int] = []
    all_dense_token_lengths: list[int] = []
    all_char_lengths: list[int] = []
    type_occurrences: Counter[str] = Counter()
    type_chunk_counts: Counter[str] = Counter()
    pure_type_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    type_chunk_tokens: dict[str, list[int]] = defaultdict(list)
    pure_type_tokens: dict[str, list[int]] = defaultdict(list)
    oversized_by_type: Counter[str] = Counter()
    table_previous_context: list[int] = []
    table_next_context: list[int] = []
    table_child_tokens: list[int] = []
    table_child_dense_tokens: list[int] = []
    table_children_per_parent: list[int] = []
    table_parent_count = 0
    table_child_count = 0
    table_child_oversized = 0
    table_child_budget_violations = 0
    table_child_storage_ids: set[str] = set()
    errors: list[str] = []
    missing: list[str] = []

    for document_name in document_names:
        pdf_path = pdfs.get(document_name.lower())
        if pdf_path is None:
            missing.append(f"{document_name}: PDF not found")
            continue

        document = cache.get(_file_hash(pdf_path), source_path=pdf_path)
        if document is None:
            missing.append(f"{document_name}: matching parsed cache not found")
            continue

        started = time.perf_counter()
        chunks = chunker.split_document(document)
        elapsed = time.perf_counter() - started
        doc_tokens: list[int] = []
        doc_types: Counter[str] = Counter()
        doc_compositions: Counter[str] = Counter()
        doc_oversized = 0
        document_table_children: dict[str, list[Any]] = defaultdict(list)

        for chunk_index, chunk in enumerate(chunks):
            types = _unit_types(chunk.metadata)
            composition = _composition(types)
            raw_tokens = token_length(chunk.text)
            dense_tokens = token_length(chunk.dense_index_text or chunk.text)
            doc_tokens.append(raw_tokens)
            all_token_lengths.append(raw_tokens)
            all_dense_token_lengths.append(dense_tokens)
            all_char_lengths.append(len(chunk.text))
            type_occurrences.update(types)
            doc_types.update(types)
            composition_counts[composition] += 1
            doc_compositions[composition] += 1

            unique_types = list(dict.fromkeys(types))
            for unit_type in unique_types:
                type_chunk_counts[unit_type] += 1
                type_chunk_tokens[unit_type].append(raw_tokens)
            if len(unique_types) == 1:
                pure_type_counts[unique_types[0]] += 1
                pure_type_tokens[unique_types[0]].append(raw_tokens)

            if raw_tokens > settings.ingestion.chunk_size:
                doc_oversized += 1
                oversized_by_type[composition] += 1

            start = chunk.start_offset
            end = chunk.end_offset
            is_table_child = chunk.metadata.get("chunk_role") == "table_child"
            if is_table_child:
                table_child_count += 1
                parent_id = str(chunk.metadata.get("parent_chunk_id", ""))
                document_table_children[parent_id].append(chunk)
                child_tokens = int(chunk.metadata.get("table_child_token_count", dense_tokens))
                table_child_tokens.append(child_tokens)
                table_child_dense_tokens.append(dense_tokens)
                child_limit = int(
                    settings.ingestion.structured_chunking["table_child_chunking"]["max_tokens"]
                )
                if child_tokens > child_limit:
                    table_child_oversized += 1
                    source_rows = set(chunk.metadata.get("source_row_indices", []))
                    overlap_rows = set(chunk.metadata.get("overlap_row_indices", []))
                    new_rows = source_rows - overlap_rows
                    if len(new_rows) > 1:
                        table_child_budget_violations += 1
                        errors.append(
                            f"{document_name} chunk {chunk_index}: "
                            "multiple new rows exceed the table-child token budget"
                        )

                parent_start = chunk.metadata.get("parent_start_offset")
                parent_end = chunk.metadata.get("parent_end_offset")
                table_start = chunk.metadata.get("table_start_offset")
                table_end = chunk.metadata.get("table_end_offset")
                valid_offsets = (
                    isinstance(parent_start, int)
                    and isinstance(parent_end, int)
                    and isinstance(table_start, int)
                    and isinstance(table_end, int)
                    and 0 <= parent_start <= table_start < table_end <= parent_end
                    and parent_end <= len(document.text)
                )
                if not valid_offsets:
                    errors.append(
                        f"{document_name} chunk {chunk_index}: invalid parent/table offsets"
                    )
                else:
                    source_table = document.text[table_start:table_end].casefold()
                    if "<table" not in source_table or "</table>" not in source_table:
                        errors.append(
                            f"{document_name} chunk {chunk_index}: "
                            "parent offsets do not reconstruct a complete table"
                        )
                if start != parent_start or end != parent_end:
                    errors.append(
                        f"{document_name} chunk {chunk_index}: "
                        "child offsets differ from parent offsets"
                    )
                if chunk.metadata.get("source_exact") is not False:
                    errors.append(
                        f"{document_name} chunk {chunk_index}: "
                        "generated child is not marked source_exact=false"
                    )
                if not parent_id:
                    errors.append(f"{document_name} chunk {chunk_index}: missing parent_chunk_id")
                storage_id = str(chunk.metadata.get("storage_id", ""))
                if not storage_id or storage_id in table_child_storage_ids:
                    errors.append(
                        f"{document_name} chunk {chunk_index}: missing/duplicate storage_id"
                    )
                table_child_storage_ids.add(storage_id)
                expected_dense_text = strip_table_child_section_path(chunk.text)
                if expected_dense_text != (chunk.dense_index_text or ""):
                    errors.append(
                        f"{document_name} chunk {chunk_index}: "
                        "dense child content does not remove only the Section path"
                    )
                if chunk.text != (chunk.sparse_index_text or ""):
                    errors.append(
                        f"{document_name} chunk {chunk_index}: "
                        "sparse child content differs from stored content"
                    )
            elif (
                not isinstance(start, int)
                or not isinstance(end, int)
                or document.text[start:end] != chunk.text
            ):
                errors.append(f"{document_name} chunk {chunk_index}: source offset mismatch")

            if "table" in unique_types:
                lowered = chunk.text.lower()
                if "<table" in lowered and "</table>" not in lowered:
                    errors.append(f"{document_name} chunk {chunk_index}: incomplete HTML table")
                previous = str(chunk.metadata.get("previous_context", ""))
                following = str(chunk.metadata.get("next_context", ""))
                if previous:
                    table_previous_context.append(token_length(previous))
                if following:
                    table_next_context.append(token_length(following))

        for parent_id, children in document_table_children.items():
            table_parent_count += 1
            table_children_per_parent.append(len(children))
            ordered = sorted(
                children,
                key=lambda item: int(item.metadata["table_child_index"]),
            )
            child_indices = [int(child.metadata["table_child_index"]) for child in ordered]
            declared_counts = {int(child.metadata["table_child_count"]) for child in ordered}
            if child_indices != list(range(len(ordered))) or declared_counts != {len(ordered)}:
                errors.append(
                    f"{document_name} parent {parent_id}: child indices/count are inconsistent"
                )

            parent_row_counts = {int(child.metadata["parent_table_row_count"]) for child in ordered}
            covered_rows: set[int] = set()
            for child in ordered:
                covered_rows.update(child.metadata.get("source_row_indices", []))
                covered_rows.update(child.metadata.get("repeated_prefix_row_indices", []))
            expected_rows = set(range(next(iter(parent_row_counts), 0)))
            if len(parent_row_counts) != 1 or covered_rows != expected_rows:
                errors.append(f"{document_name} parent {parent_id}: incomplete table-row coverage")

        document_rows.append(
            {
                "document": document_name,
                "pages": document.metadata.get("page_count", "n/a"),
                "sections": _section_count(document),
                "markdown_chars": len(document.text),
                "chunks": len(chunks),
                "tokens": doc_tokens,
                "types": doc_types,
                "compositions": doc_compositions,
                "oversized": doc_oversized,
                "seconds": elapsed,
            }
        )

    total_seconds = time.perf_counter() - total_started
    known_types = sorted(set(type_occurrences) | {"text", "table", "list", "code", "blockquote"})
    total_pages = sum(int(row["pages"]) for row in document_rows if isinstance(row["pages"], int))
    table_context_limit = settings.ingestion.structured_chunking.get(
        "table_context_tokens",
        80,
    )
    table_child_config = settings.ingestion.structured_chunking.get(
        "table_child_chunking",
        {},
    )
    table_child_limit = int(table_child_config.get("max_tokens", 768))
    oversized_contexts = sum(
        value > table_context_limit for value in table_previous_context + table_next_context
    )

    lines = [
        "# FinanceBench 30 Pure Chunking Statistics",
        "",
        "## Scope",
        "",
        f"- Benchmark cases: `{len(cases)}`",
        f"- Unique referenced PDFs: `{len(document_names)}`",
        f"- Successfully loaded parsed documents: `{len(document_rows)}`",
        f"- Missing documents/cache entries: `{len(missing)}`",
        f"- Total PDF pages: `{total_pages}`",
        f"- Splitter: `{settings.ingestion.splitter}`",
        f"- Length unit: `{settings.ingestion.length_unit}`",
        f"- Chunk target: `{settings.ingestion.chunk_size}` tokens",
        f"- Chunk overlap: `{settings.ingestion.chunk_overlap}` tokens",
        f"- Table context limit: `{table_context_limit}` tokens",
        f"- Table-child chunking enabled: `{bool(table_child_config.get('enabled'))}`",
        f"- Table-child token limit: `{table_child_limit}` tokens",
        "- Embedding/API calls: `none`",
        f"- Total chunking time: `{total_seconds:.3f} s`",
        "",
        "## Overall Chunk Statistics",
        "",
        f"- Retrieval chunks: `{len(all_token_lengths)}`",
        f"- Raw token distribution: `{_stats(all_token_lengths)}`",
        f"- Dense-input token distribution: `{_stats(all_dense_token_lengths)}`",
        f"- Raw character distribution: `{_stats(all_char_lengths)}`",
        f"- Above-target protected chunks: `{sum(oversized_by_type.values())}`",
        f"- Source/table integrity errors: `{len(errors)}`",
        "",
        "## Table Child Audit",
        "",
        f"- Parent tables represented by children: `{table_parent_count}`",
        f"- Retrieval table children: `{table_child_count}`",
        f"- Children per parent: `{_stats(table_children_per_parent)}`",
        f"- Stored/Sparse child token distribution: `{_stats(table_child_tokens)}`",
        f"- Dense child token distribution (Section removed): "
        f"`{_stats(table_child_dense_tokens)}`",
        f"- Children above `{table_child_limit}` tokens: `{table_child_oversized}`",
        "- Above-limit children with more than one newly packed row: "
        f"`{table_child_budget_violations}`",
        f"- Unique stable storage IDs: `{len(table_child_storage_ids)}`",
        "- Text contract: `sparse_index_text == Chunk.text`; Dense removes only "
        "the leading generated Section path.",
        "- Full parent recovery: `parsed cache + parent/table character offsets`",
        "",
        "## Unit Statistics",
        "",
        "`Occurrences` counts unit segments before/while packing. "
        "`Containing chunks` counts retrieval chunks containing that type. "
        "Mixed chunks therefore appear in more than one type row.",
        "",
        "| Unit type | Occurrences | Containing chunks | Pure chunks | "
        "Containing-chunk token distribution | Pure-chunk token distribution |",
        "|---|---:|---:|---:|---|---|",
    ]
    for unit_type in known_types:
        lines.append(
            f"| {unit_type} | {type_occurrences[unit_type]} | "
            f"{type_chunk_counts[unit_type]} | {pure_type_counts[unit_type]} | "
            f"{_stats(type_chunk_tokens[unit_type])} | "
            f"{_stats(pure_type_tokens[unit_type])} |"
        )

    lines.extend(
        [
            "",
            "## Chunk Compositions",
            "",
            f"| Composition | Chunks | Above {settings.ingestion.chunk_size} tokens |",
            "|---|---:|---:|",
        ]
    )
    for composition, count in composition_counts.most_common():
        lines.append(f"| {composition} | {count} | {oversized_by_type[composition]} |")

    lines.extend(
        [
            "",
            "## Table Context",
            "",
            f"- Previous-context distribution: `{_stats(table_previous_context)}`",
            f"- Next-context distribution: `{_stats(table_next_context)}`",
            f"- Contexts above configured limit: `{oversized_contexts}`",
            "",
            "## Per-document Statistics",
            "",
            "| Document | Pages | Sections | Markdown chars | Chunks | Text | "
            "Table | List | Code | Blockquote | Avg tokens | P95 tokens | "
            "Oversized | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in document_rows:
        token_values = row["tokens"]
        lines.append(
            f"| {row['document']} | {row['pages']} | {row['sections']} | "
            f"{row['markdown_chars']} | {row['chunks']} | "
            f"{row['types']['text']} | {row['types']['table']} | "
            f"{row['types']['list']} | {row['types']['code']} | "
            f"{row['types']['blockquote']} | "
            f"{statistics.mean(token_values):.1f} | "
            f"{_percentile(token_values, 0.95)} | {row['oversized']} | "
            f"{row['seconds']:.3f} |"
        )

    if missing:
        lines.extend(["", "## Missing Inputs", "", *[f"- {item}" for item in missing]])
    if errors:
        lines.extend(["", "## Integrity Errors", "", *[f"- {item}" for item in errors]])

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={output}")
    print(f"cases={len(cases)} documents={len(document_rows)}")
    print(f"chunks={len(all_token_lengths)} errors={len(errors)}")
    print(f"seconds={total_seconds:.3f}")
    return 0 if not missing and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
