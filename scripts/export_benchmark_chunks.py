"""Export current offline chunking results for the fixed FinanceBench sample."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.settings import load_settings
from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.loader.canonical_document_assembler import CanonicalDocumentAssembler
from src.libs.loader.markdown_section_tree import build_markdown_section_tree
from src.libs.loader.mineru_artifact_normalizer import MinerUArtifactNormalizer
from src.libs.loader.table_continuation_merger import TableContinuationMerger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "financebench"
    / "data"
    / "financebench_open_source.jsonl"
)
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT
    / "config"
    / "evaluation"
    / "financebench_30_atomic_facts.v1.jsonl"
)
DEFAULT_PARSED = PROJECT_ROOT / "data" / "parsed"
DEFAULT_OUTPUT = PROJECT_ROOT / "tmp" / "financebench_30_chunking_current"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classify_chunk(metadata: dict[str, Any]) -> str:
    """Return a stable human-readable chunk type."""
    role = str(metadata.get("chunk_role") or "")
    if role == "table_group":
        return "table_group"
    if role == "table_summary":
        return "table_summary"
    unit_types = metadata.get("unit_types") or []
    if isinstance(unit_types, str):
        unit_types = [item.strip() for item in unit_types.split(",") if item.strip()]
    normalized = [str(item) for item in unit_types]
    if len(set(normalized)) == 1 and normalized:
        return normalized[0]
    return "mixed:" + "+".join(dict.fromkeys(normalized)) if normalized else "text"


def _safe_reset_output(output_dir: Path) -> None:
    resolved = output_dir.resolve(strict=False)
    temp_root = (PROJECT_ROOT / "tmp").resolve(strict=False)
    if resolved == temp_root or temp_root not in resolved.parents:
        raise ValueError(f"Output directory must be below {temp_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _cache_documents(parsed_dir: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for path in parsed_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = payload.get("document")
        if not isinstance(document, dict):
            continue
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            continue
        source_path = metadata.get("source_path")
        artifact = metadata.get("parsed_source_artifact")
        if isinstance(source_path, str) and isinstance(artifact, dict):
            documents[Path(source_path).stem] = document
    return documents


def _current_document(
    cached: dict[str, Any],
    *,
    ignored_block_types: list[str],
    grouping_config: dict[str, Any],
) -> tuple[Document, dict[str, int]]:
    metadata = cached["metadata"]
    parsed = MinerUArtifactNormalizer().normalize(metadata["parsed_source_artifact"])
    tables_before = sum(
        block.type == "table" for page in parsed.pages for block in page.blocks
    )
    grouped = TableContinuationMerger(**grouping_config).process(parsed)
    tables_after = sum(
        block.type == "table" for page in grouped.pages for block in page.blocks
    )
    multi_unit_groups = sum(
        int(block.metadata.get("unit_count", 1)) > 1
        for page in grouped.pages
        for block in page.blocks
        if block.type == "table"
    )
    document = CanonicalDocumentAssembler(ignored_block_types).assemble(
        grouped,
        source_path=str(metadata["source_path"]),
        doc_id=str(cached["id"]),
        doc_hash=str(metadata.get("doc_hash") or "") or None,
    )
    document.metadata["section_tree"] = build_markdown_section_tree(
        document.text,
        document_id=document.id,
        page_spans=document.metadata.get("page_spans"),
    )
    title = next(
        (line.lstrip("# ").strip() for line in document.text.splitlines() if line.strip()),
        None,
    )
    if title:
        document.metadata["title"] = title
    return document, {
        "pages": int(document.metadata.get("page_count", 0)),
        "tables_before_grouping": tables_before,
        "tables_after_grouping": tables_after,
        "multi_unit_table_groups": multi_unit_groups,
    }


def _chunk_record(chunk: Any, document_name: str) -> dict[str, Any]:
    metadata = dict(chunk.metadata)
    selected_metadata = {
        key: metadata[key]
        for key in (
            "chunk_role",
            "table_group_id",
            "retrieval_group_id",
            "unit_count",
            "units",
            "source_block_ids",
            "header_path",
            "section_id",
            "unit_types",
            "page_start",
            "page_end",
            "page_num",
            "table_title",
            "table_captions",
            "vision_footnotes",
            "previous_context",
            "next_context",
            "source_bbox",
            "parsed_block_id",
            "embedding_source_type",
            "source_exact",
        )
        if key in metadata
    }
    return {
        "document_name": document_name,
        "doc_id": chunk.source_ref,
        "chunk_id": chunk.id,
        "chunk_index": metadata.get("chunk_index"),
        "chunk_type": classify_chunk(metadata),
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "char_count": len(chunk.text),
        "text": chunk.text,
        "dense_index_text": chunk.dense_index_text or chunk.text,
        "sparse_index_text": chunk.sparse_index_text or chunk.text,
        "metadata": selected_metadata,
    }


def export_chunks(
    *,
    settings_path: Path,
    cases_path: Path,
    annotations_path: Path,
    parsed_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    settings = load_settings(settings_path)
    if settings.ingestion.loader.provider != "mineru":
        raise ValueError("This export requires ingestion.loader.provider=mineru")
    summary_config = settings.ingestion.structured_chunking.get("table_summary", {})
    if bool(summary_config.get("enabled", False)):
        raise ValueError("Disable table_summary before running an offline chunk export")

    case_ids = [str(item["case_id"]) for item in _read_jsonl(annotations_path)]
    all_cases = {str(item["financebench_id"]): item for item in _read_jsonl(cases_path)}
    cases = [all_cases[case_id] for case_id in case_ids]
    document_names = sorted(
        {
            str(evidence["doc_name"])
            for case in cases
            for evidence in case.get("evidence", [])
        }
    )
    cached = _cache_documents(parsed_dir)
    missing = sorted(set(document_names) - set(cached))
    if missing:
        raise FileNotFoundError(f"Missing parsed cache for: {', '.join(missing)}")

    _safe_reset_output(output_dir)
    chunks_path = output_dir / "chunks.jsonl"
    documents_path = output_dir / "documents.jsonl"
    index_path = output_dir / "chunk_index.csv"
    grouping = dict(settings.ingestion.loader.mineru.get("table_grouping") or {})
    grouping.pop("enabled", None)
    ignored = list(settings.ingestion.loader.mineru.get("ignored_block_types") or [])
    if settings.vision_llm is None or not settings.vision_llm.enabled:
        ignored.append("image")
    chunker = DocumentChunker(settings)

    type_counts: Counter[str] = Counter()
    document_summaries: list[dict[str, Any]] = []
    total_chars = 0
    total_chunks = 0
    samples: dict[str, dict[str, Any]] = {}
    with chunks_path.open("w", encoding="utf-8", newline="\n") as chunk_file, documents_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as document_file, index_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as index_file:
        index_writer = csv.DictWriter(
            index_file,
            fieldnames=[
                "document_name",
                "chunk_index",
                "chunk_id",
                "chunk_type",
                "page_start",
                "page_end",
                "start_offset",
                "end_offset",
                "char_count",
                "unit_count",
                "header_path",
                "text_preview",
            ],
        )
        index_writer.writeheader()
        for document_name in document_names:
            document_started = time.perf_counter()
            document, parse_stats = _current_document(
                cached[document_name],
                ignored_block_types=ignored,
                grouping_config=grouping,
            )
            chunks = chunker.split_document(document)
            document_types: Counter[str] = Counter()
            for chunk in chunks:
                record = _chunk_record(chunk, document_name)
                chunk_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_metadata = record["metadata"]
                index_writer.writerow(
                    {
                        "document_name": document_name,
                        "chunk_index": record["chunk_index"],
                        "chunk_id": record["chunk_id"],
                        "chunk_type": record["chunk_type"],
                        "page_start": record_metadata.get("page_start", ""),
                        "page_end": record_metadata.get("page_end", ""),
                        "start_offset": record["start_offset"],
                        "end_offset": record["end_offset"],
                        "char_count": record["char_count"],
                        "unit_count": record_metadata.get("unit_count", ""),
                        "header_path": " > ".join(
                            str(item)
                            for item in record_metadata.get("header_path", [])
                        ),
                        "text_preview": " ".join(record["text"].split())[:240],
                    }
                )
                samples.setdefault(record["chunk_type"], record)
                document_types[record["chunk_type"]] += 1
                type_counts[record["chunk_type"]] += 1
                total_chars += record["char_count"]
            summary = {
                "document_name": document_name,
                "source_path": document.metadata["source_path"],
                "doc_id": document.id,
                **parse_stats,
                "markdown_char_count": len(document.text),
                "chunk_count": len(chunks),
                "chunk_type_counts": dict(sorted(document_types.items())),
                "elapsed_seconds": round(time.perf_counter() - document_started, 3),
            }
            document_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
            document_summaries.append(summary)
            total_chunks += len(chunks)

    manifest = {
        "schema_version": 1,
        "purpose": "Offline current chunking export; no OCR/API, LLM, embedding, or storage",
        "question_count": len(cases),
        "document_count": len(document_names),
        "document_names": document_names,
        "chunk_count": total_chunks,
        "chunk_type_counts": dict(sorted(type_counts.items())),
        "average_chunk_chars": round(total_chars / total_chunks, 2) if total_chunks else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "settings": {
            "loader": "mineru_cached_artifact",
            "table_grouping": {"enabled": True, **grouping},
            "splitter": settings.ingestion.splitter,
            "length_unit": settings.ingestion.length_unit,
            "tokenizer_model": settings.ingestion.tokenizer_model,
            "chunk_size": settings.ingestion.chunk_size,
            "chunk_overlap": settings.ingestion.chunk_overlap,
            "table_dense_representation": settings.ingestion.structured_chunking.get(
                "table_dense_representation"
            ),
            "table_context_tokens": settings.ingestion.structured_chunking.get(
                "table_context_tokens"
            ),
            "table_summary_enabled": False,
        },
        "files": {
            "chunks": str(chunks_path),
            "documents": str(documents_path),
            "index": str(index_path),
            "report": str(output_dir / "REPORT.md"),
            "samples": str(output_dir / "SAMPLES.md"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# FinanceBench 30 Current Chunking Export",
        "",
        "This is an offline export from the retained MinerU parsed cache. It does not run OCR/API, table-summary LLM, embeddings, reranking, Chroma, BM25, or evaluation.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(manifest["settings"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Overall Statistics",
        "",
        f"- Questions: **{len(cases)}**",
        f"- Unique PDF documents: **{len(document_names)}**",
        f"- Chunks: **{total_chunks}**",
        f"- Average original characters per Chunk: **{manifest['average_chunk_chars']}**",
        f"- Runtime: **{manifest['elapsed_seconds']} seconds**",
        "",
        "### Chunk Types",
        "",
        "| Type | Count | Meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "text": "Recursive text Chunk within one Markdown Section",
        "table_group": "Complete source table or merged multi-unit Table Group",
        "list": "Complete Markdown list unit",
        "code": "Complete code block",
        "equation": "Equation unit",
    }
    for chunk_type, count in sorted(type_counts.items()):
        report_lines.append(
            f"| `{chunk_type}` | {count} | {meanings.get(chunk_type, 'Packed special/mixed units')} |"
        )
    report_lines.extend(
        [
            "",
            "## Per-document Statistics",
            "",
            "| Document | Pages | Tables before→after | Multi-unit groups | Chunks | Types | Time (s) |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for item in document_summaries:
        types = ", ".join(
            f"{key}={value}" for key, value in item["chunk_type_counts"].items()
        )
        report_lines.append(
            f"| `{item['document_name']}` | {item['pages']} | "
            f"{item['tables_before_grouping']}→{item['tables_after_grouping']} | "
            f"{item['multi_unit_table_groups']} | {item['chunk_count']} | "
            f"{types} | {item['elapsed_seconds']} |"
        )
    report_lines.extend(
        [
            "",
            "## JSONL Record Contract",
            "",
            "Each `chunks.jsonl` line contains the complete original Chunk text, Dense input, Sparse input, type, offsets, page metadata, Section path, and Table Group/Unit metadata. No text is truncated.",
            "",
            "```json",
            "{",
            '  "chunk_type": "table_group",',
            '  "text": "complete original HTML/context returned after retrieval",',
            '  "dense_index_text": "exact text that would be embedded",',
            '  "sparse_index_text": "visible row text/context that BM25 would index",',
            '  "metadata": {"page_start": 1, "page_end": 1, "unit_count": 2}',
            "}",
            "```",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    sample_lines = [
        "# Chunk Samples",
        "",
        "These are representative records only. `chunks.jsonl` contains every full, untruncated record.",
        "",
    ]
    for chunk_type, record in sorted(samples.items()):
        metadata = record["metadata"]
        sample_lines.extend(
            [
                f"## {chunk_type}",
                "",
                f"- Document: `{record['document_name']}`",
                f"- Chunk ID: `{record['chunk_id']}`",
                f"- Page: {metadata.get('page_start')}–{metadata.get('page_end')}",
                f"- Offset: `[{record['start_offset']}, {record['end_offset']})`",
                f"- Characters: {record['char_count']}",
                f"- Unit types: `{metadata.get('unit_types')}`",
                f"- Unit count: {metadata.get('unit_count', 1)}",
                "",
                "### Original returned text",
                "",
                "````text",
                record["text"][:4000],
                "````",
                "",
                "### Dense input",
                "",
                "````text",
                record["dense_index_text"][:4000],
                "````",
                "",
                "### Sparse input",
                "",
                "````text",
                record["sparse_index_text"][:4000],
                "````",
                "",
            ]
        )
    (output_dir / "SAMPLES.md").write_text(
        "\n".join(sample_lines),
        encoding="utf-8",
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "config/settings.yaml")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = export_chunks(
        settings_path=args.settings,
        cases_path=args.cases,
        annotations_path=args.annotations,
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
