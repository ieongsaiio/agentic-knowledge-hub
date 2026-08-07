"""Re-embed selected Chroma records without rebuilding the whole index.

The migration rebuilds dense index text from parsed documents and the current
splitter. It refuses to update a record unless its source document, chunk index,
and original Chroma document all match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import chromadb

from src.core.settings import load_settings, resolve_path
from src.core.types import Chunk, Document
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.libs.embedding.embedding_factory import EmbeddingFactory


@dataclass(frozen=True)
class TargetRecord:
    """Original Chroma state needed for validation and rollback."""

    id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]


def read_target_records(
    collection: Any,
    source_type: str,
) -> list[TargetRecord]:
    """Read records selected by their current embedding source type."""
    payload = collection.get(
        where={"embedding_source_type": source_type},
        include=["documents", "metadatas", "embeddings"],
    )
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    embeddings = payload.get("embeddings")
    if embeddings is None:
        embeddings = []
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise RuntimeError("Chroma returned misaligned target record fields")

    return [
        TargetRecord(
            id=str(record_id),
            document=str(document),
            metadata=dict(metadata or {}),
            embedding=[float(value) for value in embedding],
        )
        for record_id, document, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
        )
    ]


def load_cached_documents(
    cache_dir: str | Path,
    document_ids: set[str],
) -> dict[str, Document]:
    """Load requested Documents directly from schema-v3 parsed cache files."""
    root = resolve_path(cache_dir)
    found: dict[str, Document] = {}

    for document_id in sorted(document_ids):
        hash_prefix = document_id.removeprefix("doc_")
        candidates = sorted(root.glob(f"{hash_prefix}*.json"))
        if not candidates:
            candidates = sorted(root.glob("*.json"))
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                document_data = payload.get("document")
                if not isinstance(document_data, dict):
                    continue
                if document_data.get("id") != document_id:
                    continue
                found[document_id] = Document(
                    id=document_data["id"],
                    text=document_data["text"],
                    metadata=document_data["metadata"],
                )
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue

    missing = sorted(document_ids - found.keys())
    if missing:
        raise RuntimeError(
            "Parsed cache is missing target document(s): " + ", ".join(missing)
        )
    return found


def build_migration_settings(settings: Any) -> Any:
    """Use original table text and prohibit an LLM summary fallback."""
    structured = dict(settings.ingestion.structured_chunking or {})
    table_summary = dict(structured.get("table_summary") or {})
    table_summary["enabled"] = False
    structured.update(
        {
            "embedding_max_tokens": int(settings.embedding.max_tokens),
            "table_dense_representation": "original",
            "table_summary": table_summary,
        }
    )
    return replace(
        settings,
        ingestion=replace(
            settings.ingestion,
            structured_chunking=structured,
        ),
    )


def rebuild_document_chunks(
    documents: dict[str, Document],
    settings: Any,
) -> tuple[dict[str, list[Chunk]], DocumentChunker]:
    """Re-run deterministic chunking only; no loader or OCR is invoked."""
    chunker = DocumentChunker(settings)
    rebuilt = {
        document_id: chunker.split_document(document)
        for document_id, document in documents.items()
    }
    return rebuilt, chunker


def match_rebuilt_chunks(
    records: Iterable[TargetRecord],
    rebuilt_by_document: dict[str, list[Chunk]],
) -> list[tuple[TargetRecord, Chunk]]:
    """Match targets by document and index, then require exact raw text."""
    matched: list[tuple[TargetRecord, Chunk]] = []
    for record in records:
        source_ref = record.metadata.get("source_ref")
        chunk_index = record.metadata.get("chunk_index")
        if not isinstance(source_ref, str) or not isinstance(chunk_index, int):
            raise RuntimeError(
                f"Target {record.id} lacks source_ref or integer chunk_index"
            )
        chunks = rebuilt_by_document.get(source_ref)
        if chunks is None:
            raise RuntimeError(
                f"No rebuilt chunks are available for {source_ref}"
            )
        candidates = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("chunk_index") == chunk_index
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Target {record.id} matched {len(candidates)} rebuilt chunks"
            )
        chunk = candidates[0]
        if chunk.text != record.document:
            raise RuntimeError(
                f"Target {record.id} rebuilt text does not match Chroma document"
            )
        matched.append((record, chunk))
    return matched


def validate_dense_inputs(
    matched: Iterable[tuple[TargetRecord, Chunk]],
    chunker: DocumentChunker,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Require original-table inputs that fit the configured model limit."""
    length_function = getattr(chunker._splitter, "_length", None)
    if not callable(length_function):
        raise RuntimeError("Configured splitter cannot measure dense input tokens")

    diagnostics: list[dict[str, Any]] = []
    for record, chunk in matched:
        dense_text = chunk.dense_index_text or chunk.text
        source_type = chunk.metadata.get("embedding_source_type")
        token_count = int(length_function(dense_text))
        if source_type != "original_table":
            raise RuntimeError(
                f"Target {record.id} rebuilt as {source_type!r}, not 'original_table'"
            )
        if token_count > max_tokens:
            raise RuntimeError(
                f"Target {record.id} has {token_count} tokens; "
                f"embedding limit is {max_tokens}"
            )
        diagnostics.append(
            {
                "id": record.id,
                "source_ref": record.metadata.get("source_ref"),
                "chunk_index": record.metadata.get("chunk_index"),
                "document_chars": len(record.document),
                "dense_chars": len(dense_text),
                "dense_tokens": token_count,
                "dense_sha256": hashlib.sha256(
                    dense_text.encode("utf-8")
                ).hexdigest(),
            }
        )
    return diagnostics


def update_target_embeddings(
    collection: Any,
    records: list[TargetRecord],
    vectors: list[list[float]],
    *,
    target_source_type: str,
) -> None:
    """Update embeddings and one metadata field, leaving documents untouched."""
    if len(records) != len(vectors):
        raise ValueError("Target record and vector counts must match")
    metadatas = []
    for record in records:
        metadata = dict(record.metadata)
        metadata["embedding_source_type"] = target_source_type
        metadatas.append(metadata)
    collection.update(
        ids=[record.id for record in records],
        embeddings=vectors,
        metadatas=metadatas,
    )


def write_backup(
    backup_dir: str | Path,
    collection_name: str,
    records: list[TargetRecord],
    diagnostics: list[dict[str, Any]],
) -> Path:
    """Persist the old vectors and metadata before an in-place update."""
    root = resolve_path(backup_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = root / f"{collection_name}_{stamp}.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection_name,
        "record_count": len(records),
        "records": [
            {
                **asdict(record),
                "document_sha256": hashlib.sha256(
                    record.document.encode("utf-8")
                ).hexdigest(),
                "diagnostic": diagnostic,
            }
            for record, diagnostic in zip(records, diagnostics)
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _verify_update(
    collection: Any,
    records: list[TargetRecord],
    vectors: list[list[float]],
    target_source_type: str,
    original_count: int,
) -> None:
    if collection.count() != original_count:
        raise RuntimeError("Collection count changed during re-embedding")
    payload = collection.get(
        ids=[record.id for record in records],
        include=["documents", "metadatas", "embeddings"],
    )
    by_id = {
        str(record_id): (document, metadata, embedding)
        for record_id, document, metadata, embedding in zip(
            payload["ids"],
            payload["documents"],
            payload["metadatas"],
            payload["embeddings"],
        )
    }
    for record, expected_vector in zip(records, vectors):
        document, metadata, embedding = by_id[record.id]
        if document != record.document:
            raise RuntimeError(f"Document changed for {record.id}")
        if metadata.get("embedding_source_type") != target_source_type:
            raise RuntimeError(f"Metadata was not updated for {record.id}")
        if len(embedding) != len(expected_vector):
            raise RuntimeError(f"Embedding dimension changed for {record.id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-embed selected Chroma records from parsed cache."
    )
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--parsed-dir", default="data/parsed")
    parser.add_argument(
        "--source-type",
        default="llm_table_summary",
    )
    parser.add_argument(
        "--target-source-type",
        default="original_table",
    )
    parser.add_argument(
        "--backup-dir",
        default="data/migrations/reembedding",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Call the embedding API and update Chroma. Omit for dry-run.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = load_settings(args.settings)
    migration_settings = build_migration_settings(settings)
    client = chromadb.PersistentClient(
        path=str(resolve_path(settings.vector_store.persist_directory))
    )
    collection = client.get_collection(args.collection)
    original_count = collection.count()
    records = read_target_records(collection, args.source_type)
    if not records:
        print(
            json.dumps(
                {
                    "mode": "apply" if args.apply else "dry-run",
                    "collection": args.collection,
                    "matched": 0,
                    "message": "No matching records; nothing to do.",
                }
            )
        )
        return 0

    document_ids = {
        str(record.metadata.get("source_ref"))
        for record in records
        if record.metadata.get("source_ref")
    }
    documents = load_cached_documents(args.parsed_dir, document_ids)
    rebuilt, chunker = rebuild_document_chunks(documents, migration_settings)
    matched = match_rebuilt_chunks(records, rebuilt)
    diagnostics = validate_dense_inputs(
        matched,
        chunker,
        int(settings.embedding.max_tokens),
    )
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "collection": args.collection,
        "collection_count": original_count,
        "matched": len(records),
        "documents": len(documents),
        "source_type": args.source_type,
        "target_source_type": args.target_source_type,
        "embedding_model": settings.embedding.model,
        "embedding_dimensions": settings.embedding.dimensions,
        "maximum_dense_tokens": max(
            diagnostic["dense_tokens"] for diagnostic in diagnostics
        ),
        "records": diagnostics,
    }
    if not args.apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    chunks = [chunk for _, chunk in matched]
    encoder = DenseEncoder(
        EmbeddingFactory.create(migration_settings),
        batch_size=int(settings.ingestion.batch_size),
    )
    vectors = encoder.encode(chunks)
    expected_dimension = int(settings.embedding.dimensions)
    if any(len(vector) != expected_dimension for vector in vectors):
        raise RuntimeError(
            f"Embedding API returned a dimension other than {expected_dimension}"
        )
    backup_path = write_backup(
        args.backup_dir,
        args.collection,
        records,
        diagnostics,
    )
    update_target_embeddings(
        collection,
        records,
        vectors,
        target_source_type=args.target_source_type,
    )
    _verify_update(
        collection,
        records,
        vectors,
        args.target_source_type,
        original_count,
    )
    summary["backup_path"] = str(backup_path)
    summary["updated"] = len(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
