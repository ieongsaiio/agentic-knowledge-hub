"""Re-embed table children without their generated Section path prefix."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from scripts.reembed_collection import TargetRecord, read_target_records
from src.core.settings import load_settings, resolve_path
from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.splitter.structured_markdown_splitter import (
    strip_table_child_section_path,
)
from src.observability.evaluation.experiment_runner import ExperimentRunner


def strip_leading_section_path(text: str) -> str:
    """Expose the production transformation for migration callers and tests."""
    return strip_table_child_section_path(text)


def build_dense_chunks(records: Iterable[TargetRecord]) -> list[Chunk]:
    """Build dense-only inputs while preserving stored Chroma documents."""
    chunks: list[Chunk] = []
    for record in records:
        dense_text = strip_leading_section_path(record.document)
        if not dense_text.strip():
            raise RuntimeError(f"Removing the Section path emptied {record.id}")
        chunks.append(
            Chunk(
                id=record.id,
                text=record.document,
                metadata=dict(record.metadata),
                source_ref=str(record.metadata.get("source_ref", "")),
                dense_index_text=dense_text,
            )
        )
    return chunks


def _effective_settings(settings: Any, experiment_name: str) -> Any:
    if not experiment_name:
        return settings
    plans = ExperimentRunner(settings, lambda _plan: None).plan([experiment_name])
    return plans[0].settings


def _write_backup(
    backup_dir: str | Path,
    collection_name: str,
    records: list[TargetRecord],
) -> Path:
    root = resolve_path(backup_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = root / f"{collection_name}_table_children_{stamp}.json.gz"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection_name,
        "record_count": len(records),
        "records": [asdict(record) for record in records],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def _update_embeddings(
    collection: Any,
    records: list[TargetRecord],
    vectors: list[list[float]],
    target_source_type: str,
) -> None:
    metadatas: list[dict[str, Any]] = []
    for record in records:
        metadata = dict(record.metadata)
        metadata["embedding_source_type"] = target_source_type
        metadata["dense_section_path_removed"] = True
        metadata["dense_index_sha256"] = hashlib.sha256(
            strip_leading_section_path(record.document).encode("utf-8")
        ).hexdigest()
        metadatas.append(metadata)
    collection.update(
        ids=[record.id for record in records],
        embeddings=vectors,
        metadatas=metadatas,
    )


def _verify(
    collection: Any,
    records: list[TargetRecord],
    expected_dimension: int,
    target_source_type: str,
    original_count: int,
) -> None:
    if collection.count() != original_count:
        raise RuntimeError("Collection count changed during table-child migration")
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
    for record in records:
        document, metadata, embedding = by_id[record.id]
        if document != record.document:
            raise RuntimeError(f"Stored document changed for {record.id}")
        if metadata.get("embedding_source_type") != target_source_type:
            raise RuntimeError(f"Source type was not updated for {record.id}")
        if len(embedding) != expected_dimension:
            raise RuntimeError(f"Embedding dimension changed for {record.id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--experiment", default="")
    parser.add_argument("--source-type", default="original_table_child")
    parser.add_argument(
        "--target-source-type",
        default="original_table_child_no_section",
    )
    parser.add_argument(
        "--backup-dir",
        default="data/migrations/reembedding",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    base_settings = load_settings(args.settings)
    settings = _effective_settings(base_settings, args.experiment)
    client = chromadb.PersistentClient(
        path=str(resolve_path(settings.vector_store.persist_directory))
    )
    collection = client.get_collection(args.collection)
    original_count = collection.count()
    records = read_target_records(collection, args.source_type)
    chunks = build_dense_chunks(records)
    changed = sum(
        chunk.dense_index_text != record.document
        for chunk, record in zip(chunks, records)
    )
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "collection": args.collection,
        "collection_count": original_count,
        "matched": len(records),
        "section_paths_removed": changed,
        "embedding_model": settings.embedding.model,
        "embedding_dimensions": settings.embedding.dimensions,
        "source_type": args.source_type,
        "target_source_type": args.target_source_type,
    }
    if not records or not args.apply:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    encoder = DenseEncoder(
        EmbeddingFactory.create(settings),
        batch_size=int(settings.ingestion.batch_size),
    )
    vectors = encoder.encode(chunks)
    expected_dimension = int(settings.embedding.dimensions)
    if any(len(vector) != expected_dimension for vector in vectors):
        raise RuntimeError("Embedding API returned an unexpected dimension")
    backup_path = _write_backup(args.backup_dir, args.collection, records)
    _update_embeddings(
        collection,
        records,
        vectors,
        args.target_source_type,
    )
    _verify(
        collection,
        records,
        expected_dimension,
        args.target_source_type,
        original_count,
    )
    summary["updated"] = len(records)
    summary["backup_path"] = str(backup_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
