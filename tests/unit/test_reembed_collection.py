import json
from pathlib import Path

import pytest

from scripts.reembed_collection import (
    TargetRecord,
    load_cached_documents,
    match_rebuilt_chunks,
    read_target_records,
    update_target_embeddings,
)
from src.core.types import Chunk


class _FakeCollection:
    def __init__(self) -> None:
        self.updated = None

    def get(self, **kwargs):
        assert kwargs["where"] == {
            "embedding_source_type": "llm_table_summary"
        }
        return {
            "ids": ["chunk-1"],
            "documents": ["<table>Revenue</table>"],
            "metadatas": [
                {
                    "source_ref": "doc-1",
                    "chunk_index": 3,
                    "embedding_source_type": "llm_table_summary",
                }
            ],
            "embeddings": [[0.1, 0.2]],
        }

    def update(self, **kwargs):
        self.updated = kwargs


def test_read_target_records_preserves_original_record() -> None:
    collection = _FakeCollection()

    records = read_target_records(collection, "llm_table_summary")

    assert records == [
        TargetRecord(
            id="chunk-1",
            document="<table>Revenue</table>",
            metadata={
                "source_ref": "doc-1",
                "chunk_index": 3,
                "embedding_source_type": "llm_table_summary",
            },
            embedding=[0.1, 0.2],
        )
    ]


def test_load_cached_documents_finds_requested_document(tmp_path: Path) -> None:
    payload = {
        "cache_schema_version": 3,
        "document": {
            "id": "doc-1",
            "text": "# Results\n\nRevenue increased.",
            "metadata": {"source_path": "report.pdf"},
        },
    }
    (tmp_path / "hash_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    documents = load_cached_documents(tmp_path, {"doc-1"})

    assert documents["doc-1"].text == "# Results\n\nRevenue increased."
    assert documents["doc-1"].metadata["source_path"] == "report.pdf"


def test_load_cached_documents_rejects_missing_document(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="doc-missing"):
        load_cached_documents(tmp_path, {"doc-missing"})


def test_match_rebuilt_chunks_requires_exact_original_text() -> None:
    record = TargetRecord(
        id="stored-id",
        document="<table>Revenue</table>",
        metadata={"source_ref": "doc-1", "chunk_index": 3},
        embedding=[0.1, 0.2],
    )
    rebuilt = Chunk(
        id="rebuilt-id",
        text="<table>Revenue</table>",
        metadata={
            "source_ref": "doc-1",
            "chunk_index": 3,
            "embedding_source_type": "original_table",
            "source_path": "report.pdf",
        },
        source_ref="doc-1",
        dense_index_text="Section: Results\n\n<table>Revenue</table>",
    )

    matched = match_rebuilt_chunks([record], {"doc-1": [rebuilt]})

    assert matched == [(record, rebuilt)]


def test_match_rebuilt_chunks_rejects_text_mismatch() -> None:
    record = TargetRecord(
        id="stored-id",
        document="<table>Revenue</table>",
        metadata={
            "source_ref": "doc-1",
            "chunk_index": 3,
            "source_path": "report.pdf",
        },
        embedding=[0.1, 0.2],
    )
    rebuilt = Chunk(
        id="rebuilt-id",
        text="<table>Different</table>",
        metadata={
            "source_ref": "doc-1",
            "chunk_index": 3,
            "source_path": "report.pdf",
        },
        source_ref="doc-1",
        dense_index_text="<table>Different</table>",
    )

    with pytest.raises(RuntimeError, match="text does not match"):
        match_rebuilt_chunks([record], {"doc-1": [rebuilt]})


def test_update_target_embeddings_changes_only_vectors_and_source_type() -> None:
    collection = _FakeCollection()
    record = TargetRecord(
        id="chunk-1",
        document="<table>Revenue</table>",
        metadata={
            "source_ref": "doc-1",
            "chunk_index": 3,
            "embedding_source_type": "llm_table_summary",
            "page_num": 7,
        },
        embedding=[0.1, 0.2],
    )

    update_target_embeddings(
        collection,
        [record],
        [[0.8, 0.9]],
        target_source_type="original_table",
    )

    assert collection.updated == {
        "ids": ["chunk-1"],
        "embeddings": [[0.8, 0.9]],
        "metadatas": [
            {
                "source_ref": "doc-1",
                "chunk_index": 3,
                "embedding_source_type": "original_table",
                "page_num": 7,
            }
        ],
    }
