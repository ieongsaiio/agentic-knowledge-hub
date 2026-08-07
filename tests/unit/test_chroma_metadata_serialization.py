"""Tests for Chroma metadata serialization used by multimodal retrieval."""

from __future__ import annotations

import json
from unittest.mock import Mock

from src.libs.vector_store.chroma_store import ChromaStore


def test_multimodal_metadata_is_serialized_as_json() -> None:
    store = object.__new__(ChromaStore)
    metadata = {
        "images": [{"id": "img1", "path": "images/img1.png"}],
        "image_captions": [{"id": "img1", "caption": "A diagram"}],
        "tags": ["rag", "mcp"],
    }

    sanitized = store._sanitize_metadata(metadata)

    assert json.loads(sanitized["images"]) == metadata["images"]
    assert json.loads(sanitized["image_captions"]) == metadata["image_captions"]
    assert sanitized["tags"] == "rag,mcp"


def test_upsert_stores_document_once_outside_metadata() -> None:
    store = object.__new__(ChromaStore)
    store.collection = Mock()
    store.collection_name = "test"
    raw_text = "<table><tr><td>Revenue</td><td>100</td></tr></table>"

    store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [0.1, 0.2],
                "document": raw_text,
                "metadata": {"source_path": "report.pdf", "page_num": 1},
            }
        ]
    )

    payload = store.collection.upsert.call_args.kwargs
    assert payload["documents"] == [raw_text]
    assert "text" not in payload["metadatas"][0]
