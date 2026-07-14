"""Tests for Chroma metadata serialization used by multimodal retrieval."""

from __future__ import annotations

import json

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
