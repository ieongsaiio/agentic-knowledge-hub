"""Tests for parsed document cache."""

from pathlib import Path

from src.core.types import Document
from src.libs.loader.parsed_document_cache import ParsedDocumentCache


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = ParsedDocumentCache(
        cache_dir=tmp_path,
        loader_config={"loader": "test", "extract_images": False},
    )
    document = Document(
        id="doc_abc",
        text="page one\n\npage two",
        metadata={
            "source_path": "original.pdf",
            "doc_type": "pdf",
            "doc_hash": "abc",
            "page_count": 2,
            "page_spans": [
                {"page": 1, "start_offset": 0, "end_offset": 8},
                {"page": 2, "start_offset": 10, "end_offset": 18},
            ],
        },
    )

    path = cache.put("abc", document)
    assert path.exists()

    restored = cache.get("abc")
    assert restored == document


def test_cache_key_changes_with_loader_config(tmp_path: Path) -> None:
    cache_a = ParsedDocumentCache(
        cache_dir=tmp_path,
        loader_config={"loader": "test", "extract_images": False},
    )
    cache_b = ParsedDocumentCache(
        cache_dir=tmp_path,
        loader_config={"loader": "test", "extract_images": True},
    )

    assert cache_a.cache_path("abc") != cache_b.cache_path("abc")


def test_cache_get_updates_source_path(tmp_path: Path) -> None:
    cache = ParsedDocumentCache(
        cache_dir=tmp_path,
        loader_config={"loader": "test", "extract_images": False},
    )
    document = Document(
        id="doc_abc",
        text="content",
        metadata={"source_path": "old.pdf", "doc_type": "pdf"},
    )
    cache.put("abc", document)

    restored = cache.get("abc", source_path=Path("new.pdf"))

    assert restored is not None
    assert restored.metadata["source_path"] == "new.pdf"
