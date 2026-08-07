"""Tests for parsed document cache."""

import hashlib
import json
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


def test_cache_preserves_section_tree(tmp_path: Path) -> None:
    cache = ParsedDocumentCache(
        cache_dir=tmp_path,
        loader_config={"loader": "test"},
    )
    section_tree = {
        "schema_version": 1,
        "root": {
            "id": "doc_tree_section_root",
            "subsections": [],
        },
    }
    document = Document(
        id="doc_tree",
        text="# Report\n\nBody.",
        metadata={
            "source_path": "report.pdf",
            "section_tree": section_tree,
        },
    )

    cache.put("tree_hash", document)
    restored = cache.get("tree_hash")

    assert restored is not None
    assert restored.metadata["section_tree"] == section_tree


def test_cache_upgrades_schema_two_document_without_reparsing_pdf(
    tmp_path: Path,
) -> None:
    current_loader_config = {
        "loader": "PaddlePdfLoader",
        "loader_schema_version": 6,
        "provider": "paddle",
    }
    legacy_loader_config = {
        "loader": "PaddlePdfLoader",
        "loader_schema_version": 5,
        "provider": "paddle",
        "cache_schema_version": 2,
    }
    legacy_hash = hashlib.sha256(
        json.dumps(
            legacy_loader_config,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    legacy_path = tmp_path / f"legacy_file_hash_{legacy_hash}.json"
    legacy_path.write_text(
        json.dumps(
            {
                "cache_schema_version": 2,
                "file_hash": "legacy_file_hash",
                "loader_config": legacy_loader_config,
                "document": {
                    "id": "doc_legacy",
                    "text": "# Report\n\n## Results\n\nRevenue increased.",
                    "metadata": {
                        "source_path": "report.pdf",
                        "page_spans": [
                            {
                                "page": 1,
                                "start_offset": 0,
                                "end_offset": 41,
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cache = ParsedDocumentCache(
        cache_dir=tmp_path,
        loader_config=current_loader_config,
    )

    restored = cache.get("legacy_file_hash")

    assert restored is not None
    assert restored.metadata["section_tree"]["section_count"] == 2
    assert cache.cache_path("legacy_file_hash").exists()
