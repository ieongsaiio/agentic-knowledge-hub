"""Real-data checks against PaddleOCR tables currently stored in Chroma."""

from __future__ import annotations

import html
from pathlib import Path

import chromadb
import pytest

from src.libs.splitter.html_table_chunker import HTMLTableChunker

COLLECTION = "financebench__e4473342e89c"
REAL_TABLE_IDS = (
    "doc_fe9126b17c3656e5_c5d59391_0042_3c3bf94f",
    "doc_7d715f552d4eeb6b_4d2caf6d_0102_9d70c601",
    "doc_84d7a0291a12f4e6_06cbd0c5_0168_b008ed2a",
)


@pytest.mark.integration
def test_real_financebench_tables_form_valid_overlapping_children() -> None:
    chroma_path = Path("data/db/chroma")
    if not chroma_path.exists():
        pytest.skip("Local FinanceBench Chroma database is not available")

    client = chromadb.PersistentClient(path=str(chroma_path))
    if COLLECTION not in {item.name for item in client.list_collections()}:
        pytest.skip(f"Collection {COLLECTION!r} is not available")

    collection = client.get_collection(COLLECTION)
    records = collection.get(
        ids=list(REAL_TABLE_IDS),
        include=["documents", "metadatas"],
    )
    if len(records["ids"]) != len(REAL_TABLE_IDS):
        pytest.skip("One or more real FinanceBench table fixtures are unavailable")

    chunker = HTMLTableChunker(
        target_children=4,
        overlap_rows=1,
        repeated_context_rows=2,
    )
    for document, metadata in zip(records["documents"], records["metadatas"]):
        preferred_title = str(metadata.get("table_title") or "")
        title = chunker.extract_caption(
            document,
            preferred_title=preferred_title,
        )
        source = chunker.parse(document)
        children = chunker.split(document, title=title)

        assert 3 <= len(children) <= 5
        assert source.width > 1
        assert len(source.rows) > 5
        covered_source_rows = {
            row_index
            for child in children
            for row_index in (
                child.repeated_context_row_indices + child.source_row_indices
            )
        }
        assert covered_source_rows == set(range(len(source.rows)))

        for previous, current in zip(children, children[1:]):
            if (
                previous.repeated_context_row_indices
                == current.repeated_context_row_indices
            ):
                assert set(previous.source_row_indices) & set(
                    current.source_row_indices
                )
        for child in children:
            parsed_child = chunker.parse(child.html)
            assert parsed_child.width == source.width
            expected_rows = tuple(
                source.rows[row_index]
                for row_index in (
                    child.repeated_context_row_indices + child.source_row_indices
                )
            )
            assert parsed_child.rows == expected_rows
            if title:
                assert html.escape(title) in child.html
            assert "Section:" not in child.html
