from __future__ import annotations

from scripts.reembed_collection import TargetRecord
from scripts.reembed_table_children_without_section import (
    build_dense_chunks,
    strip_leading_section_path,
)


def _record(document: str) -> TargetRecord:
    return TargetRecord(
        id="table-child-1",
        document=document,
        metadata={
            "chunk_role": "table_child",
            "embedding_source_type": "original_table_child",
            "header_path": "Report,Results",
            "source_path": "report.pdf",
        },
        embedding=[0.1, 0.2],
    )


def test_strip_leading_section_path_keeps_context_caption_table_and_footnote() -> None:
    document = (
        "Section: Wrong > Path\n\n"
        "Useful preceding context.\n\n"
        "<div class=\"table-title\">Annual Results</div>\n"
        "<table><tr><td>Revenue</td><td>100</td></tr></table>\n\n"
        "Footnote: Amounts are in millions."
    )

    cleaned = strip_leading_section_path(document)

    assert "Section: Wrong > Path" not in cleaned
    assert cleaned.startswith("Useful preceding context.")
    assert "Annual Results" in cleaned
    assert "<table>" in cleaned
    assert "Footnote: Amounts are in millions." in cleaned


def test_strip_leading_section_path_does_not_change_non_section_text() -> None:
    document = "Context.\n\n<table><tr><td>Revenue</td></tr></table>"

    assert strip_leading_section_path(document) == document


def test_build_dense_chunks_preserves_stored_document_and_metadata() -> None:
    record = _record(
        "Section: Wrong > Path\n\n"
        "Context.\n\n<table><tr><td>Revenue</td></tr></table>"
    )

    chunks = build_dense_chunks([record])

    assert len(chunks) == 1
    assert chunks[0].text == record.document
    assert chunks[0].dense_index_text == (
        "Context.\n\n<table><tr><td>Revenue</td></tr></table>"
    )
    assert chunks[0].metadata == record.metadata
