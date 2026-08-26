"""Tests for canonical Markdown assembly from normalized parser output."""

from src.libs.loader.canonical_document_assembler import CanonicalDocumentAssembler
from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage


def test_assembles_stable_markdown_offsets_and_metadata() -> None:
    parsed = ParsedDocument(
        schema_version=1,
        provider="mineru",
        parser_version="vlm-1",
        pages=[
            ParsedPage(
                page_index=1,
                blocks=[
                    ParsedBlock(
                        block_id="p1_text",
                        type="text",
                        content="Second page.",
                        page_index=1,
                        order=0,
                    )
                ],
            ),
            ParsedPage(
                page_index=0,
                width=1000,
                height=1400,
                blocks=[
                    ParsedBlock(
                        block_id="p0_table",
                        type="table",
                        content="<table><tr><td>10</td></tr></table>",
                        page_index=0,
                        order=2,
                        bbox=[10, 30, 900, 600],
                        caption=["Revenue"],
                        footnotes=["USD millions"],
                    ),
                    ParsedBlock(
                        block_id="p0_title",
                        type="title",
                        content="Results",
                        page_index=0,
                        order=0,
                        level=2,
                    ),
                    ParsedBlock(
                        block_id="p0_text",
                        type="text",
                        content="First page.",
                        page_index=0,
                        order=1,
                    ),
                ],
            ),
        ],
        raw_artifact={"content_list": [{"type": "text"}]},
    )

    document = CanonicalDocumentAssembler().assemble(
        parsed,
        source_path="reports/example.pdf",
        doc_id="doc_123",
        doc_hash="abc123",
    )

    assert document.text == (
        "## Results\n\n"
        "First page.\n\n"
        "Revenue\n\n"
        "<table><tr><td>10</td></tr></table>\n\n"
        "USD millions\n\n"
        "Second page."
    )
    assert document.id == "doc_123"
    assert document.metadata["doc_hash"] == "abc123"
    assert document.metadata["doc_type"] == "pdf"
    assert document.metadata["page_count"] == 2
    assert document.metadata["parser_provider"] == "mineru"
    assert document.metadata["parser_version"] == "vlm-1"
    assert document.metadata["parsed_source_artifact"] == parsed.raw_artifact

    first_page, second_page = document.metadata["page_spans"]
    assert first_page == {
        "page": 1,
        "page_index": 0,
        "start_offset": 0,
        "end_offset": document.text.index("Second page.") - 2,
    }
    assert document.text[
        second_page["start_offset"] : second_page["end_offset"]
    ] == "Second page."

    structure = document.metadata["parsed_structure"]
    table = next(block for block in structure["blocks"] if block["block_id"] == "p0_table")
    assert document.text[table["start_offset"] : table["end_offset"]] == (
        "Revenue\n\n"
        "<table><tr><td>10</td></tr></table>\n\n"
        "USD millions"
    )
    assert table["page_index"] == 0
    assert table["bbox"] == [10.0, 30.0, 900.0, 600.0]
    assert table["caption"] == ["Revenue"]
    assert table["footnotes"] == ["USD millions"]


def test_ignored_blocks_do_not_create_text_or_offsets() -> None:
    parsed = ParsedDocument(
        schema_version=1,
        provider="mineru",
        pages=[
            ParsedPage(
                page_index=0,
                blocks=[
                    ParsedBlock("header", "page_header", "Annual report", 0, order=0),
                    ParsedBlock("body", "text", "Useful content.", 0, order=1),
                    ParsedBlock("footer", "page_footer", "Page 1", 0, order=2),
                ],
            )
        ],
    )

    document = CanonicalDocumentAssembler(
        ignored_block_types={"page_header", "page_footer"}
    ).assemble(parsed, source_path="report.pdf", doc_id="doc_report")

    assert document.text == "Useful content."
    assert [block["block_id"] for block in document.metadata["parsed_structure"]["blocks"]] == [
        "body"
    ]
    assert document.metadata["page_spans"] == [
        {
            "page": 1,
            "page_index": 0,
            "start_offset": 0,
            "end_offset": len("Useful content."),
        }
    ]


def test_title_with_existing_marker_is_not_duplicated() -> None:
    parsed = ParsedDocument(
        schema_version=1,
        provider="fixture",
        pages=[
            ParsedPage(
                page_index=0,
                blocks=[ParsedBlock("title", "title", "### Existing", 0, level=2)],
            )
        ],
    )

    document = CanonicalDocumentAssembler().assemble(
        parsed,
        source_path="notes.md",
        doc_id="doc_notes",
        doc_type="markdown",
    )

    assert document.text == "### Existing"
    assert document.metadata["doc_type"] == "markdown"


def test_empty_page_keeps_a_zero_length_page_span() -> None:
    parsed = ParsedDocument(
        schema_version=1,
        provider="fixture",
        pages=[
            ParsedPage(page_index=0, blocks=[]),
            ParsedPage(
                page_index=1,
                blocks=[ParsedBlock("text", "text", "Page two", 1)],
            ),
        ],
    )

    document = CanonicalDocumentAssembler().assemble(
        parsed,
        source_path="two-pages.pdf",
        doc_id="doc_two_pages",
    )

    assert document.metadata["page_spans"][0]["start_offset"] == 0
    assert document.metadata["page_spans"][0]["end_offset"] == 0
    assert document.metadata["page_spans"][1]["start_offset"] == 0
    assert document.text == "Page two"


def test_special_blocks_are_rendered_as_standard_markdown() -> None:
    parsed = ParsedDocument(
        schema_version=1,
        provider="mineru",
        pages=[
            ParsedPage(
                page_index=0,
                blocks=[
                    ParsedBlock(
                        "list-1",
                        "list",
                        "Revenue\nNet income",
                        0,
                        order=0,
                    ),
                    ParsedBlock(
                        "code-1",
                        "code",
                        "print('ok')",
                        0,
                        order=1,
                        metadata={"language": "python"},
                    ),
                    ParsedBlock(
                        "equation-1",
                        "equation",
                        "x = y + 1",
                        0,
                        order=2,
                    ),
                ],
            )
        ],
    )

    document = CanonicalDocumentAssembler().assemble(
        parsed,
        source_path="sample.pdf",
        doc_id="doc_sample",
    )

    assert "- Revenue\n- Net income" in document.text
    assert "```python\nprint('ok')\n```" in document.text
    assert "$$\nx = y + 1\n$$" in document.text
