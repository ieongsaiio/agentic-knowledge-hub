"""Tests for converting PaddleOCR artifacts to the neutral parse contract."""

from __future__ import annotations

from src.libs.loader.paddle_artifact_normalizer import PaddleArtifactNormalizer


def _page(
    page_index: int,
    markdown: str,
    blocks: list[dict[str, object]] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_page: dict[str, object] = {"res": {"page_index": page_index}}
    if blocks is not None:
        raw_page["res"]["prunedResult"] = {"parsing_res_list": blocks}  # type: ignore[index]
    restructured = {
        "page_index": page_index,
        "markdown_text": markdown,
        "json": {"page_index": page_index},
    }
    return raw_page, restructured


def _artifact(
    *page_specs: tuple[dict[str, object], dict[str, object]],
) -> dict[str, object]:
    return {
        "pages": [spec[0] for spec in page_specs],
        "restructured_pages": [spec[1] for spec in page_specs],
    }


def test_normalizes_pages_and_attaches_adjacent_table_context() -> None:
    table = "<table><tr><td>Revenue</td><td>100</td></tr></table>"
    raw_page, restructured = _page(
        0,
        f"# Results\n\nFinancial Results\n\n{table}\n\nAmounts in millions.",
        [
            {
                "block_label": "doc_title",
                "block_content": "Results",
                "block_bbox": [0, 0, 100, 20],
                "block_order": 0,
            },
            {
                "block_label": "table_title",
                "block_content": "Financial Results",
                "block_bbox": [0, 30, 100, 45],
                "block_order": 1,
            },
            {
                "block_label": "table",
                "block_content": table,
                "block_bbox": [0, 50, 100, 150],
                "block_order": 2,
            },
            {
                "block_label": "vision_footnote",
                "block_content": "Amounts in millions.",
                "block_bbox": [0, 155, 100, 170],
                "block_order": 3,
            },
            {
                "block_label": "paragraph",
                "block_content": "Revenue increased.",
                "block_order": 4,
            },
        ],
    )

    result = PaddleArtifactNormalizer().normalize(_artifact((raw_page, restructured)))

    assert result.provider == "paddle"
    assert result.pages[0].page_index == 0
    assert [block.type for block in result.pages[0].blocks] == [
        "title",
        "table",
        "text",
    ]
    table_block = result.pages[0].blocks[1]
    assert table_block.content == table
    assert table_block.caption == ["Financial Results"]
    assert table_block.footnotes == ["Amounts in millions."]
    assert table_block.bbox == [0.0, 50.0, 100.0, 150.0]
    assert table_block.order == 2
    assert table_block.metadata["source_block_label"] == "table"


def test_only_immediately_adjacent_caption_and_footnotes_are_claimed() -> None:
    raw_page, restructured = _page(
        2,
        "Unrelated title\n\nIntervening text\n\n| A | B |\n|---|---|\n| 1 | 2 |",
        [
            {"block_label": "table_title", "block_content": "Unrelated title"},
            {"block_label": "paragraph", "block_content": "Intervening text"},
            {"block_label": "table", "block_content": "<table></table>"},
            {"block_label": "footnote", "block_content": "First note"},
            {"block_label": "vision_footnote", "block_content": "Second note"},
            {"block_label": "paragraph", "block_content": "Following text"},
            {"block_label": "footnote", "block_content": "Detached note"},
        ],
    )

    result = PaddleArtifactNormalizer().normalize(_artifact((raw_page, restructured)))
    blocks = result.pages[0].blocks
    table_block = next(block for block in blocks if block.type == "table")

    assert result.pages[0].page_index == 2
    assert table_block.caption == []
    assert table_block.footnotes == ["First note", "Second note"]
    assert any(block.content == "Unrelated title" for block in blocks)
    assert any(block.content == "Detached note" for block in blocks)
    assert not any(block.content == "First note" for block in blocks)
    assert not any(block.content == "Second note" for block in blocks)


def test_maps_supported_special_blocks_and_preserves_source_label() -> None:
    labels_and_expected = [
        ("text", "text"),
        ("paragraph_title", "title"),
        ("list", "list"),
        ("code_block", "code"),
        ("formula", "equation"),
        ("figure", "image"),
        ("header", "page_header"),
        ("footer", "page_footer"),
        ("number", "page_number"),
        ("aside_text", "page_aside_text"),
    ]
    blocks = [
        {
            "block_label": label,
            "block_content": f"content-{index}",
            "block_order": index,
        }
        for index, (label, _) in enumerate(labels_and_expected)
    ]
    raw_page, restructured = _page(0, "page markdown", blocks)

    result = PaddleArtifactNormalizer().normalize(_artifact((raw_page, restructured)))

    assert [block.type for block in result.pages[0].blocks] == [
        expected for _, expected in labels_and_expected
    ]
    assert [block.metadata["source_block_label"] for block in result.pages[0].blocks] == [
        label for label, _ in labels_and_expected
    ]


def test_sorts_pages_and_uses_markdown_fallback_without_parsing_blocks() -> None:
    raw_page_1, restructured_1 = _page(1, "Second page markdown", None)
    raw_page_0, restructured_0 = _page(0, "First page markdown", None)
    artifact = _artifact(
        (raw_page_1, restructured_1),
        (raw_page_0, restructured_0),
    )

    result = PaddleArtifactNormalizer().normalize(artifact)

    assert [page.page_index for page in result.pages] == [0, 1]
    assert [page.blocks[0].content for page in result.pages] == [
        "First page markdown",
        "Second page markdown",
    ]
    assert all(page.blocks[0].type == "text" for page in result.pages)
    assert all(page.blocks[0].metadata["fallback"] is True for page in result.pages)
    assert result.raw_markdown == "First page markdown\n\nSecond page markdown"
