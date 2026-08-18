"""Tests for conservative same-page HTML table continuation merging."""

from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage
from src.libs.loader.table_continuation_merger import TableContinuationMerger


def _table(
    block_id: str,
    html: str,
    bbox: list[float],
    *,
    caption: list[str] | None = None,
    order: int = 0,
) -> ParsedBlock:
    return ParsedBlock(
        block_id=block_id,
        type="table",
        content=html,
        page_index=0,
        bbox=bbox,
        order=order,
        caption=caption or [],
    )


def _document(*blocks: ParsedBlock) -> ParsedDocument:
    return ParsedDocument(
        schema_version=1,
        provider="mineru",
        pages=[ParsedPage(page_index=0, width=1000, height=1000, blocks=list(blocks))],
    )


def test_merges_rows_and_inserts_later_caption_as_full_width_row() -> None:
    first = _table(
        "t1",
        "<table><tr><td></td><td>2018</td><td>2017</td></tr>"
        "<tr><td>Cash</td><td>10</td><td>9</td></tr></table>",
        [20, 100, 980, 400],
        caption=["Consolidated Balance Sheets"],
    )
    second = _table(
        "t2",
        "<table><tr><td>Accounts payable</td><td>4</td><td>3</td></tr></table>",
        [20, 420, 980, 700],
        caption=["Liabilities and Stockholders' Equity"],
        order=1,
    )

    result = TableContinuationMerger().process(_document(first, second))

    assert len(result.pages[0].blocks) == 1
    merged = result.pages[0].blocks[0]
    assert merged.caption == []
    assert merged.content.count("<tr") == 5
    assert 'class="table-section-caption"' in merged.content
    assert 'colspan="3"' in merged.content
    assert "Liabilities and Stockholders&#x27; Equity" in merged.content
    assert "Consolidated Balance Sheets" in merged.content
    assert merged.metadata["unit_count"] == 2
    assert merged.metadata["source_block_ids"] == ["t1", "t2"]
    assert merged.metadata["merged_table"] is True


def test_merges_three_aligned_table_parts() -> None:
    blocks = [
        _table(
            f"t{index}",
            f"<table><tr><td>Part {index}</td><td>{index}</td></tr></table>",
            [20, 100 + index * 200, 980, 280 + index * 200],
            caption=[] if index == 0 else [f"Section {index}"],
            order=index,
        )
        for index in range(3)
    ]

    result = TableContinuationMerger().process(_document(*blocks))

    merged = result.pages[0].blocks[0]
    assert len(result.pages[0].blocks) == 1
    assert merged.content.count("table-section-caption") == 2
    assert merged.metadata["source_block_ids"] == ["t0", "t1", "t2"]


def test_does_not_merge_tables_with_incompatible_widths() -> None:
    first = _table(
        "t1",
        "<table><tr><td>A</td><td>1</td><td>2</td></tr></table>",
        [20, 100, 980, 300],
    )
    second = _table(
        "t2",
        "<table><tr><td>B</td><td>1</td><td>2</td><td>3</td><td>4</td></tr></table>",
        [20, 320, 980, 500],
        order=1,
    )

    result = TableContinuationMerger().process(_document(first, second))

    assert len(result.pages[0].blocks) == 2


def test_does_not_merge_across_intervening_body_text() -> None:
    first = _table(
        "t1",
        "<table><tr><td>A</td><td>1</td></tr></table>",
        [20, 100, 980, 300],
    )
    body = ParsedBlock(
        block_id="p1",
        type="text",
        content="This paragraph starts another subject.",
        page_index=0,
        bbox=[20, 310, 980, 350],
        order=1,
    )
    second = _table(
        "t2",
        "<table><tr><td>B</td><td>2</td></tr></table>",
        [20, 360, 980, 550],
        order=2,
    )

    result = TableContinuationMerger().process(_document(first, body, second))

    assert [block.block_id for block in result.pages[0].blocks] == ["t1", "p1", "t2"]


def test_visual_bbox_order_overrides_unreliable_source_order() -> None:
    table = _table(
        "t1",
        "<table><tr><td>A</td><td>1</td></tr></table>",
        [20, 200, 980, 400],
        order=0,
    )
    title = ParsedBlock(
        block_id="title",
        type="title",
        content="Balance Sheets",
        page_index=0,
        bbox=[300, 40, 700, 80],
        order=1,
        level=2,
    )

    result = TableContinuationMerger().process(_document(table, title))

    assert [block.block_id for block in result.pages[0].blocks] == ["title", "t1"]
