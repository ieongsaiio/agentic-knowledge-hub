"""Tests for deterministic row-group children built from HTML tables."""

from __future__ import annotations

import html

from src.libs.splitter.html_table_chunker import HTMLTableChunker


def test_splits_table_into_overlapping_standalone_html_children() -> None:
    source = """
    <table border="1">
      <tr><td>Metric</td><td>FY2022</td><td>FY2021</td></tr>
      <tr><td>Revenue</td><td>120</td><td>100</td></tr>
      <tr><td>Gross profit</td><td>60</td><td>45</td></tr>
      <tr><td>Operating income</td><td>30</td><td>20</td></tr>
      <tr><td>Interest expense</td><td>5</td><td>4</td></tr>
      <tr><td>Net income</td><td>18</td><td>12</td></tr>
      <tr><td>Adjusted EBITDAR</td><td>40</td><td>31</td></tr>
    </table>
    """
    title = "Reconciliation of Net Income to Adjusted EBITDAR"
    chunker = HTMLTableChunker(
        target_children=3,
        overlap_rows=1,
        repeated_context_rows=1,
    )

    children = chunker.split(source, title=title)

    assert len(children) == 3
    assert children[0].source_row_indices == (1, 2)
    assert children[1].source_row_indices == (2, 3, 4)
    assert children[2].source_row_indices == (4, 5, 6)
    assert children[0].overlap_row_indices == ()
    assert children[1].overlap_row_indices == (2,)
    assert children[2].overlap_row_indices == (4,)

    for child in children:
        assert child.html.startswith('<div class="table-title">')
        assert html.escape(title) in child.html
        assert child.html.count("<table") == 1
        assert child.html.count("</table>") == 1
        assert "Section:" not in child.html
        parsed_child = chunker.parse(child.html)
        assert parsed_child.rows[0] == ("Metric", "FY2022", "FY2021")
        assert parsed_child.width == 3


def test_rowspan_and_colspan_are_expanded_before_cross_boundary_split() -> None:
    source = """
    <table>
      <tr><td>Region</td><td colspan="2">Results</td></tr>
      <tr><td rowspan="2">Americas</td><td>Revenue</td><td>100</td></tr>
      <tr><td>EBIT</td><td>20</td></tr>
      <tr><td>EMEA</td><td>Revenue</td><td>80</td></tr>
    </table>
    """
    chunker = HTMLTableChunker(
        target_children=2,
        overlap_rows=1,
        repeated_context_rows=1,
    )

    parsed = chunker.parse(source)
    children = chunker.split(source, title="Regional results")

    assert parsed.rows == (
        ("Region", "Results", "Results"),
        ("Americas", "Revenue", "100"),
        ("Americas", "EBIT", "20"),
        ("EMEA", "Revenue", "80"),
    )
    assert len(children) == 2
    assert children[1].source_row_indices == (2, 3)
    second_rows = chunker.parse(children[1].html).rows
    assert second_rows == (
        ("Region", "Results", "Results"),
        ("Americas", "EBIT", "20"),
        ("EMEA", "Revenue", "80"),
    )
    assert "rowspan" not in children[1].html
    assert "colspan" not in children[1].html


def test_small_table_remains_one_child_without_duplicate_context_row() -> None:
    source = (
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    chunker = HTMLTableChunker(
        target_children=5,
        overlap_rows=2,
        repeated_context_rows=1,
    )

    children = chunker.split(source, title="Small table")

    assert len(children) == 1
    assert children[0].source_row_indices == (1,)
    assert chunker.parse(children[0].html).rows == (
        ("Metric", "Value"),
        ("Revenue", "100"),
    )


def test_plain_table_does_not_repeat_first_data_row_as_context() -> None:
    source = """
    <table>
      <tr><td>Metric</td><td>FY2023</td><td>FY2022</td></tr>
      <tr><td>Net sales</td><td>120</td><td>100</td></tr>
      <tr><td>Cost of sales</td><td>70</td><td>60</td></tr>
      <tr><td>Gross profit</td><td>50</td><td>40</td></tr>
      <tr><td>Operating income</td><td>25</td><td>20</td></tr>
      <tr><td>Net income</td><td>18</td><td>12</td></tr>
    </table>
    """
    chunker = HTMLTableChunker(
        target_children=3,
        overlap_rows=1,
        repeated_context_rows=2,
    )

    children = chunker.split(source, title="Income statement")

    assert len(children) == 3
    assert all(child.repeated_context_row_indices == (0,) for child in children)
    assert "Net sales" not in children[1].html
    assert "Net sales" not in children[2].html


def test_internal_group_header_uses_local_context_instead_of_first_header() -> None:
    source = """
    <table>
      <tr><td colspan="3">Three months: Consumer Banking</td></tr>
      <tr><td>Metric</td><td>2022</td><td>2021</td></tr>
      <tr><td>Revenue</td><td>120</td><td>100</td></tr>
      <tr><td>Income</td><td>30</td><td>25</td></tr>
      <tr><td colspan="3">Six months: Asset Management</td></tr>
      <tr><td>Metric</td><td>2022</td><td>2021</td></tr>
      <tr><td>Revenue</td><td>80</td><td>75</td></tr>
      <tr><td>Income</td><td>20</td><td>18</td></tr>
    </table>
    """
    chunker = HTMLTableChunker(
        target_children=2,
        overlap_rows=1,
        repeated_context_rows=2,
    )

    children = chunker.split(source, title="Segment results")

    assert len(children) == 2
    assert children[0].repeated_context_row_indices == (0, 1)
    assert children[0].source_row_indices == (2, 3)
    assert children[1].repeated_context_row_indices == (4, 5)
    assert children[1].source_row_indices == (6, 7)
    assert "Consumer Banking" not in children[1].html
    assert "Asset Management" in children[1].html


def test_caption_only_block_is_carried_into_following_local_header() -> None:
    source = """
    <table>
      <tr><td colspan="3">Segment performance</td></tr>
      <tr><td colspan="3">Three months ended June 30</td></tr>
      <tr><td>Metric</td><td>2022</td><td>2021</td></tr>
      <tr><td>Revenue</td><td>120</td><td>100</td></tr>
      <tr><td>Income</td><td>30</td><td>25</td></tr>
    </table>
    """
    chunker = HTMLTableChunker(
        target_children=1,
        overlap_rows=1,
        repeated_context_rows=2,
    )

    children = chunker.split(source, title="")

    assert len(children) == 1
    assert children[0].repeated_context_row_indices == (0, 1, 2)
    assert children[0].source_row_indices == (3, 4)
    assert chunker.parse(children[0].html).rows == chunker.parse(source).rows


def test_rejects_input_without_an_html_table() -> None:
    chunker = HTMLTableChunker()

    try:
        chunker.split("Revenue was 100", title="Not a table")
    except ValueError as exc:
        assert "HTML table" in str(exc)
    else:
        raise AssertionError("Expected input without a table to be rejected")


def test_extracts_markdown_heading_as_table_caption() -> None:
    source = (
        "#### Note 2 - Goodwill and Acquired Intangibles\n\n"
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Goodwill</td><td>100</td></tr></table>"
    )
    chunker = HTMLTableChunker()

    children = chunker.split(source)

    assert chunker.extract_caption(source) == (
        "Note 2 - Goodwill and Acquired Intangibles"
    )
    assert (
        '<div class="table-title">Note 2 - Goodwill and Acquired Intangibles</div>'
        in children[0].html
    )


def test_extracts_last_centered_div_before_table() -> None:
    source = (
        '<div style="text-align:center">Consolidated Statements of Income</div>\n'
        "<table><tr><td>Metric</td><td>2023</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    chunker = HTMLTableChunker()

    assert chunker.extract_caption(source) == "Consolidated Statements of Income"


def test_preferred_metadata_caption_wins_over_document_heading() -> None:
    source = (
        "### OCR heading\n"
        "<table><tr><td>Metric</td><td>Value</td></tr></table>"
    )
    chunker = HTMLTableChunker()

    assert (
        chunker.extract_caption(source, preferred_title="Canonical table title")
        == "Canonical table title"
    )


def test_short_colon_label_is_caption_but_long_prose_is_rejected() -> None:
    short = (
        "Provision for Income Taxes:\n"
        "<table><tr><td>Metric</td><td>Value</td></tr></table>"
    )
    long = (
        "Operating income declined during the year because several business units "
        "experienced inflation, restructuring charges, foreign exchange movements, "
        "and other market conditions that management discusses in detail below.\n"
        "<table><tr><td>Metric</td><td>Value</td></tr></table>"
    )
    chunker = HTMLTableChunker()

    assert chunker.extract_caption(short) == "Provision for Income Taxes:"
    assert chunker.extract_caption(long) == ""


def test_token_budget_greedily_packs_complete_rows_with_one_row_overlap() -> None:
    rows = "".join(
        "<tr><td>Metric "
        + str(index)
        + "</td><td>"
        + " ".join(f"value_{index}_{word}" for word in range(145))
        + "</td></tr>"
        for index in range(1, 7)
    )
    source = (
        "<table><tr><td>Metric</td><td>FY2024</td></tr>"
        f"{rows}</table>"
    )
    chunker = HTMLTableChunker(
        max_tokens=768,
        overlap_rows=1,
        repeated_context_rows=1,
        length_function=lambda value: len(value.split()),
    )

    children = chunker.split(
        source,
        title="Annual results",
        prefix_text="Section: Results\n\n" + " ".join(["before"] * 40),
        suffix_text=" ".join(["after"] * 40),
    )

    assert len(children) == 2
    assert children[0].source_row_indices == (1, 2, 3, 4)
    assert children[1].source_row_indices == (4, 5, 6)
    assert children[0].overlap_row_indices == ()
    assert children[1].overlap_row_indices == (4,)
    assert all(child.token_count <= 768 for child in children)
    assert all(child.index_text.startswith("Section: Results") for child in children)
    assert all(child.index_text.endswith("after " * 39 + "after") for child in children)


def test_token_budget_drops_overlap_when_it_would_exceed_limit() -> None:
    source = (
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Row 1</td><td>" + " ".join(["one"] * 330) + "</td></tr>"
        "<tr><td>Row 2</td><td>" + " ".join(["two"] * 330) + "</td></tr>"
        "</table>"
    )
    chunker = HTMLTableChunker(
        max_tokens=400,
        overlap_rows=1,
        repeated_context_rows=1,
        length_function=lambda value: len(value.split()),
    )

    children = chunker.split(source, prefix_text=" ".join(["context"] * 40))

    assert [child.source_row_indices for child in children] == [(1,), (2,)]
    assert children[1].overlap_row_indices == ()
    assert all(child.token_count <= 400 for child in children)


def test_single_oversized_row_is_preserved_as_one_child() -> None:
    source = (
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Very large disclosure</td><td>"
        + " ".join(["detail"] * 900)
        + "</td></tr></table>"
    )
    chunker = HTMLTableChunker(
        max_tokens=768,
        overlap_rows=1,
        repeated_context_rows=1,
        length_function=lambda value: len(value.split()),
    )

    children = chunker.split(source, title="Oversized row")

    assert len(children) == 1
    assert children[0].source_row_indices == (1,)
    assert children[0].token_count > 768
    assert "Very large disclosure" in children[0].html
