"""Tests for provider-neutral HTML table grid parsing."""

from src.libs.splitter.html_table_parser import HTMLTableParser


def test_expands_rowspan_and_colspan_into_rectangular_grid() -> None:
    table = (
        "<table>"
        '<tr><th rowspan="2">Metric</th><th colspan="2">Years</th></tr>'
        "<tr><th>2024</th><th>2023</th></tr>"
        "<tr><td>Revenue</td><td>100</td><td>90</td></tr>"
        "</table>"
    )

    parsed = HTMLTableParser().parse(table)

    assert parsed.width == 3
    assert parsed.rows == (
        ("Metric", "Years", "Years"),
        ("Metric", "2024", "2023"),
        ("Revenue", "100", "90"),
    )


def test_rejects_missing_table() -> None:
    try:
        HTMLTableParser().parse("<p>Not a table</p>")
    except ValueError as exc:
        assert "non-empty HTML table" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_visible_text_keeps_cells_but_drops_html_markup() -> None:
    table = (
        '<table class="financial"><tr class="table-section-caption">'
        '<th colspan="2">Revenue breakdown</th></tr>'
        '<tr><td style="text-align: center">Service fees</td>'
        '<td>$100</td></tr></table>'
    )

    visible = HTMLTableParser().visible_text(table)

    assert visible == (
        "Table row 1: Revenue breakdown | Revenue breakdown\n"
        "Table row 2: Service fees | $100"
    )
    assert "table-section-caption" not in visible
    assert "text-align" not in visible
    assert "<td" not in visible
