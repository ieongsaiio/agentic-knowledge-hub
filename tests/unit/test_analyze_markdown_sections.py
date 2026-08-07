"""Tests for the Markdown parent-section analysis utility."""

from scripts.analyze_markdown_sections import (
    analyze_strategy,
    find_table_spans,
    split_by_heading_depth,
)


def test_heading_depth_controls_parent_section_granularity() -> None:
    source = (
        "# Report\n\nIntro.\n\n"
        "## Revenue\n\nRevenue body.\n\n"
        "### Region\n\nRegion body.\n\n"
        "## Debt\n\nDebt body."
    )

    h1_sections = split_by_heading_depth(source, 1)
    h2_sections = split_by_heading_depth(source, 2)
    h3_sections = split_by_heading_depth(source, 3)

    assert len(h1_sections) == 1
    assert len(h2_sections) == 3
    assert len(h3_sections) == 4
    assert h1_sections[0].text == source
    assert h2_sections[1].heading == "Revenue"
    assert "### Region" in h2_sections[1].text


def test_sections_retain_exact_source_offsets() -> None:
    source = "Preamble.\n\n# First\n\nBody.\n\n# Second\n\nOther."

    sections = split_by_heading_depth(source, 1)

    assert len(sections) == 3
    for section in sections:
        assert source[section.start_offset:section.end_offset] == section.text


def test_html_table_remains_inside_one_heading_section() -> None:
    table = "<table><tr><td>Revenue</td><td>10</td></tr></table>"
    source = f"# Results\n\nBefore.\n\n{table}\n\nAfter.\n\n# Notes\n\nText."

    stats, sections = analyze_strategy(source, 1)

    assert stats.table_count == 1
    assert stats.split_table_count == 0
    assert stats.table_integrity_rate == 1.0
    assert any(table in section.text for section in sections)


def test_pipe_table_is_detected_as_one_span() -> None:
    table = "| Item | 2023 |\n| --- | --- |\n| Revenue | 10 |"
    source = f"# Results\n\n{table}\n\nText."

    spans = find_table_spans(source)

    assert len(spans) == 1
    start, end = spans[0]
    assert source[start:end].strip() == table


def test_preamble_is_preserved_without_a_heading() -> None:
    source = "Introductory text.\n\n## Details\n\nBody."

    sections = split_by_heading_depth(source, 2)

    assert [section.heading for section in sections] == [None, "Details"]
    assert sections[0].text == "Introductory text."


def test_empty_markdown_returns_no_sections() -> None:
    assert split_by_heading_depth("  \n", 2) == []
