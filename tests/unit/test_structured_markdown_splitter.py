"""TDD contract tests for Section-aware semantic-unit chunking."""

from types import SimpleNamespace

from src.core.types import Document
from src.libs.loader.markdown_section_tree import build_markdown_section_tree
from src.libs.splitter.structured_markdown_splitter import (
    StructuredMarkdownSplitter,
)
from src.libs.splitter.table_linearizer import TableLinearizer


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()

    def decode(self, tokens: list[str], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return " ".join(tokens)


class _RecordingSummarizer:
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls: list[str] = []

    def summarize(self, table_text: str, **_: object) -> str:
        self.calls.append(table_text)
        return self.summary


def _settings(
    *,
    chunk_size: int = 100,
    chunk_overlap: int = 0,
    embedding_max_tokens: int | None = 200,
    model_max_tokens: int = 32768,
    table_dense_representation: str = "linearized",
    table_context_tokens: int = 80,
    table_child_enabled: bool = False,
    table_child_max_tokens: int = 768,
    table_summary_enabled: bool = False,
) -> SimpleNamespace:
    structured_chunking = {
        "text_splitter": "recursive",
        "embedding_safety_margin": 0,
        "table_dense_representation": table_dense_representation,
        "table_context_tokens": table_context_tokens,
        "table_summary": {"enabled": table_summary_enabled},
        "table_child_chunking": {
            "enabled": table_child_enabled,
            "max_tokens": table_child_max_tokens,
            "overlap_rows": 1,
            "repeated_context_rows": 1,
        },
    }
    if embedding_max_tokens is not None:
        structured_chunking["embedding_max_tokens"] = embedding_max_tokens
    ingestion = SimpleNamespace(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_unit="tokens",
        tokenizer_model="fixture-tokenizer",
        structured_chunking=structured_chunking,
    )
    return SimpleNamespace(
        ingestion=ingestion,
        embedding=SimpleNamespace(max_tokens=model_max_tokens),
    )


def test_table_claims_caption_footnote_and_copies_bounded_text_context() -> None:
    before_words = [f"before_{index}" for index in range(80)]
    after_words = [f"after_{index}" for index in range(80)]
    caption = '<div style="text-align: center;">Debt Maturities</div>'
    table = (
        "<table><tr><td>Year</td><td>Amount</td></tr>"
        "<tr><td>2025</td><td>100</td></tr></table>"
    )
    footnote = "(1) Amounts are reported in millions."
    text = "\n\n".join(
        [
            "# Liquidity",
            " ".join(before_words),
            caption,
            table,
            footnote,
            " ".join(after_words),
        ]
    )
    blocks = [
        {"block_label": "paragraph", "block_content": " ".join(before_words)},
        {"block_label": "table_title", "block_content": "Debt Maturities"},
        {"block_label": "table", "block_content": table},
        {"block_label": "vision_footnote", "block_content": footnote},
        {"block_label": "paragraph", "block_content": " ".join(after_words)},
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=512, table_context_tokens=64),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text, parsing_blocks=blocks))
    table_fragment = next(
        fragment for fragment in fragments if fragment.metadata["unit_types"] == ["table"]
    )
    ordinary_text = "\n".join(
        fragment.text
        for fragment in fragments
        if fragment is not table_fragment
    )

    assert table_fragment.metadata["previous_context"].split() == before_words[-16:]
    assert table_fragment.metadata["next_context"].split() == after_words[:64]
    assert table_fragment.text.startswith(before_words[-16])
    assert table_fragment.text.endswith(after_words[63])
    assert caption in table_fragment.text
    assert table in table_fragment.text
    assert footnote in table_fragment.text
    assert caption not in ordinary_text
    assert footnote not in ordinary_text
    assert before_words[-1] in ordinary_text
    assert after_words[0] in ordinary_text
    assert "Previous context:" not in table_fragment.dense_index_text
    assert "Next context:" not in table_fragment.dense_index_text


def test_recursive_text_boundaries_pack_only_the_tail_with_an_atomic_list() -> None:
    text_a = " ".join(f"a{index}" for index in range(600))
    source_list = "\n".join(f"- l{index}" for index in range(150))
    text_b = " ".join(f"b{index}" for index in range(700))
    source = f"# Results\n\n{text_a}\n\n{source_list}\n\n{text_b}"
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=512, chunk_overlap=50),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(source))

    assert [fragment.metadata["unit_types"] for fragment in fragments] == [
        ["text"],
        ["text", "list"],
        ["text"],
        ["text"],
    ]
    assert all(splitter._length(fragment.dense_index_text) <= 514 for fragment in fragments)
    assert source_list in fragments[1].text
    assert source_list not in fragments[0].text
    assert source_list not in fragments[2].text
    assert source_list not in fragments[3].text
    assert "a500" in fragments[0].text
    assert "a500" in fragments[1].text
    assert "b500" in fragments[2].text
    assert "b500" in fragments[3].text


def test_table_is_a_packing_barrier_even_when_neighbors_would_fit() -> None:
    before = "Before table context."
    table = "<table><tr><td>Revenue</td><td>10</td></tr></table>"
    after = "After table context."
    source = f"# Results\n\n{before}\n\n{table}\n\n{after}"
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=100, table_context_tokens=2),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(source))

    assert [fragment.metadata["unit_types"] for fragment in fragments] == [
        ["text"],
        ["table"],
        ["text"],
    ]
    assert fragments[1].metadata["previous_context"] == "context."
    assert fragments[1].metadata["next_context"] == "After table"


def test_section_path_is_counted_in_recursive_text_and_packing_budget() -> None:
    body = " ".join(f"token_{index}" for index in range(700))
    source = "# Annual Report\n\n## Detailed Financial Results\n\n" + body
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=100, chunk_overlap=10),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(source))

    assert len(fragments) > 1
    assert all(
        splitter._length(fragment.dense_index_text) <= 100
        for fragment in fragments
    )


def test_original_table_dense_representation_changes_only_table_body() -> None:
    caption = '<div style="text-align: center;">Financial Results</div>'
    table = (
        "<table><tr><td>Item</td><td>2023</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    footnote = "Amounts are in millions."
    text = f"# Results\n\n{caption}\n\n{table}\n\n{footnote}"
    blocks = [
        {"block_label": "table_title", "block_content": "Financial Results"},
        {"block_label": "table", "block_content": table},
        {"block_label": "vision_footnote", "block_content": footnote},
    ]
    document = _document(text, parsing_blocks=blocks)

    linearized = StructuredMarkdownSplitter(
        _settings(table_dense_representation="linearized"),
        tokenizer=_WhitespaceTokenizer(),
    ).split_document(document)[0]
    original = StructuredMarkdownSplitter(
        _settings(table_dense_representation="original"),
        tokenizer=_WhitespaceTokenizer(),
    ).split_document(document)[0]

    assert original.text == linearized.text
    assert original.start_offset == linearized.start_offset
    assert original.end_offset == linearized.end_offset
    assert original.sparse_index_text == linearized.sparse_index_text
    assert "Section: Results" in original.dense_index_text
    assert "Table title: Financial Results" in original.dense_index_text
    assert f"Footnote: {footnote}" in original.dense_index_text
    assert table in original.dense_index_text
    assert "Table row 1:" not in original.dense_index_text
    assert table not in linearized.dense_index_text
    assert "Table row 1:" in linearized.dense_index_text
    assert original.metadata["embedding_source_type"] == "original_table"
    assert linearized.metadata["embedding_source_type"] == "linearized_table"


def _document(
    text: str,
    *,
    parsing_blocks: list[dict] | None = None,
) -> Document:
    metadata = {
        "source_path": "report.pdf",
        "page_spans": [
            {
                "page": 1,
                "start_offset": 0,
                "end_offset": len(text),
            }
        ],
    }
    if parsing_blocks is not None:
        metadata["parsed_artifact"] = {
            "pages": [
                {
                    "res": {
                        "page_index": 0,
                        "prunedResult": {
                            "parsing_res_list": parsing_blocks,
                        },
                    }
                }
            ],
            "restructured_pages": [],
        }
    document = Document(id="doc_report", text=text, metadata=metadata)
    document.metadata["section_tree"] = build_markdown_section_tree(
        text,
        document_id=document.id,
        page_spans=metadata["page_spans"],
    )
    return document


def test_html_table_linearizer_preserves_physical_rows_and_columns() -> None:
    table = (
        "<table><tr><th>Item</th><th>2023</th><th>2022</th></tr>"
        "<tr><td>Revenue</td><td>100</td><td>80</td></tr></table>"
    )

    result = TableLinearizer().linearize(table)

    assert result == (
        'Table columns: 3\n'
        'Table row 1: column 1="Item"; column 2="2023"; column 3="2022"\n'
        'Table row 2: column 1="Revenue"; column 2="100"; column 3="80"'
    )


def test_pipe_table_linearizer_preserves_physical_rows_and_columns() -> None:
    table = (
        "| Item | 2023 | 2022 |\n"
        "|---|---:|---:|\n"
        "| Revenue | 100 | 80 |"
    )

    result = TableLinearizer().linearize(table)

    assert result == (
        'Table columns: 3\n'
        'Table row 1: column 1="Item"; column 2="2023"; column 3="2022"\n'
        'Table row 2: column 1="Revenue"; column 2="100"; column 3="80"'
    )


def test_complex_table_preserves_rowspan_and_colspan_without_false_keys() -> None:
    table = (
        '<table><tr><td colspan="4">Twelve Months Ended June 30,</td>'
        '<td rowspan="2">Comparable\nconstant\ncurrency $ \\Delta\\% $</td></tr>'
        '<tr><td>Adjusted non-GAAP results</td><td>2022 million</td>'
        '<td>2023 million</td><td>Reported $ \\Delta\\% $</td></tr>'
        '<tr><td>Net sales</td><td>14,544</td><td>14,694</td>'
        '<td>1</td><td>—</td></tr></table>'
    )

    result = TableLinearizer().linearize(table)

    assert "Table columns: 5" in result
    assert (
        'Table row 1: columns 1-4="Twelve Months Ended June 30,"; '
        'column 5, rows 1-2="Comparable constant currency $ \\Delta\\% $"'
        in result
    )
    assert (
        'Table row 2: column 1="Adjusted non-GAAP results"; '
        'column 2="2022 million"; column 3="2023 million"; '
        'column 4="Reported $ \\Delta\\% $"'
        in result
    )
    assert (
        'Table row 3: column 1="Net sales"; column 2="14,544"; '
        'column 3="14,694"; column 4="1"; column 5="—"'
        in result
    )
    assert "Twelve Months Ended June 30,=Net sales" not in result


def test_grouped_year_headers_keep_nine_physical_columns() -> None:
    table = (
        '<table><tr><td></td><td colspan="4">Twelve Months Ended June 30, 2022</td>'
        '<td colspan="4">Twelve Months Ended June 30, 2023</td></tr>'
        '<tr><td>Adjusted non-GAAP results</td><td>Net sales million</td>'
        '<td>EBIT million</td><td>EBIT / Sales %</td>'
        '<td>EBIT / Average funds employed %</td><td>Net sales million</td>'
        '<td>EBIT million</td><td>EBIT / Sales %</td>'
        '<td>EBIT / Average funds employed %</td></tr>'
        '<tr><td>Flexibles</td><td>11,151</td><td>1,517</td><td>13.6</td>'
        '<td></td><td>11,154</td><td>1,429</td><td>12.8</td><td></td></tr></table>'
    )

    result = TableLinearizer().linearize(table)

    assert "Table columns: 9" in result
    assert 'columns 2-5="Twelve Months Ended June 30, 2022"' in result
    assert 'columns 6-9="Twelve Months Ended June 30, 2023"' in result
    assert (
        'Table row 3: column 1="Flexibles"; column 2="11,151"; '
        'column 3="1,517"; column 4="13.6"; column 5=""; '
        'column 6="11,154"; column 7="1,429"; column 8="12.8"; '
        'column 9=""'
        in result
    )


def test_table_linearizer_normalizes_literal_ocr_newline_escapes() -> None:
    table = (
        '<table><tr><td>Metric</td><td>Comparable\\nconstant\\ncurrency</td></tr>'
        '<tr><td>Revenue</td><td>1</td></tr></table>'
    )

    result = TableLinearizer().linearize(table)

    assert "Comparable constant currency" in result
    assert "\\n" not in result


def test_html_body_rowspan_keeps_following_values_in_correct_columns() -> None:
    table = (
        '<table><tr><td>Region</td><td>Metric</td><td>2023</td></tr>'
        '<tr><td rowspan="2">Americas</td><td>Revenue</td><td>100</td></tr>'
        '<tr><td>EBIT</td><td>20</td></tr></table>'
    )

    result = TableLinearizer().linearize(table)

    assert (
        'Table row 2: column 1, rows 2-3="Americas"; '
        'column 2="Revenue"; column 3="100"'
        in result
    )
    assert 'Table row 3: column 2="EBIT"; column 3="20"' in result


def test_ragged_pipe_table_keeps_empty_physical_columns() -> None:
    table = (
        "| Item | 2023 | 2022 |\n"
        "|---|---:|---:|\n"
        "| Revenue | 100 |\n"
        "| EBIT | | 20 |"
    )

    result = TableLinearizer().linearize(table)

    assert 'Table row 2: column 1="Revenue"; column 2="100"; column 3=""' in result
    assert 'Table row 3: column 1="EBIT"; column 2=""; column 3="20"' in result


def test_table_unit_uses_paddle_title_and_footnote_for_dense_text() -> None:
    table = (
        "<table><tr><td>Item</td><td>2023</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    text = f"# Results\n\n{table}"
    blocks = [
        {
            "block_label": "table_title",
            "block_content": "Financial Results",
            "block_bbox": [10, 10, 300, 30],
            "block_order": 1,
        },
        {
            "block_label": "table",
            "block_content": table,
            "block_bbox": [10, 40, 300, 160],
            "block_order": 2,
        },
        {
            "block_label": "vision_footnote",
            "block_content": "Amounts are in millions.",
            "block_bbox": [10, 170, 300, 190],
            "block_order": 3,
        },
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text, parsing_blocks=blocks))
    table_fragment = next(
        fragment
        for fragment in fragments
        if "table" in fragment.metadata["unit_types"]
    )

    assert table in table_fragment.text
    assert table_fragment.metadata["table_title"] == "Financial Results"
    assert table_fragment.metadata["vision_footnotes"] == [
        "Amounts are in millions."
    ]
    assert "Section: Results" in table_fragment.dense_index_text
    assert "Table title: Financial Results" in table_fragment.dense_index_text
    assert "Footnote: Amounts are in millions." in table_fragment.dense_index_text
    assert (
        'Table row 2: column 1="Revenue"; column 2="100"'
        in table_fragment.dense_index_text
    )
    assert table in table_fragment.sparse_index_text
    assert "Financial Results" in table_fragment.sparse_index_text
    assert "Amounts are in millions." in table_fragment.sparse_index_text


def test_table_unit_claims_adjacent_caption_and_footnote_before_text_chunking() -> None:
    caption = '<div style="text-align: center;">Financial Results</div>'
    table = (
        "<table><tr><td>Item</td><td>2023</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    footnote = "(1) Amounts are reported in millions."
    before = "Introductory analysis remains ordinary text."
    after = "This explanation after the footnote also remains ordinary text."
    text = (
        f"# Results\n\n{before}\n\n{caption}\n\n{table}\n\n"
        f"{footnote}\n\n{after}"
    )
    blocks = [
        {"block_label": "paragraph", "block_content": before},
        {"block_label": "table_title", "block_content": "Financial Results"},
        {"block_label": "table", "block_content": table},
        {"block_label": "vision_footnote", "block_content": footnote},
        {"block_label": "paragraph", "block_content": after},
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=12),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text, parsing_blocks=blocks))
    table_fragment = next(
        fragment
        for fragment in fragments
        if fragment.metadata["unit_types"] == ["table"]
    )
    other_text = "\n".join(
        fragment.text
        for fragment in fragments
        if fragment is not table_fragment
    )

    assert f"{caption}\n\n{table}\n\n{footnote}" in table_fragment.text
    assert table_fragment.metadata["previous_context"] == (
        f"# Results\n\n{before}"
    )
    assert table_fragment.metadata["next_context"] == after
    assert table_fragment.metadata["table_title"] == "Financial Results"
    assert table_fragment.metadata["vision_footnotes"] == [footnote]
    assert caption not in other_text
    assert footnote not in other_text
    assert before in other_text
    assert after in other_text


def test_real_paddle_html_table_shape_claims_centered_caption() -> None:
    caption = '<div style="text-align: center;">As of May 31, 2017</div>'
    table = (
        "<table border=1 style='margin: auto; word-wrap: break-word;'>"
        "<tr><td style='text-align: center;'>Assets at Fair Value</td>"
        "<td style='text-align: center;'>Cash Equivalents</td></tr>"
        "<tr><td style='text-align: center;'>Level 1</td>"
        "<td style='text-align: center;'>$ 2,371</td></tr></table>"
    )
    footnote = "(1) The amounts above are measured on a recurring basis."
    text = f"# Fair Value Measurements\n\n{caption}\n\n{table}\n\n{footnote}"
    blocks = [
        {"block_label": "table_title", "block_content": "As of May 31, 2017"},
        {"block_label": "table", "block_content": table},
        {"block_label": "vision_footnote", "block_content": footnote},
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=8),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text, parsing_blocks=blocks))

    assert len(fragments) == 1
    assert fragments[0].metadata["unit_types"] == ["table"]
    assert fragments[0].text == (
        f"# Fair Value Measurements\n\n{caption}\n\n{table}\n\n{footnote}"
    )
    assert fragments[0].metadata["previous_context"] == "# Fair Value Measurements"
    assert "Table title: As of May 31, 2017" in fragments[0].dense_index_text
    assert f"Footnote: {footnote}" in fragments[0].dense_index_text


def test_context_and_complete_table_pack_when_they_fit() -> None:
    table = "<table><tr><td>Year</td><td>Amount</td></tr><tr><td>2025</td><td>10</td></tr></table>"
    text = f"# Debt\n\nThe following table presents maturities:\n\n{table}"
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=100),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text))

    assert len(fragments) == 2
    assert fragments[0].metadata["unit_types"] == ["text"]
    assert fragments[1].metadata["unit_types"] == ["table"]
    assert fragments[0].text == "# Debt\n\nThe following table presents maturities:"
    assert fragments[1].metadata["previous_context"] == fragments[0].text
    assert fragments[1].text == text


def test_context_and_table_are_separate_when_they_do_not_fit() -> None:
    table = "<table><tr><td>Year</td><td>Amount</td></tr><tr><td>2025</td><td>10</td></tr></table>"
    text = (
        "# Debt\n\n"
        "This context contains several words and cannot fit beside the table.\n\n"
        f"{table}"
    )
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=7),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text))

    assert len(fragments) >= 2
    table_fragment = next(
        fragment
        for fragment in fragments
        if fragment.metadata["unit_types"] == ["table"]
    )
    assert table_fragment.text.endswith(table)
    assert table_fragment.metadata["previous_context"] == (
        "# Debt\n\nThis context contains several words and cannot fit beside "
        "the table."
    )


def test_list_and_code_blocks_remain_complete_even_above_target() -> None:
    source_list = "* first item has several words\n* second item has several words"
    source_code = "```python\nprint('one')\nprint('two')\n```"
    text = f"# Examples\n\n{source_list}\n\n{source_code}"
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=3),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text))

    list_fragment = next(
        fragment
        for fragment in fragments
        if fragment.metadata["unit_types"] == ["list"]
    )
    code_fragment = next(
        fragment
        for fragment in fragments
        if fragment.metadata["unit_types"] == ["code"]
    )
    assert list_fragment.text == source_list
    assert code_fragment.text == source_code


def test_table_summary_adds_dense_supplement_without_replacing_original() -> None:
    table = (
        "<table><tr><td>Item</td><td>Year</td></tr>"
        + "".join(
            f"<tr><td>Revenue {index}</td><td>{2020 + index}</td></tr>"
            for index in range(10)
        )
        + "</table>"
    )
    summarizer = _RecordingSummarizer(
        "The table reports revenue across multiple fiscal years."
    )
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=100, embedding_max_tokens=12),
        tokenizer=_WhitespaceTokenizer(),
        table_summarizer=summarizer,
    )

    splitter._table_summary_enabled = True

    fragments = splitter.split_document(_document(f"# Results\n\n{table}"))
    table_fragment = next(
        fragment
        for fragment in fragments
        if "table" in fragment.metadata["unit_types"]
        and fragment.metadata.get("chunk_role") != "table_summary"
    )
    summary_fragment = next(
        fragment
        for fragment in fragments
        if fragment.metadata.get("chunk_role") == "table_summary"
    )

    assert table in table_fragment.text
    assert "Revenue 9" in table_fragment.dense_index_text
    assert summarizer.calls == [table]
    assert summary_fragment.text == table_fragment.text
    assert summary_fragment.dense_index_text == (
        "The table reports revenue across multiple fiscal years."
    )
    assert summary_fragment.sparse_index_text == table_fragment.text
    assert summary_fragment.metadata["embedding_source_type"] == (
        "llm_table_summary_supplement"
    )
    assert summary_fragment.metadata["sparse_index_enabled"] is False
    assert summary_fragment.metadata["storage_id"].startswith(
        summary_fragment.metadata["parent_chunk_id"] + "_summary_"
    )


def test_table_budget_defaults_to_embedding_model_max_tokens() -> None:
    splitter = StructuredMarkdownSplitter(
        _settings(
            embedding_max_tokens=None,
            model_max_tokens=32768,
        ),
        tokenizer=_WhitespaceTokenizer(),
    )

    assert splitter.embedding_max_tokens == 32768


def test_text_unit_dense_text_contains_header_path_once() -> None:
    text = "# Report\n\n## Results\n\nRevenue increased by 12 percent."
    splitter = StructuredMarkdownSplitter(
        _settings(),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text))
    result_fragment = next(
        fragment
        for fragment in fragments
        if fragment.metadata["header_path"] == ["Report", "Results"]
    )

    assert result_fragment.text.startswith("## Results")
    assert "Section: Report > Results" in result_fragment.dense_index_text
    assert result_fragment.dense_index_text.count("Results") == 1


def test_two_tables_on_one_page_do_not_share_the_first_tables_footnote() -> None:
    first = (
        "<table><tr><td>Revenue</td><td>100</td></tr></table>"
    )
    second = (
        "<table><tr><td>Net debt</td><td>42</td></tr></table>"
    )
    styled_first = first.replace("<table>", "<table border=1>")
    styled_second = second.replace("<table>", "<table border=1>")
    text = f"# Results\n\n{styled_first}\n\n## Debt\n\n{styled_second}"
    blocks = [
        {"block_label": "table", "block_content": first},
        {"block_label": "vision_footnote", "block_content": "Revenue footnote."},
        {"block_label": "paragraph_title", "block_content": "Debt"},
        {"block_label": "table", "block_content": second},
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text, parsing_blocks=blocks))
    tables = [
        fragment
        for fragment in fragments
        if "table" in fragment.metadata["unit_types"]
    ]

    assert tables[0].metadata["vision_footnotes"] == ["Revenue footnote."]
    assert "vision_footnotes" not in tables[1].metadata


def test_heading_only_fragment_before_oversized_table_is_removed() -> None:
    table = (
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    text = f"# Financial Results\n\n{table}"
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=3),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text))

    assert len(fragments) == 1
    assert fragments[0].metadata["unit_types"] == ["table"]
    assert fragments[0].text == f"# Financial Results\n\n{table}"
    assert fragments[0].metadata["previous_context"] == "# Financial Results"
    assert "Section: Financial Results" in fragments[0].dense_index_text


def test_footnote_text_is_claimed_by_the_table_unit_before_chunking() -> None:
    table = (
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    footnote = "(1) Amounts are reported in millions."
    text = f"# Results\n\n{table}\n\n{footnote}"
    blocks = [
        {"block_label": "table", "block_content": table},
        {"block_label": "vision_footnote", "block_content": footnote},
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(chunk_size=100),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text, parsing_blocks=blocks))

    assert len(fragments) == 1
    assert "table" in fragments[0].metadata["unit_types"]
    assert fragments[0].text.endswith(f"{table}\n\n{footnote}")
    assert fragments[0].metadata["vision_footnotes"] == [footnote]
    assert f"Footnote: {footnote}" in fragments[0].dense_index_text
    assert footnote in fragments[0].sparse_index_text


def test_unattached_footnote_like_text_is_retained() -> None:
    text = "# Notes\n\n(1) This note is not attached to a parsed table."
    splitter = StructuredMarkdownSplitter(
        _settings(),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(text))

    assert len(fragments) == 1
    assert "This note is not attached" in fragments[0].text


def test_table_child_mode_replaces_parent_with_budgeted_retrieval_children() -> None:
    rows = "".join(
        "<tr><td>Metric "
        + str(index)
        + "</td><td>"
        + " ".join(f"value_{index}_{word}" for word in range(145))
        + "</td></tr>"
        for index in range(1, 7)
    )
    table = (
        "<table><tr><td>Metric</td><td>FY2024</td></tr>"
        f"{rows}</table>"
    )
    text = f"# Results\n\nIntroductory table context.\n\n{table}\n\nFollowing context."
    splitter = StructuredMarkdownSplitter(
        _settings(
            chunk_size=512,
            table_child_enabled=True,
            table_child_max_tokens=768,
        ),
        tokenizer=_WhitespaceTokenizer(),
    )

    first = splitter.split_document(_document(text))
    second = splitter.split_document(_document(text))
    children = [
        fragment
        for fragment in first
        if fragment.metadata.get("chunk_role") == "table_child"
    ]

    assert len(children) == 2
    assert all(fragment.text != fragment.dense_index_text for fragment in children)
    assert all(fragment.text == fragment.sparse_index_text for fragment in children)
    assert all(fragment.text.startswith("Section: Results") for fragment in children)
    assert all("Section: Results" not in fragment.dense_index_text for fragment in children)
    assert all("Introductory table context." in fragment.dense_index_text for fragment in children)
    assert all("<table>" in fragment.dense_index_text for fragment in children)
    assert all(fragment.metadata["table_child_count"] == 2 for fragment in children)
    assert [fragment.metadata["table_child_index"] for fragment in children] == [0, 1]
    first_rows = children[0].metadata["source_row_indices"]
    second_rows = children[1].metadata["source_row_indices"]
    assert set(first_rows + second_rows) == set(range(1, 7))
    assert children[1].metadata["overlap_row_indices"] == [first_rows[-1]]
    assert second_rows[0] == first_rows[-1]
    assert children[0].metadata["parent_chunk_id"] == children[1].metadata["parent_chunk_id"]
    assert children[0].metadata["preserve_raw_content"] is True
    assert children[0].metadata["source_exact"] is False
    assert children[0].start_offset == children[1].start_offset
    assert children[0].end_offset == children[1].end_offset
    assert all(splitter._length(fragment.text) <= 768 for fragment in children)
    second_children = [
        fragment
        for fragment in second
        if fragment.metadata.get("chunk_role") == "table_child"
    ]
    assert [fragment.metadata["storage_id"] for fragment in children] == [
        fragment.metadata["storage_id"] for fragment in second_children
    ]


def test_table_summary_supplement_points_to_same_parent_as_table_children() -> None:
    table = (
        "<table><tr><td>Metric</td><td>FY2024</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr>"
        "<tr><td>Income</td><td>20</td></tr></table>"
    )
    summarizer = _RecordingSummarizer(
        "FY2024 revenue is 100 and income is 20."
    )
    splitter = StructuredMarkdownSplitter(
        _settings(
            table_child_enabled=True,
            table_summary_enabled=True,
        ),
        tokenizer=_WhitespaceTokenizer(),
        table_summarizer=summarizer,
    )

    fragments = splitter.split_document(
        _document(f"# Results\n\nContext before.\n\n{table}\n\nContext after.")
    )
    children = [
        fragment
        for fragment in fragments
        if fragment.metadata.get("chunk_role") == "table_child"
    ]
    summary = next(
        fragment
        for fragment in fragments
        if fragment.metadata.get("chunk_role") == "table_summary"
    )

    assert len(children) == 1
    assert summarizer.calls == [table]
    assert summary.metadata["parent_chunk_id"] == children[0].metadata["parent_chunk_id"]
    assert "Context before." in summary.text
    assert table in summary.text
    assert "Context after." in summary.text
    assert summary.dense_index_text == "FY2024 revenue is 100 and income is 20."
    assert "Section: Results" not in summary.dense_index_text


def test_table_child_dense_text_strips_only_leading_section_path() -> None:
    table = (
        "<table><tr><td>Metric</td><td>FY2024</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    footnote = "(1) Amounts are in millions."
    text = f"# Results\n\nUseful preceding context.\n\n{table}\n\n{footnote}"
    blocks = [
        {"block_label": "paragraph", "block_content": "Useful preceding context."},
        {"block_label": "table_title", "block_content": "Annual Results"},
        {"block_label": "table", "block_content": table},
        {"block_label": "vision_footnote", "block_content": footnote},
    ]
    splitter = StructuredMarkdownSplitter(
        _settings(table_child_enabled=True),
        tokenizer=_WhitespaceTokenizer(),
    )

    child = next(
        fragment
        for fragment in splitter.split_document(_document(text, parsing_blocks=blocks))
        if fragment.metadata.get("chunk_role") == "table_child"
    )

    assert child.text.startswith("Section: Results")
    assert not child.dense_index_text.startswith("Section:")
    assert "Useful preceding context." in child.dense_index_text
    assert "Annual Results" in child.dense_index_text
    assert "<table>" in child.dense_index_text
    assert "<td>Revenue</td><td>100</td>" in child.dense_index_text
    assert f"Footnote: {footnote}" in child.dense_index_text
    assert child.metadata["header_path"] == ["Results"]
    assert child.metadata["embedding_source_type"] == "original_table_child_no_section"


def test_disabled_table_child_mode_keeps_source_exact_parent_table() -> None:
    table = (
        "<table><tr><td>Metric</td><td>Value</td></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>"
    )
    splitter = StructuredMarkdownSplitter(
        _settings(table_child_enabled=False),
        tokenizer=_WhitespaceTokenizer(),
    )

    fragments = splitter.split_document(_document(f"# Results\n\n{table}"))

    assert len(fragments) == 1
    assert fragments[0].text.endswith(table)
    assert "chunk_role" not in fragments[0].metadata
