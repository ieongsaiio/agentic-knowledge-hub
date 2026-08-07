"""Markdown-table regression tests using FinanceBench extraction shapes."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.libs.splitter.recursive_splitter import RecursiveSplitter

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "markdown"


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _settings(
    chunk_size: int,
    chunk_overlap: int = 0,
    *,
    length_unit: str = "characters",
) -> Any:
    settings = MagicMock()
    settings.ingestion = MagicMock()
    settings.ingestion.chunk_size = chunk_size
    settings.ingestion.chunk_overlap = chunk_overlap
    settings.ingestion.length_unit = length_unit
    settings.ingestion.tokenizer_model = (
        "fixture-tokenizer" if length_unit == "tokens" else None
    )
    return settings


def _fixture(name: str) -> str:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8").strip()


def _assert_exact_offsets(
    source: str,
    chunks: list[tuple[str, int, int]],
) -> None:
    for chunk, start, end in chunks:
        assert source[start:end] == chunk


def test_default_separators_do_not_treat_every_period_as_a_sentence_boundary() -> None:
    splitter = RecursiveSplitter(_settings(chunk_size=800, chunk_overlap=150))

    assert ". " not in splitter.separators
    assert splitter.separators.index(".\n") < splitter.separators.index("\n")
    assert splitter.separators.index("\n## ") < splitter.separators.index("\n")


def test_default_separators_do_not_leave_us_abbreviation_period_in_next_chunk() -> None:
    source = (
        "Forward-looking statements include the impact of recent U.S. tax reform "
        "legislation on our results of operations and cash flows."
    )
    splitter = RecursiveSplitter(_settings(chunk_size=70, chunk_overlap=0))

    chunks = splitter.split_text_with_offsets(source)

    assert all(not chunk.startswith(". ") for chunk, _, _ in chunks)
    assert not any(
        previous.endswith("U.S") and current.startswith(". tax")
        for (previous, _, _), (current, _, _) in zip(chunks, chunks[1:])
    )
    _assert_exact_offsets(source, chunks)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "finance_table_3m_average_debt.txt",
        "finance_table_mgm_multiline_cells.txt",
    ],
)
def test_table_within_limit_remains_one_exact_chunk(fixture_name: str) -> None:
    table = _fixture(fixture_name)
    prefix = "Introductory prose. " * 40
    suffix = "Following explanatory prose. " * 40
    source = f"{prefix}\n\n{table}\n\n{suffix}"
    splitter = RecursiveSplitter(
        _settings(chunk_size=len(table) + 10, chunk_overlap=40)
    )

    chunks = splitter.split_text_with_offsets(source)

    assert any(chunk == table for chunk, _, _ in chunks)
    assert all(len(chunk) <= len(table) + 10 for chunk, _, _ in chunks)
    _assert_exact_offsets(source, chunks)


def test_oversized_real_table_splits_only_at_physical_line_boundaries() -> None:
    table = _fixture("finance_table_walmart_debt.txt")
    splitter = RecursiveSplitter(_settings(chunk_size=320, chunk_overlap=80))

    chunks = splitter.split_text_with_offsets(table)
    chunk_texts = [chunk for chunk, _, _ in chunks]

    assert len(chunks) > 1
    assert all(len(chunk) <= 320 for chunk in chunk_texts)
    _assert_exact_offsets(table, chunks)

    table_rows = [line for line in table.splitlines() if line.strip()]
    for row in table_rows:
        assert any(row in chunk for chunk in chunk_texts), row


def test_split_text_and_offset_api_return_the_same_table_chunks() -> None:
    table = _fixture("finance_table_3m_average_debt.txt")
    source = f"Before the table.\n\n{table}\n\nAfter the table."
    splitter = RecursiveSplitter(_settings(chunk_size=len(table) + 5))

    chunks = splitter.split_text(source)
    chunks_with_offsets = splitter.split_text_with_offsets(source)

    assert chunks == [chunk for chunk, _, _ in chunks_with_offsets]
    _assert_exact_offsets(source, chunks_with_offsets)


def test_token_mode_uses_token_limit_when_protecting_a_real_table() -> None:
    tokenizer = _WhitespaceTokenizer()
    table = _fixture("finance_table_3m_average_debt.txt")
    table_tokens = len(tokenizer.encode(table))
    source = f"{'intro ' * 100}\n\n{table}\n\n{'after ' * 100}"
    splitter = RecursiveSplitter(
        _settings(
            chunk_size=table_tokens + 5,
            chunk_overlap=10,
            length_unit="tokens",
        ),
        tokenizer=tokenizer,
    )

    chunks = splitter.split_text_with_offsets(source)

    assert any(chunk == table for chunk, _, _ in chunks)
    assert all(
        len(tokenizer.encode(chunk)) <= table_tokens + 5
        for chunk, _, _ in chunks
    )
    _assert_exact_offsets(source, chunks)


@pytest.mark.parametrize("page_marker", ["78", "Page 46", "iv", "\u00a0 9 \u00a0"])
def test_postprocessing_removes_standalone_page_number_chunks(
    page_marker: str,
) -> None:
    source = f"First paragraph.\n\n{page_marker}\n\nSecond paragraph."
    first_start = source.index("First paragraph.")
    page_start = source.index(page_marker)
    second_start = source.index("Second paragraph.")
    splitter = RecursiveSplitter(_settings(chunk_size=100))

    chunks = splitter._postprocess_chunks_with_offsets(
        source,
        [
            ("First paragraph.", first_start, first_start + 16),
            (page_marker, page_start, page_start + len(page_marker)),
            ("Second paragraph.", second_start, second_start + 17),
        ],
    )

    assert [chunk for chunk, _, _ in chunks] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    _assert_exact_offsets(source, chunks)


def test_four_digit_year_is_not_treated_as_a_page_number() -> None:
    splitter = RecursiveSplitter(_settings(chunk_size=100))

    assert splitter._is_page_number_only("2018") is False
    assert splitter._is_page_number_only("Page 2018") is False


@pytest.mark.parametrize(
    "heading",
    [
        "NOTE 5 - Accrued Liabilities",
        "Walmart U.S. Segment",
        "## Consolidated Statements of Cash Flows",
    ],
)
def test_short_heading_merges_forward_into_following_content(heading: str) -> None:
    body = "Accrued liabilities included the following amounts."
    source = f"{heading}\n\n{body}"
    body_start = source.index(body)
    splitter = RecursiveSplitter(_settings(chunk_size=len(source)))

    chunks = splitter._postprocess_chunks_with_offsets(
        source,
        [
            (heading, 0, len(heading)),
            (body, body_start, len(source)),
        ],
    )

    assert chunks == [(source, 0, len(source))]
    _assert_exact_offsets(source, chunks)


def test_short_heading_merges_forward_into_real_table() -> None:
    heading = "NOTE 5 - Accrued Liabilities"
    table = _fixture("finance_table_3m_average_debt.txt")
    source = f"{heading}\n\n{table}"
    table_start = source.index(table)
    splitter = RecursiveSplitter(_settings(chunk_size=len(source)))

    chunks = splitter._postprocess_chunks_with_offsets(
        source,
        [
            (heading, 0, len(heading)),
            (table, table_start, len(source)),
        ],
    )

    assert chunks == [(source, 0, len(source))]
    _assert_exact_offsets(source, chunks)


def test_short_heading_does_not_merge_when_combined_chunk_exceeds_limit() -> None:
    heading = "Walmart U.S. Segment"
    body = "Detailed segment information without punctuation"
    source = f"{heading}\n\n{body}"
    body_start = source.index(body)
    splitter = RecursiveSplitter(_settings(chunk_size=len(body)))

    chunks = splitter._postprocess_chunks_with_offsets(
        source,
        [
            (heading, 0, len(heading)),
            (body, body_start, len(source)),
        ],
    )

    assert [chunk for chunk, _, _ in chunks] == [heading, body]


def test_short_heading_does_not_merge_across_removed_page_number() -> None:
    heading = "Walmart U.S. Segment"
    page_marker = "78"
    body = "Detailed segment information follows."
    source = f"{heading}\n\n{page_marker}\n\n{body}"
    page_start = source.index(page_marker)
    body_start = source.index(body)
    splitter = RecursiveSplitter(_settings(chunk_size=len(source)))

    chunks = splitter._postprocess_chunks_with_offsets(
        source,
        [
            (heading, 0, len(heading)),
            (page_marker, page_start, page_start + len(page_marker)),
            (body, body_start, len(source)),
        ],
    )

    assert [chunk for chunk, _, _ in chunks] == [heading, body]
    _assert_exact_offsets(source, chunks)


def test_token_mode_limits_short_heading_merge() -> None:
    tokenizer = _WhitespaceTokenizer()
    heading = "Consolidated Balance Sheet"
    body = "one two three four five"
    source = f"{heading}\n\n{body}"
    body_start = source.index(body)
    splitter = RecursiveSplitter(
        _settings(chunk_size=5, length_unit="tokens"),
        tokenizer=tokenizer,
    )

    chunks = splitter._postprocess_chunks_with_offsets(
        source,
        [
            (heading, 0, len(heading)),
            (body, body_start, len(source)),
        ],
    )

    assert [chunk for chunk, _, _ in chunks] == [heading, body]


def test_public_split_api_filters_page_marker_and_merges_short_heading() -> None:
    heading = "## Consolidated Statements of Cash Flows"
    body = "Cash flows increased."
    source = f"{heading}\n\n{body}\n\nPage 46"
    splitter = RecursiveSplitter(_settings(chunk_size=65))

    chunks = splitter.split_text_with_offsets(source)

    assert all(chunk != "Page 46" for chunk, _, _ in chunks)
    assert any(heading in chunk and body in chunk for chunk, _, _ in chunks)
    _assert_exact_offsets(source, chunks)
