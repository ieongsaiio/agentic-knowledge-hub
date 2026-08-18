"""Tests for table-summary latency benchmark helpers."""

import json

import pytest

from scripts.benchmark_table_summary_latency import (
    _find_table,
    _percentile,
    _stats,
    _usage,
)


class _Response:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 80},
    }
    raw_response = None


def test_usage_extracts_cached_and_uncached_tokens() -> None:
    usage = _usage(_Response())

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["cached_input_tokens"] == 80
    assert usage["uncached_input_tokens"] == 20


def test_percentiles_use_linear_interpolation() -> None:
    values = [float(value) for value in range(1, 11)]

    assert _percentile(values, 0.50) == 5.5
    assert _percentile(values, 0.95) == pytest.approx(9.55)
    assert _percentile(values, 0.99) == pytest.approx(9.91)


def test_stats_preserve_missing_usage_signal() -> None:
    assert _stats([None, None])["count"] == 0
    assert _stats([1, 2, 3])["avg"] == 2


def test_find_table_selects_exact_table_group(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    records = [
        {"chunk_id": "text-1", "chunk_type": "text"},
        {"chunk_id": "table-1", "chunk_type": "table_group"},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    assert _find_table(path, "table-1") == records[1]
