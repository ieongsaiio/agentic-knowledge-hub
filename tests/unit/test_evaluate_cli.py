"""Focused tests for benchmark evaluation CLI modes."""

from __future__ import annotations

import pytest

from scripts.evaluate import parse_args


def test_parse_args_accepts_index_only_mode() -> None:
    args = parse_args(["--index-only", "--force-reindex", "--json"])

    assert args.index_only is True
    assert args.force_reindex is True
    assert args.json is True


def test_parse_args_rejects_index_only_without_search() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--index-only", "--no-search"])
