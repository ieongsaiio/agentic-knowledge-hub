"""Shared tokenization helpers for sparse indexing and query processing."""

from __future__ import annotations

import re
from typing import List

import jieba


_COMPOUND_TOKEN = re.compile(r"[^\W_]+(?:[._-][^\W_]+)+", re.UNICODE)
_PUNCTUATION_ONLY = re.compile(r"[\s\W]+", re.UNICODE)


def tokenize_mixed_text(text: str) -> List[str]:
    """Tokenize Chinese text while preserving model names and compound terms.

    Jieba segments ordinary text and Chinese runs. Terms such as ``GPT-4``,
    ``deep_learning`` and ``3.11`` are protected and emitted as one token so
    ingestion and query-time BM25 tokenization remain symmetrical.
    """

    raw_tokens: List[str] = []
    cursor = 0
    for match in _COMPOUND_TOKEN.finditer(text):
        if match.start() > cursor:
            raw_tokens.extend(jieba.lcut(text[cursor : match.start()]))
        raw_tokens.append(match.group(0))
        cursor = match.end()

    if cursor < len(text):
        raw_tokens.extend(jieba.lcut(text[cursor:]))

    tokens: List[str] = []
    for token in raw_tokens:
        cleaned = token.strip()
        if cleaned and not _PUNCTUATION_ONLY.fullmatch(cleaned):
            tokens.append(cleaned)
    return tokens
