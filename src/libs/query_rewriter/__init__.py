"""Pluggable query-understanding and rewriting components."""

from src.libs.query_rewriter.base_query_rewriter import (
    BaseQueryRewriter,
    QueryRewritePlan,
    QuerySlot,
    RewrittenQuery,
    SourceHint,
)
from src.libs.query_rewriter.llm_query_rewriter import LLMQueryRewriter
from src.libs.query_rewriter.query_rewriter_factory import QueryRewriterFactory

QueryRewriterFactory.register_provider("llm", LLMQueryRewriter)

__all__ = [
    "BaseQueryRewriter",
    "LLMQueryRewriter",
    "QueryRewritePlan",
    "QueryRewriterFactory",
    "QuerySlot",
    "RewrittenQuery",
    "SourceHint",
]

