"""Integration tests for rewritten-query retrieval and RRF fusion."""

from __future__ import annotations

from typing import Any

from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
from src.core.query_engine.query_processor import QueryProcessor
from src.core.types import RetrievalResult
from src.libs.query_rewriter import (
    BaseQueryRewriter,
    QueryRewritePlan,
    RewrittenQuery,
)


class FixedRewriter(BaseQueryRewriter):
    def __init__(self) -> None:
        self.calls = 0

    def rewrite(
        self,
        query: str,
        *,
        context: str | None = None,
        trace: Any | None = None,
    ) -> QueryRewritePlan:
        self.calls += 1
        return QueryRewritePlan(
            original_query=query,
            rewrite_needed=True,
            strategy="test expansion",
            rewritten_queries=[
                RewrittenQuery("q1", "formal metric", "formal revenue metric"),
                RewrittenQuery("q2", "table wording", "revenue table 2023"),
            ],
        )


class QueryAwareDenseRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.queries.append(query)
        chunk_id = {
            "original": "original_hit",
            "formal revenue metric": "shared_hit",
            "revenue table 2023": "shared_hit",
            "external query": "external_hit",
        }.get(query, "other")
        return [RetrievalResult(chunk_id, 1.0, chunk_id, {})]


def make_search(
    dense: QueryAwareDenseRetriever,
    rewriter: FixedRewriter,
) -> HybridSearch:
    return HybridSearch(
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=None,
        fusion=RRFFusion(k=60),
        query_rewriter=rewriter,
        config=HybridSearchConfig(
            enable_dense=True,
            enable_sparse=False,
            dense_weight=1.0,
            sparse_weight=0.0,
            rewrite_weight=0.7,
            parallel_retrieval=False,
        ),
    )


def test_rewrites_are_retrieved_and_fused_with_original() -> None:
    dense = QueryAwareDenseRetriever()
    rewriter = FixedRewriter()
    search = make_search(dense, rewriter)

    details = search.search("original", top_k=2, return_details=True)

    assert dense.queries == [
        "original",
        "formal revenue metric",
        "revenue table 2023",
    ]
    assert [result.chunk_id for result in details.results] == [
        "original_hit",
        "shared_hit",
    ]
    assert details.searched_queries == dense.queries
    assert details.rewrite_plan.strategy == "test expansion"


def test_external_variants_bypass_internal_rewriter() -> None:
    dense = QueryAwareDenseRetriever()
    rewriter = FixedRewriter()
    search = make_search(dense, rewriter)

    details = search.search(
        "original",
        query_variants=["external query", "external query", "original", ""],
        return_details=True,
    )

    assert rewriter.calls == 0
    assert dense.queries == ["original", "external query"]
    assert details.rewrite_plan.strategy == "external_query_variants"
