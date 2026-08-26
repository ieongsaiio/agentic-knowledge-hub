"""Hybrid Search Engine orchestrating Dense + Sparse retrieval with RRF Fusion.

This module implements the HybridSearch class that combines:
1. QueryProcessor: Preprocess queries and extract keywords/filters
2. DenseRetriever: Semantic search using embeddings
3. SparseRetriever: Keyword search using BM25
4. RRFFusion: Combine results using Reciprocal Rank Fusion

Design Principles:
- Graceful Degradation: If one retrieval path fails, fall back to the other
- Pluggable: All components injected via constructor for testability
- Observable: TraceContext integration for debugging and monitoring
- Config-Driven: Top-k and other parameters read from settings
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.types import ProcessedQuery, RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.settings import Settings
    from src.libs.query_rewriter import BaseQueryRewriter, QueryRewritePlan

logger = logging.getLogger(__name__)


def _snapshot_results(
    results: Optional[List[RetrievalResult]],
) -> List[Dict[str, Any]]:
    """Create a serialisable snapshot of retrieval results for trace storage.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        List of dicts with chunk_id, score, full text, source.
    """
    if not results:
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "text": r.text or "",
            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
        }
        for r in results
    ]


@dataclass
class HybridSearchConfig:
    """Configuration for HybridSearch.
    
    Attributes:
        dense_top_k: Number of results from dense retrieval
        sparse_top_k: Number of results from sparse retrieval
        fusion_top_k: Final number of results after fusion
        enable_dense: Whether to use dense retrieval
        enable_sparse: Whether to use sparse retrieval
        parallel_retrieval: Whether to run retrievals in parallel
        metadata_filter_post: Apply metadata filters after fusion (fallback)
    """
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True
    rewrite_weight: float = 0.7
    max_rewrite_queries: int = 4

    def __post_init__(self) -> None:
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("at least one retrieval weight must be greater than zero")
        if self.rewrite_weight < 0:
            raise ValueError("rewrite_weight must be non-negative")
        if self.max_rewrite_queries <= 0:
            raise ValueError("max_rewrite_queries must be greater than zero")


@dataclass
class HybridSearchResult:
    """Result of a hybrid search operation.
    
    Attributes:
        results: Final ranked list of RetrievalResults
        dense_results: Results from dense retrieval (for debugging)
        sparse_results: Results from sparse retrieval (for debugging)
        dense_error: Error message if dense retrieval failed
        sparse_error: Error message if sparse retrieval failed
        used_fallback: Whether fallback mode was used
        processed_query: The processed query (for debugging)
    """
    results: List[RetrievalResult] = field(default_factory=list)
    dense_results: Optional[List[RetrievalResult]] = None
    sparse_results: Optional[List[RetrievalResult]] = None
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    used_fallback: bool = False
    processed_query: Optional[ProcessedQuery] = None
    rewrite_plan: Optional[Any] = None
    searched_queries: List[str] = field(default_factory=list)


class HybridSearch:
    """Hybrid Search Engine combining Dense and Sparse retrieval.
    
    This class orchestrates the complete hybrid search flow:
    1. Query Processing: Extract keywords and filters from raw query
    2. Parallel Retrieval: Run Dense and Sparse retrievers concurrently
    3. Fusion: Combine results using RRF algorithm
    4. Post-Filtering: Apply metadata filters if specified
    
    Design Principles Applied:
    - Graceful Degradation: If one path fails, use results from the other
    - Pluggable: All components via dependency injection
    - Observable: TraceContext support for debugging
    - Config-Driven: All parameters from settings
    
    Example:
        >>> # Initialize components
        >>> query_processor = QueryProcessor()
        >>> dense_retriever = DenseRetriever(settings, embedding_client, vector_store)
        >>> sparse_retriever = SparseRetriever(settings, bm25_indexer, vector_store)
        >>> fusion = RRFFusion(k=60)
        >>> 
        >>> # Create HybridSearch
        >>> hybrid = HybridSearch(
        ...     settings=settings,
        ...     query_processor=query_processor,
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ...     fusion=fusion
        ... )
        >>> 
        >>> # Search
        >>> results = hybrid.search("如何配置 Azure OpenAI？", top_k=10)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
        config: Optional[HybridSearchConfig] = None,
        query_rewriter: Optional[BaseQueryRewriter] = None,
    ) -> None:
        """Initialize HybridSearch with components.
        
        Args:
            settings: Application settings for extracting configuration.
            query_processor: QueryProcessor for preprocessing queries.
            dense_retriever: DenseRetriever for semantic search.
            sparse_retriever: SparseRetriever for keyword search.
            fusion: RRFFusion for combining results.
            config: Optional HybridSearchConfig. If not provided, extracted from settings.
        
        Note:
            At least one of dense_retriever or sparse_retriever must be provided
            for search to function. The search will gracefully degrade if one
            is unavailable or fails.
        """
        self.query_processor = query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        
        # Extract config from settings or use provided/default
        self.config = config or self._extract_config(settings)
        self.query_rewriter = query_rewriter or self._create_query_rewriter(settings)
        
        logger.info(
            f"HybridSearch initialized: dense={self.dense_retriever is not None}, "
            f"sparse={self.sparse_retriever is not None}, "
            f"rewriter={self.query_rewriter is not None}, "
            f"config={self.config}"
        )

    @staticmethod
    def _create_query_rewriter(
        settings: Optional[Settings],
    ) -> Optional[BaseQueryRewriter]:
        if settings is None:
            return None
        retrieval = getattr(settings, "retrieval", None)
        config = getattr(retrieval, "query_rewriter", None)
        if config is None or not getattr(config, "enabled", False):
            return None
        from src.libs.query_rewriter import QueryRewriterFactory

        return QueryRewriterFactory.create(settings)
    
    def _extract_config(self, settings: Optional[Settings]) -> HybridSearchConfig:
        """Extract HybridSearchConfig from Settings.
        
        Args:
            settings: Application settings object.
            
        Returns:
            HybridSearchConfig with values from settings or defaults.
        """
        if settings is None:
            return HybridSearchConfig()
        
        retrieval_config = getattr(settings, 'retrieval', None)
        if retrieval_config is None:
            return HybridSearchConfig()
        
        return HybridSearchConfig(
            dense_top_k=getattr(retrieval_config, 'dense_top_k', 20),
            sparse_top_k=getattr(retrieval_config, 'sparse_top_k', 20),
            fusion_top_k=getattr(retrieval_config, 'fusion_top_k', 10),
            enable_dense=getattr(retrieval_config, 'enable_dense', True),
            enable_sparse=getattr(retrieval_config, 'enable_sparse', True),
            dense_weight=getattr(retrieval_config, 'dense_weight', 0.5),
            sparse_weight=getattr(retrieval_config, 'sparse_weight', 0.5),
            parallel_retrieval=True,
            metadata_filter_post=True,
            rewrite_weight=getattr(
                getattr(retrieval_config, "query_rewriter", None),
                "rewrite_weight",
                0.7,
            ),
            max_rewrite_queries=getattr(
                getattr(retrieval_config, "query_rewriter", None),
                "max_queries",
                4,
            ),
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        return_details: bool = False,
        query_variants: Optional[List[str]] = None,
        rewrite_context: Optional[str] = None,
    ) -> List[RetrievalResult] | HybridSearchResult:
        """Perform hybrid search combining Dense and Sparse retrieval.
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return. If None, uses config.fusion_top_k.
            filters: Optional metadata filters (e.g., {"collection": "docs"}).
            trace: Optional TraceContext for observability.
            return_details: If True, return HybridSearchResult with debug info.
            query_variants: Optional externally prepared alternatives. Providing this
                bypasses the internal LLM rewriter to avoid double rewriting.
            rewrite_context: Optional conversation context for the internal rewriter.
        
        Returns:
            If return_details=False: List of RetrievalResult sorted by relevance.
            If return_details=True: HybridSearchResult with full details.
        
        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If both retrievers fail or are unavailable.
        
        Example:
            >>> results = hybrid.search("Azure configuration", top_k=5)
            >>> for r in results:
            ...     print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:50]}...")
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        
        effective_top_k = top_k if top_k is not None else self.config.fusion_top_k

        rewrite_plan = self._build_rewrite_plan(
            query,
            query_variants=query_variants,
            context=rewrite_context,
            trace=trace,
        )
        searched_queries = [query, *rewrite_plan.queries]
        
        logger.debug(f"HybridSearch: query='{query[:50]}...', top_k={effective_top_k}")
        
        # Step 1: Process the original query for filters and debug details.
        _t0 = time.monotonic()
        processed_query = self._process_query(query)
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("query_processing", {
                "method": "query_processor",
                "original_query": query,
                "keywords": processed_query.keywords,
            }, elapsed_ms=_elapsed)
        
        # Merge explicit filters with query-extracted filters
        merged_filters = self._merge_filters(processed_query.filters, filters)

        if len(searched_queries) > 1:
            return self._search_multiple_queries(
                searched_queries=searched_queries,
                original_processed_query=processed_query,
                merged_filters=merged_filters,
                effective_top_k=effective_top_k,
                rewrite_plan=rewrite_plan,
                trace=trace,
                return_details=return_details,
            )
        
        # Step 2: Run retrievals
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            filters=merged_filters,
            trace=trace,
        )

        # Step 3: Handle fallback scenarios
        used_fallback = False
        if dense_error and sparse_error:
            # Both failed - raise error
            raise RuntimeError(
                f"Both retrieval paths failed. "
                f"Dense error: {dense_error}. Sparse error: {sparse_error}"
            )
        elif dense_error:
            # Dense failed, use sparse only
            logger.warning(f"Dense retrieval failed, using sparse only: {dense_error}")
            used_fallback = True
            fused_results = sparse_results or []
        elif sparse_error:
            # Sparse failed, use dense only
            logger.warning(f"Sparse retrieval failed, using dense only: {sparse_error}")
            used_fallback = True
            fused_results = dense_results or []
        elif not dense_results and not sparse_results:
            # Both succeeded but returned empty
            fused_results = []
        else:
            # Step 4: Fuse results
            fused_results = self._fuse_results(
                dense_results=dense_results or [],
                sparse_results=sparse_results or [],
                top_k=effective_top_k,
                trace=trace,
            )

        # Step 5: Apply post-fusion metadata filters (if any)
        if merged_filters and self.config.metadata_filter_post:
            fused_results = self._apply_metadata_filters(fused_results, merged_filters)
        
        # Step 6: Limit to top_k
        final_results = fused_results[:effective_top_k]
        
        logger.debug(f"HybridSearch: returning {len(final_results)} results")
        
        if return_details:
            return HybridSearchResult(
                results=final_results,
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_error=dense_error,
                sparse_error=sparse_error,
                used_fallback=used_fallback,
                processed_query=processed_query,
                rewrite_plan=rewrite_plan,
                searched_queries=searched_queries,
            )
        
        return final_results

    def _build_rewrite_plan(
        self,
        query: str,
        *,
        query_variants: Optional[List[str]],
        context: Optional[str],
        trace: Optional[Any],
    ) -> QueryRewritePlan:
        """Resolve external variants or invoke the configured rewriter."""
        from src.libs.query_rewriter import QueryRewritePlan, RewrittenQuery

        if query_variants is not None:
            seen = {query.strip().casefold()}
            rewrites = []
            for index, variant in enumerate(query_variants, start=1):
                cleaned = str(variant).strip()
                identity = cleaned.casefold()
                if not cleaned or identity in seen:
                    continue
                seen.add(identity)
                rewrites.append(
                    RewrittenQuery(
                        query_id=f"external_{index}",
                        purpose="externally supplied retrieval query",
                        query=cleaned,
                    )
                )
                if len(rewrites) >= self.config.max_rewrite_queries:
                    break
            return QueryRewritePlan(
                original_query=query,
                rewrite_needed=bool(rewrites),
                strategy="external_query_variants",
                reason="Internal rewriting bypassed because variants were supplied.",
                rewritten_queries=rewrites,
            )

        if self.query_rewriter is None:
            return QueryRewritePlan(original_query=query)

        _t0 = time.monotonic()
        plan = self.query_rewriter.rewrite(query, context=context, trace=trace)
        elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage(
                "query_rewriting",
                plan.to_dict(),
                elapsed_ms=elapsed,
            )
        return plan

    def _search_multiple_queries(
        self,
        *,
        searched_queries: List[str],
        original_processed_query: ProcessedQuery,
        merged_filters: Dict[str, Any],
        effective_top_k: int,
        rewrite_plan: QueryRewritePlan,
        trace: Optional[Any],
        return_details: bool,
    ) -> List[RetrievalResult] | HybridSearchResult:
        """Retrieve each query and combine every ranking list with weighted RRF."""
        ranking_lists: List[List[RetrievalResult]] = []
        weights: List[float] = []
        all_dense: List[RetrievalResult] = []
        all_sparse: List[RetrievalResult] = []
        dense_errors: List[str] = []
        sparse_errors: List[str] = []
        rewrite_count = max(1, len(searched_queries) - 1)

        for query_index, current_query in enumerate(searched_queries):
            processed = (
                original_processed_query
                if query_index == 0
                else self._process_query(current_query)
            )
            dense, sparse, dense_error, sparse_error = self._run_retrievals(
                processed_query=processed,
                filters=merged_filters,
                trace=trace,
            )
            query_weight = (
                1.0
                if query_index == 0
                else self.config.rewrite_weight / rewrite_count
            )
            if dense:
                ranking_lists.append(dense)
                weights.append(self.config.dense_weight * query_weight)
                all_dense.extend(dense)
            if sparse:
                ranking_lists.append(sparse)
                weights.append(self.config.sparse_weight * query_weight)
                all_sparse.extend(sparse)
            if dense_error:
                dense_errors.append(f"{current_query}: {dense_error}")
            if sparse_error:
                sparse_errors.append(f"{current_query}: {sparse_error}")

        if not ranking_lists and (dense_errors or sparse_errors):
            raise RuntimeError(
                "All retrieval paths failed. "
                f"Dense errors: {'; '.join(dense_errors) or 'none'}. "
                f"Sparse errors: {'; '.join(sparse_errors) or 'none'}"
            )

        if self.fusion is None:
            fused = self._interleave_many(ranking_lists, effective_top_k)
        elif ranking_lists:
            fused = self.fusion.fuse_with_weights(
                ranking_lists=ranking_lists,
                weights=weights,
                top_k=effective_top_k,
                trace=trace,
            )
        else:
            fused = []

        if merged_filters and self.config.metadata_filter_post:
            fused = self._apply_metadata_filters(fused, merged_filters)
        final_results = fused[:effective_top_k]

        if trace is not None:
            trace.record_stage(
                "multi_query_fusion",
                {
                    "queries": searched_queries,
                    "ranking_list_count": len(ranking_lists),
                    "weights": weights,
                    "result_count": len(final_results),
                    "chunks": _snapshot_results(final_results),
                },
            )

        if return_details:
            return HybridSearchResult(
                results=final_results,
                dense_results=all_dense,
                sparse_results=all_sparse,
                dense_error="; ".join(dense_errors) or None,
                sparse_error="; ".join(sparse_errors) or None,
                used_fallback=bool(dense_errors or sparse_errors),
                processed_query=original_processed_query,
                rewrite_plan=rewrite_plan,
                searched_queries=searched_queries,
            )
        return final_results

    @staticmethod
    def _interleave_many(
        ranking_lists: List[List[RetrievalResult]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Round-robin fallback for an arbitrary number of ranking lists."""
        seen: set[str] = set()
        output: List[RetrievalResult] = []
        max_length = max((len(items) for items in ranking_lists), default=0)
        for rank in range(max_length):
            for items in ranking_lists:
                if rank >= len(items):
                    continue
                item = items[rank]
                if item.chunk_id in seen:
                    continue
                seen.add(item.chunk_id)
                output.append(item)
                if len(output) >= top_k:
                    return output
        return output
    
    def _process_query(self, query: str) -> ProcessedQuery:
        """Process raw query using QueryProcessor.
        
        Args:
            query: Raw query string.
            
        Returns:
            ProcessedQuery with keywords and filters.
        """
        if self.query_processor is None:
            # Fallback: create basic ProcessedQuery
            logger.warning("No QueryProcessor configured, using basic tokenization")
            keywords = query.split()
            return ProcessedQuery(
                original_query=query,
                keywords=keywords,
                filters={},
            )
        
        return self.query_processor.process(query)
    
    def _merge_filters(
        self,
        query_filters: Dict[str, Any],
        explicit_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge query-extracted filters with explicit filters.
        
        Explicit filters take precedence over query-extracted filters.
        
        Args:
            query_filters: Filters extracted from query by QueryProcessor.
            explicit_filters: Filters passed explicitly to search().
            
        Returns:
            Merged filter dictionary.
        """
        merged = query_filters.copy() if query_filters else {}
        if explicit_filters:
            merged.update(explicit_filters)
        return merged
    
    def _run_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals.
        
        Runs in parallel if configured, otherwise sequentially.
        
        Args:
            processed_query: The processed query with keywords.
            filters: Merged filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error).
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None
        
        # Determine what to run
        run_dense = (
            self.config.enable_dense 
            and self.dense_retriever is not None
        )
        run_sparse = (
            self.config.enable_sparse 
            and self.sparse_retriever is not None
            and processed_query.keywords  # Need keywords for sparse
        )
        
        if not run_dense and not run_sparse:
            # Nothing to run
            if self.dense_retriever is None and self.sparse_retriever is None:
                dense_error = "No retriever configured"
                sparse_error = "No retriever configured"
            return dense_results, sparse_results, dense_error, sparse_error
        
        if self.config.parallel_retrieval and run_dense and run_sparse:
            # Run in parallel
            dense_results, sparse_results, dense_error, sparse_error = (
                self._run_parallel_retrievals(processed_query, filters, trace)
            )
        else:
            # Run sequentially
            if run_dense:
                dense_results, dense_error = self._run_dense_retrieval(
                    processed_query.original_query, filters, trace
                )
            
            if run_sparse:
                sparse_results, sparse_error = self._run_sparse_retrieval(
                    processed_query.keywords, filters, trace
                )
        
        return dense_results, sparse_results, dense_error, sparse_error
    
    def _run_parallel_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals in parallel using ThreadPoolExecutor.
        
        Args:
            processed_query: The processed query.
            filters: Filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error).
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # Submit dense retrieval
            futures['dense'] = executor.submit(
                self._run_dense_retrieval,
                processed_query.original_query,
                filters,
                trace,
            )
            
            # Submit sparse retrieval
            futures['sparse'] = executor.submit(
                self._run_sparse_retrieval,
                processed_query.keywords,
                filters,
                trace,
            )
            
            # Collect results
            for name, future in futures.items():
                try:
                    results, error = future.result(timeout=30)
                    if name == 'dense':
                        dense_results = results
                        dense_error = error
                    else:
                        sparse_results = results
                        sparse_error = error
                except Exception as e:
                    error_msg = f"{name} retrieval failed with exception: {e}"
                    logger.error(error_msg)
                    if name == 'dense':
                        dense_error = error_msg
                    else:
                        sparse_error = error_msg
        
        return dense_results, sparse_results, dense_error, sparse_error
    
    def _run_dense_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run dense retrieval with error handling.
        
        Args:
            query: Original query string.
            filters: Filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (results, error). If successful, error is None.
        """
        if self.dense_retriever is None:
            return None, "Dense retriever not configured"
        
        try:
            _t0 = time.monotonic()
            results = self.dense_retriever.retrieve(
                query=query,
                top_k=self.config.dense_top_k,
                filters=filters,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "provider": getattr(self.dense_retriever, 'provider_name', 'unknown'),
                    "top_k": self.config.dense_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Dense retrieval error: {e}"
            logger.error(error_msg)
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "error": error_msg,
                    "result_count": 0,
                })
            return None, error_msg
    
    def _run_sparse_retrieval(
        self,
        keywords: List[str],
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run sparse retrieval with error handling.
        
        Args:
            keywords: List of keywords from QueryProcessor.
            filters: Filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (results, error). If successful, error is None.
        """
        if self.sparse_retriever is None:
            return None, "Sparse retriever not configured"
        
        if not keywords:
            return [], None  # No keywords, return empty (not an error)
        
        try:
            # Extract collection from filters if present
            collection = filters.get('collection') if filters else None
            
            _t0 = time.monotonic()
            results = self.sparse_retriever.retrieve(
                keywords=keywords,
                top_k=self.config.sparse_top_k,
                collection=collection,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("sparse_retrieval", {
                    "method": "bm25",
                    "keyword_count": len(keywords),
                    "top_k": self.config.sparse_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Sparse retrieval error: {e}"
            logger.error(error_msg)
            return None, error_msg
    
    def _fuse_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        """Fuse Dense and Sparse results using RRF.
        
        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            top_k: Number of results to return after fusion.
            trace: Optional TraceContext.
            
        Returns:
            Fused and ranked list of RetrievalResults.
        """
        if self.fusion is None:
            # Fallback: interleave results (simple round-robin)
            logger.warning("No fusion configured, using simple interleave")
            return self._interleave_results(dense_results, sparse_results, top_k)
        
        # Build ranking lists for RRF
        ranking_lists = []
        if dense_results:
            ranking_lists.append(dense_results)
        if sparse_results:
            ranking_lists.append(sparse_results)
        
        if not ranking_lists:
            return []
        
        if len(ranking_lists) == 1:
            # Only one source, no fusion needed
            return ranking_lists[0][:top_k]
        
        _t0 = time.monotonic()
        weights = [self.config.dense_weight, self.config.sparse_weight]
        fused = self.fusion.fuse_with_weights(
            ranking_lists=ranking_lists,
            weights=weights,
            top_k=top_k,
            trace=trace,
        )
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("fusion", {
                "method": "weighted_rrf",
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
                "input_lists": len(ranking_lists),
                "top_k": top_k,
                "result_count": len(fused),
                "chunks": _snapshot_results(fused),
            }, elapsed_ms=_elapsed)
        return fused

    def _interleave_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Simple interleave fallback when no fusion is configured.
        
        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            top_k: Maximum results to return.
            
        Returns:
            Interleaved results, deduped by chunk_id.
        """
        seen_ids = set()
        interleaved = []
        
        d_idx, s_idx = 0, 0
        while len(interleaved) < top_k and (d_idx < len(dense_results) or s_idx < len(sparse_results)):
            # Alternate between dense and sparse
            if d_idx < len(dense_results):
                r = dense_results[d_idx]
                d_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)
            
            if len(interleaved) >= top_k:
                break
            
            if s_idx < len(sparse_results):
                r = sparse_results[s_idx]
                s_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)
        
        return interleaved
    
    def _apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results (post-fusion fallback).
        
        This is a backup filter mechanism for cases where the underlying
        storage doesn't fully support the filter syntax.
        
        Args:
            results: Results to filter.
            filters: Filter conditions to apply.
            
        Returns:
            Filtered results.
        """
        if not filters:
            return results
        
        filtered = []
        for result in results:
            if self._matches_filters(result.metadata, filters):
                filtered.append(result)
        
        return filtered
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """Check if metadata matches all filter conditions.
        
        Args:
            metadata: Result metadata.
            filters: Filter conditions.
            
        Returns:
            True if all filters match, False otherwise.
        """
        for key, value in filters.items():
            if key == "collection":
                # Collection might be in different metadata keys
                meta_collection = (
                    metadata.get("collection") 
                    or metadata.get("source_collection")
                )
                if meta_collection != value:
                    return False
            elif key == "doc_type":
                if metadata.get("doc_type") != value:
                    return False
            elif key == "tags":
                # Tags is a list - check intersection
                meta_tags = metadata.get("tags", [])
                if not isinstance(value, list):
                    value = [value]
                if not set(meta_tags) & set(value):
                    return False
            elif key == "source_path":
                # Partial match for path
                source = metadata.get("source_path", "")
                if value not in source:
                    return False
            else:
                # Generic exact match
                if metadata.get(key) != value:
                    return False
        
        return True


def create_hybrid_search(
    settings: Optional[Settings] = None,
    query_processor: Optional[QueryProcessor] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    fusion: Optional[RRFFusion] = None,
    query_rewriter: Optional[BaseQueryRewriter] = None,
) -> HybridSearch:
    """Factory function to create HybridSearch with default components.
    
    This is a convenience function that creates a HybridSearch with
    default RRFFusion if not provided.
    
    Args:
        settings: Application settings.
        query_processor: QueryProcessor instance.
        dense_retriever: DenseRetriever instance.
        sparse_retriever: SparseRetriever instance.
        fusion: RRFFusion instance. If None, creates default with k=60.
        
    Returns:
        Configured HybridSearch instance.
    
    Example:
        >>> hybrid = create_hybrid_search(
        ...     settings=settings,
        ...     query_processor=QueryProcessor(),
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ... )
    """
    # Create default fusion if not provided
    if fusion is None:
        from src.core.query_engine.fusion import RRFFusion
        rrf_k = 60
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                rrf_k = getattr(retrieval_config, 'rrf_k', 60)
        fusion = RRFFusion(k=rrf_k)
    
    return HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
        query_rewriter=query_rewriter,
    )
