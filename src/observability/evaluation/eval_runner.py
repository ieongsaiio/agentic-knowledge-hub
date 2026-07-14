"""Evaluation runner for batch quality assessment.

EvalRunner reads a golden test set, runs HybridSearch for each test case,
optionally generates answers, then invokes the configured Evaluator(s) to
produce a structured evaluation report.

Design Principles:
- Config-Driven: Evaluator selected via settings.yaml.
- Observable: Produces EvalReport with per-query details.
- Decoupled: Works with any BaseEvaluator implementation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.benchmark.base_benchmark import BenchmarkCase
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.answer_generator import GeneratedAnswer

logger = logging.getLogger(__name__)

_ANSWER_METRICS = {
    "answer_exact_match",
    "answer_token_f1",
    "numeric_accuracy",
}


def requires_generated_answer(settings: Any) -> bool:
    """Return whether configured evaluators consume a generated answer."""
    if settings is None:
        return False

    evaluation = (
        settings.get("evaluation")
        if isinstance(settings, Mapping)
        else getattr(settings, "evaluation", None)
    )
    if evaluation is None:
        return False

    if isinstance(evaluation, Mapping):
        provider = evaluation.get("provider", "")
        backends = evaluation.get("backends", [])
        metrics = evaluation.get("metrics", [])
    else:
        provider = getattr(evaluation, "provider", "")
        backends = getattr(evaluation, "backends", [])
        metrics = getattr(evaluation, "metrics", [])

    normalized_provider = str(provider).strip().lower()
    normalized_backends = {
        str(backend).strip().lower()
        for backend in (backends if isinstance(backends, (list, tuple, set)) else [])
    }
    if normalized_provider == "ragas" or "ragas" in normalized_backends:
        return True

    metric_names = metrics if isinstance(metrics, (list, tuple, set)) else [metrics]
    return any(
        str(metric).strip().lower().split("@", 1)[0] in _ANSWER_METRICS
        for metric in metric_names
    )


@dataclass
class GoldenTestCase:
    """A single evaluation test case from the golden test set.

    Attributes:
        query: The test query string.
        expected_chunk_ids: Ground-truth chunk IDs for IR metrics.
        expected_sources: Ground-truth source file names (optional).
        reference_answer: Reference answer text for LLM-as-Judge (optional).
    """

    query: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    case_id: Optional[str] = None
    expected_pages: List[int] = field(default_factory=list)
    expected_evidence: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenTestCase:
        evidences = data.get("evidences") or []

        def evidence_value(evidence: Any, name: str) -> Any:
            if isinstance(evidence, Mapping):
                return evidence.get(name)
            return getattr(evidence, name, None)

        evidence_sources = [
            value
            for evidence in evidences
            if (value := evidence_value(evidence, "document_name")) is not None
        ]
        evidence_pages = [
            value
            for evidence in evidences
            if (value := evidence_value(evidence, "page_number")) is not None
        ]
        evidence_texts = [
            value
            for evidence in evidences
            if (value := evidence_value(evidence, "text")) is not None
        ]
        expected_sources = data.get("expected_sources")
        if expected_sources is None:
            expected_sources = data.get("expected_documents", evidence_sources)

        return cls(
            query=data["query"],
            expected_chunk_ids=list(data.get("expected_chunk_ids", data.get("ids", []))),
            expected_sources=list(expected_sources),
            reference_answer=data.get("reference_answer"),
            case_id=data.get("case_id"),
            expected_pages=list(data.get("expected_pages", evidence_pages)),
            expected_evidence=list(data.get("expected_evidence", evidence_texts)),
        )

    @property
    def expected_documents(self) -> List[str]:
        """Return the normalized document-name alias."""
        return self.expected_sources


@dataclass
class QueryResult:
    """Result of evaluating a single test case.

    Attributes:
        query: The test query.
        retrieved_chunk_ids: IDs of chunks actually retrieved.
        generated_answer: The generated answer (if applicable).
        metrics: Evaluation metrics for this query.
        elapsed_ms: Time taken for retrieval + evaluation.
    """

    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    case_id: Optional[str] = None
    reference_answer: Optional[str] = None
    retrieved_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this per-query result."""
        return {
            "query": self.query,
            "case_id": self.case_id,
            "reference_answer": self.reference_answer,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieved_results": self.retrieved_results,
            "generated_answer": self.generated_answer,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QueryResult:
        """Restore a per-query result from a checkpoint record."""
        return cls(
            query=str(data.get("query") or ""),
            case_id=data.get("case_id"),
            reference_answer=data.get("reference_answer"),
            retrieved_chunk_ids=list(data.get("retrieved_chunk_ids") or []),
            retrieved_results=list(data.get("retrieved_results") or []),
            generated_answer=data.get("generated_answer"),
            metrics={
                str(key): float(value)
                for key, value in (data.get("metrics") or {}).items()
            },
            elapsed_ms=float(data.get("elapsed_ms") or 0.0),
        )


@dataclass
class EvalReport:
    """Aggregated evaluation report across all test cases.

    Attributes:
        query_results: Per-query evaluation results.
        aggregate_metrics: Averaged metrics across all queries.
        total_elapsed_ms: Total time for the entire evaluation.
        evaluator_name: Name of the evaluator used.
        test_set_path: Path to the golden test set file.
    """

    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise report to dictionary."""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {k: round(v, 4) for k, v in self.aggregate_metrics.items()},
            "query_count": len(self.query_results),
            "query_results": [qr.to_dict() for qr in self.query_results],
        }


def load_test_set(path: str | Path) -> List[GoldenTestCase]:
    """Load golden test set from a JSON file.

    Args:
        path: Path to the golden test set JSON file.

    Returns:
        List of TestCase instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "test_cases" not in data:
        raise ValueError("Invalid golden test set format: missing 'test_cases' key.")

    return [GoldenTestCase.from_dict(tc) for tc in data["test_cases"]]


class EvalRunner:
    """Runs batch evaluation against a golden test set.

    This class orchestrates:
    1. Loading the golden test set
    2. Running HybridSearch for each query
    3. Optionally generating answers
    4. Invoking the evaluator to score each result
    5. Aggregating metrics into an EvalReport

    Example::

        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
        )
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: Optional[BaseEvaluator] = None,
        answer_generator: Any = None,
        answer_overrides: Optional[Dict[int, str]] = None,
        reranker: Any = None,
    ) -> None:
        """Initialize EvalRunner.

        Args:
            settings: Application settings.
            hybrid_search: HybridSearch instance for retrieval.
            evaluator: BaseEvaluator instance for scoring.
            answer_generator: Optional callable(query, chunks) -> str
                for generating answers. If None, a simple concatenation
                is used as a placeholder.
            answer_overrides: Optional dict mapping test case index (0-based)
                to a user-provided answer string. When present, the override
                answer is used instead of auto-generation for that test case.
            reranker: Optional CoreReranker instance for reranking results.
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.answer_overrides = answer_overrides or {}
        self.reranker = reranker

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """Run evaluation on the golden test set.

        Args:
            test_set_path: Path to golden_test_set.json.
            top_k: Number of chunks to retrieve per query.
            collection: Optional collection name filter.

        Returns:
            EvalReport with per-query and aggregate metrics.

        Raises:
            FileNotFoundError: If test set file doesn't exist.
            ValueError: If evaluator or hybrid_search is not set.
        """
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")

        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        report = self.run_cases(
            test_cases,
            top_k=top_k,
            collection=collection,
        )
        report.test_set_path = str(test_set_path)
        return report

    def run_cases(
        self,
        cases: list[BenchmarkCase | GoldenTestCase],
        top_k: int = 10,
        collection: Optional[str] = None,
        checkpoint_path: str | Path | None = None,
    ) -> EvalReport:
        """Run evaluation for normalized benchmark or legacy golden cases."""
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")
        if not cases:
            raise ValueError("Evaluation case list is empty.")

        logger.info(
            "Starting evaluation: %d cases, evaluator=%s",
            len(cases),
            type(self.evaluator).__name__,
        )

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__,
        )

        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        completed = self._load_checkpoint(checkpoint)
        if completed:
            logger.info(
                "Resuming evaluation from %s with %d completed cases",
                checkpoint,
                len(completed),
            )

        for idx, tc in enumerate(cases):
            case_key = self._case_key(tc, idx)
            cached = completed.get(case_key)
            if cached is not None and cached.query == tc.query:
                report.query_results.append(cached)
                continue
            logger.info(
                "Evaluating [%d/%d]: %s",
                idx + 1,
                len(cases),
                tc.query[:60],
            )
            # Use user-provided answer override if available for this index
            answer_override = self.answer_overrides.get(idx)
            case_started = time.monotonic()
            try:
                qr = self._evaluate_single(
                    tc,
                    top_k=top_k,
                    collection=collection,
                    answer_override=answer_override,
                )
            except Exception as exc:
                logger.warning(
                    "Case failed for '%s': %s",
                    tc.query[:40],
                    exc,
                    exc_info=True,
                )
                qr = QueryResult(
                    query=tc.query,
                    case_id=getattr(tc, "case_id", None),
                    reference_answer=getattr(tc, "reference_answer", None),
                    elapsed_ms=(time.monotonic() - case_started) * 1000.0,
                )
            report.query_results.append(qr)
            if checkpoint is not None:
                self._append_checkpoint(checkpoint, case_key, qr)

        report.total_elapsed_ms = max(
            sum(result.elapsed_ms for result in report.query_results),
            0.001,
        )
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    @staticmethod
    def _case_key(case: BenchmarkCase | GoldenTestCase, index: int) -> str:
        case_id = getattr(case, "case_id", None)
        return f"id:{case_id}" if case_id is not None else f"index:{index}"

    @staticmethod
    def _load_checkpoint(path: Path | None) -> Dict[str, QueryResult]:
        if path is None or not path.is_file():
            return {}
        completed: Dict[str, QueryResult] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    case_key = str(payload["case_key"])
                    completed[case_key] = QueryResult.from_dict(payload["result"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Ignoring invalid checkpoint line %d in %s: %s",
                        line_number,
                        path,
                        exc,
                    )
        return completed

    @staticmethod
    def _append_checkpoint(path: Path, case_key: str, result: QueryResult) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_key": case_key,
            "result": result.to_dict(),
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _evaluate_single(
        self,
        test_case: BenchmarkCase | GoldenTestCase,
        top_k: int = 10,
        collection: Optional[str] = None,
        answer_override: Optional[str] = None,
    ) -> QueryResult:
        """Evaluate a single test case.

        Args:
            test_case: The test case to evaluate.
            top_k: Number of results to retrieve.
            collection: Optional collection filter.
            answer_override: User-provided answer text. When set, used
                instead of auto-generated answer from chunks.

        Returns:
            QueryResult with metrics for this test case.
        """
        t0 = time.monotonic()
        qr = QueryResult(
            query=test_case.query,
            case_id=getattr(test_case, "case_id", None),
            reference_answer=getattr(test_case, "reference_answer", None),
        )

        # Step 1: Retrieve chunks
        retrieved_chunks = self._retrieve(test_case.query, top_k, collection)
        qr.retrieved_chunk_ids = [self._get_chunk_id(c) for c in retrieved_chunks]
        qr.retrieved_results = [self._retrieved_result_to_dict(c) for c in retrieved_chunks]

        # Step 2: Generate an answer only when a configured metric requires one.
        if answer_override is not None:
            answer = answer_override
        elif self._requires_generated_answer():
            answer = self._generate_answer(
                test_case.query,
                retrieved_chunks,
                test_case,
            )
        else:
            answer = None
        qr.generated_answer = answer

        # Step 3: Build ground truth
        ground_truth = self._build_ground_truth(test_case)

        # Step 4: Evaluate
        try:
            metrics = self.evaluator.evaluate(  # type: ignore[union-attr]
                query=test_case.query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
            )
            qr.metrics = metrics
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", test_case.query[:40], exc)
            qr.metrics = {}

        qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return qr

    def _requires_generated_answer(self) -> bool:
        """Return whether configured evaluators consume a generated answer."""
        if self.settings is None:
            return self.answer_generator is not None
        return requires_generated_answer(self.settings)

    def _retrieve(
        self,
        query: str,
        top_k: int,
        collection: Optional[str],
    ) -> List[Any]:
        """Retrieve chunks using HybridSearch + optional Reranking.

        Falls back to an empty list if search is not configured.
        """
        if self.hybrid_search is None:
            logger.warning("No HybridSearch configured; returning empty results.")
            return []

        try:
            has_reranker = self.reranker is not None and getattr(self.reranker, "is_enabled", False)
            if has_reranker:
                retrieval_settings = getattr(self.settings, "retrieval", None)
                configured_fusion_top_k = int(
                    getattr(retrieval_settings, "fusion_top_k", 0) or 0
                )
                initial_top_k = max(top_k * 2, configured_fusion_top_k)
            else:
                initial_top_k = top_k

            search_kwargs: Dict[str, Any] = {
                "query": query,
                "top_k": initial_top_k,
            }
            if collection is not None:
                search_kwargs["filters"] = {"collection": collection}
            search_result = self.hybrid_search.search(**search_kwargs)
            results = search_result if isinstance(search_result, list) else search_result.results

            # Apply reranking if enabled
            if has_reranker and results:
                rerank_result = self.reranker.rerank(
                    query=query,
                    results=results,
                    top_k=top_k,
                )
                results = rerank_result.results

            return results
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", query[:40], exc)
            return []

    def _generate_answer(
        self,
        query: str,
        chunks: List[Any],
        test_case: Optional[BenchmarkCase | GoldenTestCase] = None,
    ) -> str:
        """Generate an answer from retrieved chunks.

        If a custom answer_generator is provided, use it.
        Otherwise, concatenate chunk texts as a simple placeholder.
        """
        if self.answer_generator is not None:
            try:
                generator = self.answer_generator
                if isinstance(generator, str):
                    return generator

                generate = getattr(generator, "generate", None)
                if callable(generate):
                    try:
                        generated = generate(query, chunks, case=test_case)
                    except TypeError:
                        generated = generate(query, chunks)
                elif callable(generator):
                    generated = generator(query, chunks)
                else:
                    generated = generator

                if isinstance(generated, GeneratedAnswer):
                    return generated.content
                if isinstance(generated, str):
                    return generated
                return str(generated)
            except Exception as exc:
                logger.warning("Answer generation failed: %s", exc)

        # Fallback: concatenate chunk texts
        texts = []
        for c in chunks:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            elif hasattr(c, "text"):
                texts.append(str(getattr(c, "text")))
            else:
                texts.append(str(c))

        return " ".join(texts[:5])  # first 5 chunks

    @staticmethod
    def _build_ground_truth(
        test_case: BenchmarkCase | GoldenTestCase,
    ) -> Dict[str, Any]:
        """Build the common evaluator ground-truth contract."""
        if isinstance(test_case, BenchmarkCase):
            expected_ids: List[str] = []
            expected_documents = list(test_case.expected_documents)
            expected_pages = list(test_case.expected_pages)
            expected_evidence = list(test_case.expected_evidence)
        else:
            expected_ids = list(test_case.expected_chunk_ids)
            expected_documents = list(test_case.expected_sources)
            expected_pages = list(test_case.expected_pages)
            expected_evidence = list(test_case.expected_evidence)

        ground_truth: Dict[str, Any] = {
            "case_id": getattr(test_case, "case_id", None),
            "query": test_case.query,
            "ids": expected_ids,
            "expected_sources": expected_documents,
            "expected_documents": expected_documents,
            "expected_pages": expected_pages,
            "expected_evidence": expected_evidence,
            "reference_answer": getattr(test_case, "reference_answer", None),
        }
        if isinstance(test_case, BenchmarkCase):
            ground_truth["benchmark_case"] = test_case
            ground_truth["metadata"] = dict(test_case.metadata)
        return ground_truth

    @classmethod
    def _retrieved_result_to_dict(cls, chunk: Any) -> Dict[str, Any]:
        """Extract useful retrieval details where available."""
        details: Dict[str, Any] = {"id": cls._get_chunk_id(chunk)}
        if isinstance(chunk, str):
            details["text"] = chunk
            return details

        if isinstance(chunk, Mapping):
            text = next(
                (
                    chunk[key]
                    for key in ("text", "content", "page_content")
                    if chunk.get(key) is not None
                ),
                None,
            )
            metadata = chunk.get("metadata")
            score = chunk.get("score", chunk.get("relevance_score"))
        else:
            text = next(
                (
                    getattr(chunk, key)
                    for key in ("text", "content", "page_content")
                    if getattr(chunk, key, None) is not None
                ),
                None,
            )
            metadata = getattr(chunk, "metadata", None)
            score = getattr(
                chunk,
                "score",
                getattr(chunk, "relevance_score", None),
            )

        if text is not None:
            details["text"] = str(text)
        if isinstance(metadata, Mapping):
            details["metadata"] = dict(metadata)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            details["score"] = score
        return details

    @staticmethod
    def _get_chunk_id(chunk: Any) -> str:
        """Extract chunk ID from various representations."""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, Mapping):
            for key in ("id", "chunk_id"):
                if key in chunk:
                    return str(chunk[key])
            return str(chunk)
        if hasattr(chunk, "chunk_id"):
            return str(getattr(chunk, "chunk_id"))
        if hasattr(chunk, "id"):
            return str(getattr(chunk, "id"))
        return str(chunk)

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """Compute average metrics across all query results.

        Args:
            results: List of QueryResult with per-query metrics.

        Returns:
            Dictionary of average metric values.
        """
        if not results:
            return {}

        # Collect all metric keys
        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        # Average each metric
        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [qr.metrics[key] for qr in results if key in qr.metrics]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages
