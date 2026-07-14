"""Evaluation-provider adapter for benchmark and LLM evidence metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.benchmark_metrics import (
    BenchmarkMetrics,
    parse_metric_name,
)
from src.observability.evaluation.evidence_judge import LLMEvidenceJudge

_EVIDENCE_METRICS = {"evidence_hit_rate", "evidence_mrr"}


class BenchmarkEvaluator(BaseEvaluator):
    """Adapt benchmark cases to the common evaluator provider interface."""

    def __init__(
        self,
        settings: Any = None,
        metrics: Sequence[str] | None = None,
        evidence_judge: Any = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        self.metrics = [str(metric).strip().lower() for metric in (metrics or [])]
        self._benchmark_metrics = BenchmarkMetrics(metrics=self.metrics)
        self._evidence_judge = evidence_judge
        self._evidence_cutoffs = self._find_evidence_cutoffs(self.metrics)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: list[Any],
        generated_answer: str | None = None,
        ground_truth: Any | None = None,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute configured metrics for one normalized benchmark case."""
        self.validate_query(query)
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")

        benchmark_case = self._benchmark_case_from_ground_truth(ground_truth)
        evidence_ranks: tuple[int | None, ...] = ()
        if self._evidence_cutoffs and retrieved_chunks and benchmark_case.evidences:
            judge = self._get_evidence_judge()
            judgement = judge.judge(
                benchmark_case,
                retrieved_chunks[: max(self._evidence_cutoffs)],
                trace=trace,
            )
            evidence_ranks = judgement.match_ranks

        result = self._benchmark_metrics.evaluate_case(
            case=benchmark_case,
            retrieved=retrieved_chunks,
            answer=generated_answer,
            evidence_ranks=evidence_ranks,
        )
        if not isinstance(result, Mapping):
            raise TypeError("BenchmarkMetrics.evaluate_case() must return a mapping")

        return {
            str(metric): float(value)
            for metric, value in sorted(result.items(), key=lambda item: str(item[0]))
        }

    def _get_evidence_judge(self) -> Any:
        if self._evidence_judge is None:
            self._evidence_judge = LLMEvidenceJudge(self.settings)
        return self._evidence_judge

    @staticmethod
    def _find_evidence_cutoffs(metrics: Sequence[str]) -> list[int]:
        cutoffs = []
        for metric in metrics:
            try:
                base, cutoff = parse_metric_name(metric)
            except (TypeError, ValueError):
                continue
            if base in _EVIDENCE_METRICS and cutoff is not None:
                cutoffs.append(cutoff)
        return cutoffs

    @staticmethod
    def _metrics_from_settings(settings: Any) -> list[str]:
        if settings is None:
            return []

        evaluation = (
            settings.get("evaluation", settings)
            if isinstance(settings, Mapping)
            else getattr(settings, "evaluation", settings)
        )
        if isinstance(evaluation, Mapping):
            raw_metrics = evaluation.get("metrics")
        else:
            raw_metrics = getattr(evaluation, "metrics", None)

        if raw_metrics is None:
            return []
        if isinstance(raw_metrics, (str, bytes)) or not isinstance(raw_metrics, Sequence):
            raise ValueError("evaluation.metrics must be a sequence of metric names")
        return [str(metric) for metric in raw_metrics]

    @classmethod
    def _benchmark_case_from_ground_truth(cls, ground_truth: Any) -> BenchmarkCase:
        if ground_truth is None:
            raise ValueError(
                "BenchmarkEvaluator requires ground_truth containing a "
                "BenchmarkCase or normalized benchmark fields"
            )
        if isinstance(ground_truth, BenchmarkCase):
            return ground_truth
        if not isinstance(ground_truth, Mapping):
            raise ValueError("BenchmarkEvaluator ground_truth must be a BenchmarkCase or mapping")

        if "benchmark_case" in ground_truth:
            case = ground_truth["benchmark_case"]
            if not isinstance(case, BenchmarkCase):
                raise ValueError("ground_truth['benchmark_case'] must be a BenchmarkCase")
            return case

        required = (
            "case_id",
            "query",
            "reference_answer",
            "expected_documents",
            "expected_pages",
            "expected_evidence",
        )
        missing = [name for name in required if name not in ground_truth]
        if missing:
            raise ValueError(
                "BenchmarkEvaluator ground_truth is missing required field(s): "
                + ", ".join(missing)
            )

        documents = cls._ground_truth_list(ground_truth, "expected_documents")
        pages = cls._ground_truth_list(ground_truth, "expected_pages")
        evidence_texts = cls._ground_truth_list(ground_truth, "expected_evidence")
        lengths = {len(documents), len(pages), len(evidence_texts)}
        if len(lengths) != 1:
            raise ValueError(
                "ground_truth expected_documents, expected_pages, and "
                "expected_evidence must have equal lengths"
            )

        evidences = [
            BenchmarkEvidence(
                document_name=document,
                page_number=page,
                text=text,
            )
            for document, page, text in zip(documents, pages, evidence_texts)
        ]

        raw_metadata = ground_truth.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("ground_truth metadata must be a mapping")

        return BenchmarkCase(
            case_id=ground_truth["case_id"],
            query=ground_truth["query"],
            reference_answer=ground_truth["reference_answer"],
            evidences=evidences,
            metadata=dict(raw_metadata),
        )

    @staticmethod
    def _ground_truth_list(
        ground_truth: Mapping[str, Any],
        field_name: str,
    ) -> list[Any]:
        value = ground_truth[field_name]
        if not isinstance(value, list):
            raise ValueError(f"ground_truth {field_name} must be a list")
        return value
