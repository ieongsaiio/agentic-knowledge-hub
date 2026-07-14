"""Unit tests for the deterministic benchmark evaluator adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.benchmark_evaluator import BenchmarkEvaluator
from src.observability.evaluation.evidence_judge import (
    EvidenceJudgement,
    EvidenceMatch,
)


def _benchmark_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-1",
        query="What was the annual revenue?",
        reference_answer="Annual revenue was $1,200.",
        evidences=[
            BenchmarkEvidence(
                document_name="annual-report.pdf",
                page_number=7,
                text="Annual revenue was $1,200.",
            )
        ],
        metadata={"source": "unit-test"},
    )


def test_accepts_direct_benchmark_case_ground_truth() -> None:
    evaluator = BenchmarkEvaluator(metrics=["document_hit_rate@1", "page_hit_rate@1"])
    retrieved = [
        {
            "text": "Annual revenue was $1,200.",
            "metadata": {
                "filename": "annual-report.pdf",
                "page_number": 7,
            },
        }
    ]

    result = evaluator.evaluate(
        "What was the annual revenue?",
        retrieved,
        ground_truth=_benchmark_case(),
    )

    assert result == {
        "document_hit_rate@1": 1.0,
        "page_hit_rate@1": 1.0,
    }


def test_normalized_dict_maps_expected_fields_to_evidence() -> None:
    evaluator = BenchmarkEvaluator(metrics=["document_hit_rate@2"])
    metrics = Mock()
    metrics.evaluate_case.return_value = {"document_hit_rate@2": 1}
    evaluator._benchmark_metrics = metrics
    ground_truth = {
        "case_id": "normalized-1",
        "query": "Which filings contain the evidence?",
        "reference_answer": "Both filings.",
        "expected_documents": ["first.pdf", "second.pdf"],
        "expected_pages": [3, 9],
        "expected_evidence": ["First supporting fact.", "Second supporting fact."],
        "metadata": {"split": "validation"},
    }
    retrieved = [{"metadata": {"filename": "second.pdf"}}]

    result = evaluator.evaluate(
        ground_truth["query"],
        retrieved,
        ground_truth=ground_truth,
    )

    assert result == {"document_hit_rate@2": 1.0}
    adapted_case = metrics.evaluate_case.call_args.kwargs["case"]
    assert isinstance(adapted_case, BenchmarkCase)
    assert adapted_case.case_id == "normalized-1"
    assert adapted_case.query == ground_truth["query"]
    assert adapted_case.reference_answer == "Both filings."
    assert adapted_case.expected_documents == ["first.pdf", "second.pdf"]
    assert adapted_case.expected_pages == [3, 9]
    assert adapted_case.expected_evidence == [
        "First supporting fact.",
        "Second supporting fact.",
    ]
    assert adapted_case.metadata == {"split": "validation"}
    assert metrics.evaluate_case.call_args.kwargs["retrieved"] is retrieved


def test_generated_answer_is_scored() -> None:
    evaluator = BenchmarkEvaluator(
        metrics=["answer_exact_match", "answer_token_f1", "numeric_accuracy"]
    )

    result = evaluator.evaluate(
        "What was the annual revenue?",
        [],
        generated_answer="Annual revenue was USD 1,200.",
        ground_truth=_benchmark_case(),
    )

    assert result["answer_exact_match"] == 0.0
    assert result["answer_token_f1"] == pytest.approx(10.0 / 11.0)
    assert result["numeric_accuracy"] == 1.0


def test_empty_retrieval_returns_valid_zero_scores() -> None:
    evaluator = BenchmarkEvaluator(
        metrics=[
            "document_hit_rate@5",
            "document_mrr@5",
            "page_hit_rate@5",
            "evidence_hit_rate@5",
            "evidence_mrr@5",
        ]
    )

    result = evaluator.evaluate(
        "What was the annual revenue?",
        [],
        ground_truth=_benchmark_case(),
    )

    assert result == {
        "document_hit_rate@5": 0.0,
        "document_mrr@5": 0.0,
        "evidence_hit_rate@5": 0.0,
        "evidence_mrr@5": 0.0,
        "page_hit_rate@5": 0.0,
    }


def test_evidence_metrics_share_one_llm_judgement() -> None:
    judge = Mock()
    judge.judge.return_value = EvidenceJudgement(
        matches=(
            EvidenceMatch(
                evidence_index=1,
                first_matching_rank=2,
                reason="Rank 2 contains equivalent evidence.",
            ),
        )
    )
    evaluator = BenchmarkEvaluator(
        metrics=["evidence_hit_rate@3", "evidence_mrr@5"],
        evidence_judge=judge,
    )
    retrieved = [
        {"text": "Table header"},
        {"text": "Annual revenue was $1,200."},
        {"text": "Unrelated text"},
    ]

    result = evaluator.evaluate(
        "What was the annual revenue?",
        retrieved,
        ground_truth=_benchmark_case(),
    )

    assert result == {
        "evidence_hit_rate@3": 1.0,
        "evidence_mrr@5": 0.5,
    }
    judge.judge.assert_called_once()
    assert judge.judge.call_args.args[1] is not retrieved
    assert judge.judge.call_args.args[1] == retrieved


def test_missing_ground_truth_raises_clear_error() -> None:
    evaluator = BenchmarkEvaluator(metrics=["document_hit_rate@1"])

    with pytest.raises(
        ValueError,
        match=r"requires ground_truth.*BenchmarkCase or normalized benchmark fields",
    ):
        evaluator.evaluate("A valid query", [])


def test_factory_creates_benchmark_provider() -> None:
    settings = SimpleNamespace(
        evaluation=SimpleNamespace(
            enabled=True,
            provider="benchmark",
            metrics=["answer_exact_match"],
        )
    )

    evaluator = EvaluatorFactory.create(settings)

    assert isinstance(evaluator, BenchmarkEvaluator)
    assert evaluator.metrics == ["answer_exact_match"]
