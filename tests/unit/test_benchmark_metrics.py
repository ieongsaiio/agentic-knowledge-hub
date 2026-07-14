"""Focused tests for deterministic benchmark metrics."""

from __future__ import annotations

import pytest

from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.observability.evaluation.benchmark_metrics import (
    BenchmarkMetrics,
    aggregate,
    normalize_document_name,
)


def _case(
    *,
    reference_answer: str = "The answer",
    evidences: list[BenchmarkEvidence] | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-1",
        query="What is the answer?",
        reference_answer=reference_answer,
        evidences=evidences
        if evidences is not None
        else [
            BenchmarkEvidence(
                document_name="Annual Report.pdf",
                page_number=7,
                text="Revenue increased by twenty percent.",
            )
        ],
        metadata={},
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (r"C:\filings\Annual.Report.PDF", "annual.report"),
        ("/archive/ANNUAL.REPORT.pdf?download=1#page=7", "annual.report"),
        ("  'Mixed Case.DocX'  ", "mixed case"),
        (None, ""),
    ],
)
def test_normalize_document_name_uses_basename_stem_and_casefold(
    value: object,
    expected: str,
) -> None:
    assert normalize_document_name(value) == expected


def test_document_hit_rate_respects_cutoff() -> None:
    metrics = BenchmarkMetrics(["document_hit_rate@1", "document_hit_rate@2"])
    retrieved = [
        {"metadata": {"source_path": "/filings/other.pdf"}},
        {"metadata": {"source_path": "/FILINGS/annual report.PDF"}},
    ]

    assert metrics.evaluate_case(_case(), retrieved, answer=None) == {
        "document_hit_rate@1": 0.0,
        "document_hit_rate@2": 1.0,
    }


def test_document_mrr_and_mrr_alias_report_reciprocal_rank() -> None:
    metrics = BenchmarkMetrics(["document_mrr@3", "mrr@3"])
    retrieved = [
        {"metadata": {"filename": "other.pdf"}},
        {"metadata": {"filename": "ANNUAL REPORT.pdf"}},
        {"metadata": {"filename": "later.pdf"}},
    ]

    assert metrics.evaluate_case(_case(), retrieved, answer=None) == {
        "document_mrr@3": 0.5,
        "mrr@3": 0.5,
    }


def test_document_hit_rate_scores_fraction_of_reference_evidence_documents() -> None:
    metrics = BenchmarkMetrics(["document_hit_rate@5", "document_mrr@5"])
    case = _case(
        evidences=[
            BenchmarkEvidence(
                document_name="First Filing.pdf",
                page_number=7,
                text="First supporting fact.",
            ),
            BenchmarkEvidence(
                document_name="Second Filing.pdf",
                page_number=9,
                text="Second supporting fact.",
            ),
        ]
    )
    retrieved = [
        {"metadata": {"filename": "other.pdf"}},
        {"metadata": {"filename": "first filing.pdf"}},
    ]

    assert metrics.evaluate_case(case, retrieved, answer=None) == {
        "document_hit_rate@5": pytest.approx(0.5),
        "document_mrr@5": 0.5,
    }


def test_page_hit_rate_requires_document_match_and_accepts_page_range() -> None:
    metrics = BenchmarkMetrics(["page_hit_rate@3"])
    wrong_document = {
        "metadata": {
            "filename": "wrong-document.pdf",
            "page_start": 7,
            "page_end": 7,
        }
    }
    matching_document = {
        "metadata": {
            "filename": "annual report.pdf",
            "page_start": 5,
            "page_end": 8,
        }
    }

    assert metrics.evaluate_case(_case(), [wrong_document], answer=None) == {"page_hit_rate@3": 0.0}
    assert metrics.evaluate_case(
        _case(),
        [
            wrong_document,
            matching_document,
        ],
        answer=None,
    ) == {"page_hit_rate@3": 1.0}


def test_page_hit_rate_falls_back_to_evidence_overlap_without_page_metadata() -> None:
    metrics = BenchmarkMetrics(["page_hit_rate@1"])
    retrieved = [
        {
            "text": "The filing states that revenue increased by twenty percent in 2024.",
            "metadata": {"filename": "Annual Report.pdf"},
        }
    ]

    assert metrics.evaluate_case(_case(), retrieved, answer=None) == {"page_hit_rate@1": 1.0}


def test_page_hit_rate_scores_fraction_of_reference_evidence_pages() -> None:
    metrics = BenchmarkMetrics(["page_hit_rate@5"])
    case = _case(
        evidences=[
            BenchmarkEvidence(
                document_name="Annual Report.pdf",
                page_number=7,
                text="Revenue increased.",
            ),
            BenchmarkEvidence(
                document_name="Annual Report.pdf",
                page_number=9,
                text="Operating income declined.",
            ),
        ]
    )
    retrieved = [
        {
            "metadata": {
                "filename": "annual report.pdf",
                "page_start": 7,
                "page_end": 8,
            }
        },
        {
            "metadata": {
                "filename": "other.pdf",
                "page_start": 9,
                "page_end": 9,
            }
        },
    ]

    assert metrics.evaluate_case(case, retrieved, answer=None) == {
        "page_hit_rate@5": pytest.approx(0.5)
    }


def test_evidence_hit_rate_uses_coverage_and_mrr_uses_earliest_match() -> None:
    evidences = [
        BenchmarkEvidence(
            document_name="Annual Report.pdf",
            page_number=index,
            text=f"Evidence {index}",
        )
        for index in range(1, 4)
    ]
    result = BenchmarkMetrics(
        [
            "evidence_hit_rate@3",
            "evidence_hit_rate@5",
            "evidence_mrr@3",
            "evidence_mrr@5",
        ]
    ).evaluate_case(
        _case(evidences=evidences),
        retrieved=[{"text": f"chunk {index}"} for index in range(1, 6)],
        answer=None,
        evidence_ranks=(4, 2, None),
    )

    assert result == {
        "evidence_hit_rate@3": pytest.approx(1.0 / 3.0),
        "evidence_hit_rate@5": pytest.approx(2.0 / 3.0),
        "evidence_mrr@3": 0.5,
        "evidence_mrr@5": 0.5,
    }


def test_answer_exact_match_normalizes_case_punctuation_and_whitespace() -> None:
    result = BenchmarkMetrics(["answer_exact_match"]).evaluate_case(
        _case(reference_answer="Net income: $42 million."),
        retrieved=[],
        answer="  NET INCOME $42 MILLION  ",
    )

    assert result == {"answer_exact_match": 1.0}


def test_answer_token_f1_uses_token_multiplicity() -> None:
    result = BenchmarkMetrics(["answer_token_f1"]).evaluate_case(
        _case(reference_answer="alpha alpha beta"),
        retrieved=[],
        answer="alpha beta beta",
    )

    assert result["answer_token_f1"] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(
    ("reference", "answer", "expected"),
    [
        ("Revenue was $1,200.", "Revenue was USD 1,200.", 1.0),
        ("The margin was 25%.", "The margin was 0.25.", 1.0),
        ("The loss was ($1,200).", "The loss was -1200.", 1.0),
        ("Revenue was $1,200.", "Revenue was $1,250.", 0.0),
    ],
)
def test_numeric_accuracy_handles_financial_forms_and_mismatches(
    reference: str,
    answer: str,
    expected: float,
) -> None:
    result = BenchmarkMetrics(["numeric_accuracy"]).evaluate_case(
        _case(reference_answer=reference),
        retrieved=[],
        answer=answer,
    )

    assert result == {"numeric_accuracy": expected}


def test_empty_retrieval_scores_all_retrieval_metrics_zero() -> None:
    metrics = BenchmarkMetrics(
        [
            "document_hit_rate@5",
            "document_mrr@5",
            "page_hit_rate@5",
            "evidence_hit_rate@5",
            "evidence_mrr@5",
        ]
    )

    assert metrics.evaluate_case(_case(), retrieved=[], answer=None) == {
        "document_hit_rate@5": 0.0,
        "document_mrr@5": 0.0,
        "page_hit_rate@5": 0.0,
        "evidence_hit_rate@5": 0.0,
        "evidence_mrr@5": 0.0,
    }


def test_aggregate_averages_each_metric_only_where_present() -> None:
    case_results = [
        {"document_hit_rate@5": 1.0, "numeric_accuracy": 1.0},
        {"document_hit_rate@5": 0.0},
        {"document_hit_rate@5": 1.0, "numeric_accuracy": 0.0},
        {},
    ]
    expected = {
        "document_hit_rate@5": pytest.approx(2.0 / 3.0),
        "numeric_accuracy": 0.5,
    }

    assert aggregate(case_results) == expected
    assert BenchmarkMetrics.aggregate(case_results) == expected
    assert aggregate([]) == {}
