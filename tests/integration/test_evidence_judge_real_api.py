"""Real API integration test for the benchmark evidence judge."""

from __future__ import annotations

import pytest

from src.core.settings import load_settings
from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.observability.evaluation.evidence_judge import LLMEvidenceJudge


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return value.strip().upper() in {
        "",
        "YOUR_API_KEY_HERE",
        "YOUR_OPENAI_API_KEY",
    }


@pytest.mark.integration
@pytest.mark.llm
def test_evidence_judge_real_openai_api_from_settings() -> None:
    """Use the configured OpenAI-compatible API and parse its judged rank."""
    settings = load_settings("config/settings.yaml")
    if settings.llm.provider != "openai":
        pytest.skip("This test requires llm.provider to be 'openai'")
    if _is_placeholder(settings.llm.api_key):
        pytest.skip("llm.api_key is missing or still uses a placeholder")

    case = BenchmarkCase(
        case_id="real-api-evidence-judge",
        query="Which domestic product category performed best in Q2 FY2024?",
        reference_answer=(
            "Entertainment performed best, with comparable sales growth of 9.0%."
        ),
        evidences=[
            BenchmarkEvidence(
                document_name="example_report",
                page_number=7,
                text=(
                    "Computing -6.4%; Consumer Electronics -5.7%; "
                    "Appliances -16.1%; Entertainment +9.0%; Services +7.6%."
                ),
            )
        ],
        metadata={},
    )
    chunks = [
        {"text": "Domestic comparable sales varied across categories."},
        {
            "text": (
                "Domestic comparable sales: Entertainment increased 9.0%, "
                "primarily driven by gaming."
            )
        },
        {"text": "International revenue increased by 2.0% in the quarter."},
    ]

    judgement = LLMEvidenceJudge(settings).judge(
        case,
        chunks,
        eligible_ranks=((1, 2, 3),),
    )

    print(
        {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "match_ranks": judgement.match_ranks,
            "first_matching_rank": judgement.first_matching_rank,
            "question_requirements": (
                judgement.matches[0].question_requirements
            ),
            "reason": judgement.matches[0].reason,
        }
    )
    assert judgement.match_ranks == (2,)
    assert judgement.first_matching_rank == 2
    assert judgement.matches[0].reason.strip()

