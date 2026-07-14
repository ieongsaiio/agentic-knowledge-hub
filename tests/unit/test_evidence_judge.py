"""Unit tests for the LLM-backed semantic evidence judge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.libs.llm import BaseLLM, ChatResponse, Message
from src.observability.evaluation.evidence_judge import LLMEvidenceJudge


class FakeLLM(BaseLLM):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace, kwargs
        self.calls.append(messages)
        return ChatResponse(content=self.content, model="fake-judge")


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-1",
        query="Which segment grew the most?",
        reference_answer="Data Center.",
        evidences=[
            BenchmarkEvidence(
                document_name="AMD_2022_10K",
                page_number=48,
                text="Data Center 6,043 3,694; Gaming 6,805 5,607.",
            )
        ],
        metadata={},
    )


def test_judge_parses_fenced_json_and_formats_ranked_contexts(tmp_path: Path) -> None:
    llm = FakeLLM(
        '```json\n{"evidence_matches":[{"evidence_index":1,'
        '"first_matching_rank":2,"reason":"Equivalent table row."}]}\n```'
    )
    prompt = tmp_path / "judge.txt"
    prompt.write_text("Return strict JSON.", encoding="utf-8")
    judge = LLMEvidenceJudge(
        settings=None,
        llm=llm,
        prompt_path=prompt,
    )

    result = judge.judge(
        _case(),
        [
            {
                "text": "| Segment | 2022 | 2021 |",
                "metadata": {"source_path": "AMD_2022_10K.pdf", "page_num": 48},
            },
            {"text": "| Data Center | 6,043 | 3,694 |"},
        ],
    )

    assert result.match_ranks == (2,)
    assert result.first_matching_rank == 2
    assert result.matches[0].reason == "Equivalent table row."
    assert len(llm.calls) == 1
    user_prompt = llm.calls[0][1].content
    assert "<reference_evidences>" in user_prompt
    assert '<evidence index="1">' in user_prompt
    assert "<retrieved_chunks>" in user_prompt
    assert '<chunk rank="1">' in user_prompt
    assert '<chunk rank="2">' in user_prompt
    assert "<![CDATA[" in user_prompt
    assert "Data Center 6,043 3,694; Gaming 6,805 5,607." in user_prompt
    assert "Which segment grew the most?" not in user_prompt
    assert "Data Center." not in user_prompt
    assert "AMD_2022_10K.pdf" not in user_prompt
    assert "Page: 48" not in user_prompt


def test_empty_retrieval_returns_no_hit_without_calling_llm() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,'
        '"first_matching_rank":1,"reason":"unused"}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(_case(), [])

    assert result.match_ranks == (None,)
    assert result.first_matching_rank is None
    assert llm.calls == []


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"evidence_matches":[]}',
        '{"evidence_matches":[{"evidence_index":0,"first_matching_rank":1}]}',
        '{"evidence_matches":[{"evidence_index":1,"first_matching_rank":3}]}',
        '{"evidence_matches":[{"evidence_index":1,"first_matching_rank":true}]}',
    ],
)
def test_invalid_judge_response_is_rejected(content: str) -> None:
    judge = LLMEvidenceJudge(settings=None, llm=FakeLLM(content))

    with pytest.raises(ValueError):
        judge.judge(_case(), [{"text": "first"}, {"text": "second"}])


def test_multiple_evidences_are_matched_in_one_llm_call() -> None:
    case = BenchmarkCase(
        case_id="case-multiple",
        query="This question must not be sent.",
        reference_answer="This answer must not be sent.",
        evidences=[
            BenchmarkEvidence(
                document_name="report.pdf",
                page_number=2,
                text="Revenue was $100.",
            ),
            BenchmarkEvidence(
                document_name="report.pdf",
                page_number=3,
                text="Operating income was $20.",
            ),
        ],
        metadata={},
    )
    llm = FakeLLM(
        '{"evidence_matches":['
        '{"evidence_index":1,"first_matching_rank":2,"reason":"Revenue matches."},'
        '{"evidence_index":2,"first_matching_rank":null,"reason":"Not found."}'
        "]}"
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        case,
        [{"text": "Unrelated."}, {"text": "Revenue was $100."}],
    )

    assert result.match_ranks == (2, None)
    assert result.first_matching_rank == 2
    assert len(llm.calls) == 1
    user_prompt = llm.calls[0][1].content
    assert '<evidence index="1">' in user_prompt
    assert '<evidence index="2">' in user_prompt
    assert case.query not in user_prompt
    assert case.reference_answer not in user_prompt


def test_prompt_wraps_cdata_terminators_safely() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,'
        '"first_matching_rank":1,"reason":"matches"}]}'
    )
    case = BenchmarkCase(
        case_id="case-cdata",
        query="not sent",
        reference_answer="not sent",
        evidences=[
            BenchmarkEvidence(
                document_name="report.pdf",
                page_number=1,
                text="Revenue marker ]]> safely wrapped.",
            )
        ],
        metadata={},
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    judge.judge(case, [{"text": "Revenue marker ]]> safely wrapped."}])

    user_prompt = llm.calls[0][1].content
    assert "]]]]><![CDATA[>" in user_prompt
