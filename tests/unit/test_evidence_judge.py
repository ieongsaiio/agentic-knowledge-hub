"""Unit tests for the LLM-backed semantic evidence judge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.libs.llm import BaseLLM, ChatResponse, Message
from src.observability.evaluation.evidence_judge import (
    AtomicFactMatch,
    EvidenceJudgement,
    EvidenceMatch,
    LLMEvidenceJudge,
)


class FakeLLM(BaseLLM):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[Message]] = []
        self.call_kwargs: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        return ChatResponse(content=self.content, model="fake-judge")


class SequenceFakeLLM(BaseLLM):
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace, kwargs
        self.calls.append(messages)
        return ChatResponse(
            content=self.contents[len(self.calls) - 1],
            model="fake-judge",
        )


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
        '"question_requirements":"Compare segment values.",'
        '"reason":"Equivalent table row.","first_matching_rank":2}]}\n```'
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
    assert result.matches[0].question_requirements == "Compare segment values."
    assert result.matches[0].reason == "Equivalent table row."
    assert len(llm.calls) == 1
    user_prompt = llm.calls[0][1].content
    assert "<reference_evidences>" in user_prompt
    assert '<evidence index="1">' in user_prompt
    assert "<candidate_chunks>" in user_prompt
    assert '<chunk rank="1">' in user_prompt
    assert '<chunk rank="2">' in user_prompt
    assert "<![CDATA[" in user_prompt
    assert "Data Center 6,043 3,694; Gaming 6,805 5,607." in user_prompt
    assert "<question>" in user_prompt
    assert "Which segment grew the most?" in user_prompt
    assert "<reference_answer>" in user_prompt
    assert "Data Center." in user_prompt
    assert "AMD_2022_10K.pdf" not in user_prompt
    assert "Page: 48" not in user_prompt


def test_judge_allows_output_budget_override() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,'
        '"first_matching_rank":null,"reason":"No match."}]}'
    )
    judge = LLMEvidenceJudge(
        settings=None,
        llm=llm,
        max_output_tokens=4096,
    )

    judge.judge(_case(), [{"text": "Unrelated content."}])

    assert llm.call_kwargs[0]["max_tokens"] == 4096


def test_judge_only_receives_eligible_chunks_with_original_ranks() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,'
        '"question_requirements":"Find segment growth.",'
        '"reason":"Eligible rank 2 matches.","first_matching_rank":2}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [
            {"text": "Wrong document or page."},
            {"text": "Data Center 6,043 3,694; Gaming 6,805 5,607."},
            {"text": "Another ineligible chunk."},
        ],
        eligible_ranks=((2,),),
    )

    assert result.match_ranks == (2,)
    user_prompt = llm.calls[0][1].content
    assert '<chunk rank="2">' in user_prompt
    assert '<chunk rank="1">' not in user_prompt
    assert '<chunk rank="3">' not in user_prompt
    assert "Wrong document or page." not in user_prompt
    assert "Another ineligible chunk." not in user_prompt


def test_question_guides_rank_selection_for_multi_table_evidence() -> None:
    case = BenchmarkCase(
        case_id="case-full-year",
        query="Compare U.S. and international sales growth for full-year 2022.",
        reference_answer="U.S. 3.0%; International -0.6%.",
        evidences=[
            BenchmarkEvidence(
                document_name="report.pdf",
                page_number=4,
                text="Q4: U.S. 2.9%, International -11.5%. "
                "Full Year: U.S. 3.0%, International -0.6%.",
            )
        ],
        metadata={},
    )
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,'
        '"question_requirements":"Full-year 2022 U.S. and international growth.",'
        '"reason":"Rank 2 is Q4 only; rank 3 contains the requested full-year values.",'
        '"first_matching_rank":3}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        case,
        [
            {"text": "Unrelated."},
            {"text": "Q4: U.S. 2.9%, International -11.5%."},
            {"text": "Full Year: U.S. 3.0%, International -0.6%."},
        ],
        eligible_ranks=((2, 3),),
    )

    assert result.match_ranks == (3,)
    prompt = llm.calls[0][1].content
    assert case.query in prompt
    assert "<reference_answer>" in prompt
    assert case.reference_answer in prompt


def test_empty_retrieval_returns_no_hit_without_calling_llm() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,"first_matching_rank":1,"reason":"unused"}]}'
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
        reference_answer="Revenue was $100.",
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
    assert case.query in user_prompt
    assert "<reference_answer>" in user_prompt
    assert case.reference_answer in user_prompt


def test_judge_batches_ten_chunks_into_four_calls_with_global_ranks() -> None:
    llm = SequenceFakeLLM(
        [
            '{"evidence_matches":[{"evidence_index":1,'
            '"question_requirements":"Compare segment values.",'
            '"reason":"No match in ranks 1-3.","first_matching_rank":null}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"question_requirements":"Compare segment values.",'
            '"reason":"No match in ranks 4-6.","first_matching_rank":null}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"question_requirements":"Compare segment values.",'
            '"reason":"Rank 8 contains the values.","first_matching_rank":8}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"question_requirements":"Compare segment values.",'
            '"reason":"No match in rank 10.","first_matching_rank":null}]}',
        ]
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [{"text": f"Chunk {rank}"} for rank in range(1, 11)],
    )

    assert result.match_ranks == (8,)
    assert len(llm.calls) == 4
    first_prompt = llm.calls[0][1].content
    third_prompt = llm.calls[2][1].content
    fourth_prompt = llm.calls[3][1].content
    assert first_prompt.count("<chunk rank=") == 3
    assert third_prompt.count("<chunk rank=") == 3
    assert fourth_prompt.count("<chunk rank=") == 1
    assert '<chunk rank="1">' in first_prompt
    assert '<chunk rank="3">' in first_prompt
    assert '<chunk rank="4">' not in first_prompt
    assert '<chunk rank="7">' in third_prompt
    assert '<chunk rank="9">' in third_prompt
    assert '<chunk rank="10">' in fourth_prompt
    assert '<chunk rank="1">' not in third_prompt


def test_judge_keeps_earliest_matching_rank_across_batches() -> None:
    llm = SequenceFakeLLM(
        [
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"Rank 3 matches.","first_matching_rank":3}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"No match.","first_matching_rank":null}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"Rank 8 also matches.","first_matching_rank":8}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"No match.","first_matching_rank":null}]}',
        ]
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [{"text": f"Chunk {rank}"} for rank in range(1, 11)],
    )

    assert result.match_ranks == (3,)
    assert "Rank 3 matches." in result.matches[0].reason
    assert "Rank 8 also matches." in result.matches[0].reason


def test_judge_sends_sparse_eligible_ranks_in_one_call_when_at_most_three() -> None:
    llm = SequenceFakeLLM(
        [
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"Rank 7 matches.","first_matching_rank":7}]}',
        ]
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [{"text": f"Chunk {rank}"} for rank in range(1, 11)],
        eligible_ranks=((2, 7),),
    )

    assert result.match_ranks == (7,)
    assert len(llm.calls) == 1
    assert '<chunk rank="2">' in llm.calls[0][1].content
    assert '<chunk rank="7">' in llm.calls[0][1].content
    assert llm.calls[0][1].content.count("<chunk rank=") == 2


def test_judge_batches_more_than_three_sparse_eligible_ranks() -> None:
    llm = SequenceFakeLLM(
        [
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"No match in first batch.","first_matching_rank":null}]}',
            '{"evidence_matches":[{"evidence_index":1,'
            '"reason":"Rank 10 matches.","first_matching_rank":10}]}',
        ]
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [{"text": f"Chunk {rank}"} for rank in range(1, 11)],
        eligible_ranks=((1, 2, 4, 6, 8, 10),),
    )

    assert result.match_ranks == (10,)
    assert len(llm.calls) == 2
    assert llm.calls[0][1].content.count("<chunk rank=") == 3
    assert '<chunk rank="4">' in llm.calls[0][1].content
    assert '<chunk rank="6">' not in llm.calls[0][1].content
    assert '<chunk rank="10">' not in llm.calls[0][1].content
    assert llm.calls[1][1].content.count("<chunk rank=") == 3
    assert '<chunk rank="10">' in llm.calls[1][1].content


def test_judge_merges_different_evidence_hits_across_batches() -> None:
    case = BenchmarkCase(
        case_id="case-batched-evidence",
        query="Compare revenue and operating income.",
        reference_answer="not sent",
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
    llm = SequenceFakeLLM(
        [
            '{"evidence_matches":['
            '{"evidence_index":1,"reason":"Rank 2 matches.",'
            '"first_matching_rank":2},'
            '{"evidence_index":2,"reason":"No match in first batch.",'
            '"first_matching_rank":null}]}',
            '{"evidence_matches":['
            '{"evidence_index":1,"reason":"No match in second batch.",'
            '"first_matching_rank":null},'
            '{"evidence_index":2,"reason":"No match in second batch.",'
            '"first_matching_rank":null}]}',
            '{"evidence_matches":['
            '{"evidence_index":1,"reason":"No match in second batch.",'
            '"first_matching_rank":null},'
            '{"evidence_index":2,"reason":"Rank 7 matches.",'
            '"first_matching_rank":7}]}',
            '{"evidence_matches":['
            '{"evidence_index":1,"reason":"No match in final batch.",'
            '"first_matching_rank":null},'
            '{"evidence_index":2,"reason":"No match in final batch.",'
            '"first_matching_rank":null}]}',
        ]
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        case,
        [{"text": f"Chunk {rank}"} for rank in range(1, 11)],
    )

    assert result.match_ranks == (2, 7)
    assert len(llm.calls) == 4
    assert llm.calls[0][1].content.count("<chunk rank=") == 3
    assert llm.calls[1][1].content.count("<chunk rank=") == 3
    assert "<eligible_chunk_ranks>1,2,3</eligible_chunk_ranks>" in (llm.calls[0][1].content)


def test_default_prompt_is_short_and_atomic_fact_focused() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,'
        '"question_requirements":"Identify the best domestic category.",'
        '"reason":"Entertainment at +9.0% directly supports the answer.",'
        '"first_matching_rank":1}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    judge.judge(
        _case(),
        [{"text": "Domestic Entertainment increased 9.0%, driven by gaming."}],
    )

    system_prompt = llm.calls[0][0].content
    assert "For every <fact>" in system_prompt
    assert "supporting_ranks" in system_prompt
    assert "One-shot example:" not in system_prompt
    assert len(system_prompt) < 700
    assert '<fact id="e1_f1">' in llm.calls[0][1].content


def test_atomic_fact_response_returns_support_and_evidence_completion_rank() -> None:
    llm = FakeLLM(
        '{"fact_matches":[{"evidence_index":1,"fact_id":"e1_f1",'
        '"supported":true,"supporting_ranks":[2],"relevant_ranks":[1,2]}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [{"text": "Unrelated."}, {"text": "Data Center 6,043 3,694."}],
    )

    assert result.match_ranks == (2,)
    assert result.fact_matches[0].fact_id == "e1_f1"
    assert result.fact_matches[0].supported is True
    assert result.fact_matches[0].supporting_ranks == (2,)
    assert result.fact_matches[0].relevant_ranks == (1, 2)
    assert result.context_relevant_ranks == (1, 2)
    assert result.context_recall == 1.0
    assert result.evidence_hit_rate == 1.0


def test_atomic_fact_response_defaults_relevant_ranks_to_supporting_ranks() -> None:
    llm = FakeLLM(
        '{"fact_matches":[{"evidence_index":1,"fact_id":"e1_f1",'
        '"supported":true,"supporting_ranks":[2]}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(
        _case(),
        [{"text": "Unrelated."}, {"text": "Data Center 6,043 3,694."}],
    )

    assert result.fact_matches[0].relevant_ranks == (2,)
    assert result.context_relevant_ranks == (2,)


def test_partial_fact_can_have_relevant_context_without_complete_support() -> None:
    llm = FakeLLM(
        '{"fact_matches":[{"evidence_index":1,"fact_id":"e1_f1",'
        '"supported":false,"supporting_ranks":[],"relevant_ranks":[1]}]}'
    )
    judge = LLMEvidenceJudge(settings=None, llm=llm)

    result = judge.judge(_case(), [{"text": "Partial but correct segment data."}])

    assert result.fact_matches[0].supported is False
    assert result.fact_matches[0].relevant_ranks == (1,)
    assert result.context_relevant_ranks == (1,)
    assert result.context_recall == 0.0


def test_partial_fact_support_has_recall_but_not_complete_evidence_hit() -> None:
    result = EvidenceJudgement(
        matches=(
            EvidenceMatch(
                evidence_index=1,
                first_matching_rank=None,
                fact_matches=(
                    AtomicFactMatch(1, "e1_f1", True, (1,)),
                    AtomicFactMatch(1, "e1_f2", False, ()),
                ),
            ),
        )
    )

    assert result.context_recall == 0.5
    assert result.evidence_hit_rate == 0.0


def test_prompt_wraps_cdata_terminators_safely() -> None:
    llm = FakeLLM(
        '{"evidence_matches":[{"evidence_index":1,"first_matching_rank":1,"reason":"matches"}]}'
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
