"""Tests for the retrieval-oriented table-summary judge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.libs.llm import BaseLLM, ChatResponse, Message
from src.observability.evaluation.table_summary_judge import (
    LLMTableSummaryJudge,
    TableSummaryJudgement,
)


class _FakeLLM(BaseLLM):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[Message]] = []
        self.kwargs: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return ChatResponse(content=self.content, model="fake-judge")


def _payload(**overrides: Any) -> str:
    payload = {
        "faithfulness": 4,
        "key_information_coverage": 4,
        "retrieval_utility": 4,
        "trend_relationship_quality": 4,
        "conciseness_clarity": 4,
        "unsupported_claims": [],
        "missing_key_information": [],
        "reason": "Fully supported and retrieval useful.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_judge_formats_source_and_returns_deterministic_score(tmp_path: Path) -> None:
    prompt_path = tmp_path / "judge.txt"
    prompt_path.write_text("Use the rubric.", encoding="utf-8")
    llm = _FakeLLM(f"```json\n{_payload()}\n```")
    judge = LLMTableSummaryJudge(
        settings=None,
        llm=llm,
        prompt_path=prompt_path,
    )

    result = judge.judge(
        "<table><tr><td>Revenue</td><td>120</td></tr></table>",
        "Table summary: Revenue was 120.",
        document_name="ACME_2023",
        page_range="10-11",
        section_path="Financial statements > Revenue",
        table_title="Annual results",
        previous_context="The following table presents results.",
        footnotes=["Amounts are in USD million."],
        next_context="Additional details follow.",
        table_unit_count=2,
    )

    assert result.overall_score == 100
    assert result.quality_level == "excellent"
    user_prompt = llm.calls[0][1].content
    assert "<table_unit_count>2</table_unit_count>" in user_prompt
    assert "<section_path><![CDATA[Financial statements > Revenue]]>" in user_prompt
    assert "<table_title><![CDATA[Annual results]]>" in user_prompt
    assert "<previous_context><![CDATA[The following table" in user_prompt
    assert "<footnote><![CDATA[Amounts are in USD million.]]>" in user_prompt
    assert "<next_context><![CDATA[Additional details follow.]]>" in user_prompt
    assert "<table_source><![CDATA[<table>" in user_prompt
    assert "<candidate_summary><![CDATA[Table summary:" in user_prompt
    assert llm.kwargs[0] == {"temperature": 0.0, "max_tokens": 1200}


def test_judge_with_response_retains_provider_usage(tmp_path: Path) -> None:
    prompt_path = tmp_path / "judge.txt"
    prompt_path.write_text("Use the rubric.", encoding="utf-8")
    response = ChatResponse(
        content=_payload(),
        model="fake-judge",
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    llm = _FakeLLM(response.content)
    judge = LLMTableSummaryJudge(
        settings=None,
        llm=llm,
        prompt_path=prompt_path,
    )
    llm.chat = lambda *args, **kwargs: response  # type: ignore[method-assign]

    judgement, raw_response = judge.judge_with_response("table", "summary")

    assert judgement.overall_score == 100
    assert raw_response is response
    assert raw_response.usage["prompt_tokens"] == 100


def test_faithfulness_caps_an_otherwise_high_score() -> None:
    judgement = TableSummaryJudgement(
        faithfulness=2,
        key_information_coverage=4,
        retrieval_utility=4,
        trend_relationship_quality=4,
        conciseness_clarity=4,
    )

    assert judgement.raw_score == 80
    assert judgement.overall_score == 59
    assert judgement.quality_level == "poor"


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((4, 4, 4, 4, 4), "excellent"),
        ((4, 3, 3, 3, 3), "good"),
        ((3, 2, 2, 2, 2), "usable"),
        ((2, 4, 4, 4, 4), "poor"),
        ((1, 4, 4, 4, 4), "reject"),
    ],
)
def test_quality_level_boundaries(
    scores: tuple[int, int, int, int, int],
    expected: str,
) -> None:
    judgement = TableSummaryJudgement(
        faithfulness=scores[0],
        key_information_coverage=scores[1],
        retrieval_utility=scores[2],
        trend_relationship_quality=scores[3],
        conciseness_clarity=scores[4],
    )

    assert judgement.quality_level == expected


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        _payload(faithfulness=5),
        _payload(retrieval_utility=2.5),
        _payload(unsupported_claims="none"),
    ],
)
def test_invalid_judge_response_is_rejected(tmp_path: Path, content: str) -> None:
    prompt_path = tmp_path / "judge.txt"
    prompt_path.write_text("Use the rubric.", encoding="utf-8")
    judge = LLMTableSummaryJudge(
        settings=None,
        llm=_FakeLLM(content),
        prompt_path=prompt_path,
    )

    with pytest.raises(ValueError):
        judge.judge("table", "summary")


def test_default_prompt_contains_all_score_anchors() -> None:
    prompt = Path("config/prompts/table_summary_judge.txt").read_text(encoding="utf-8")

    for dimension in (
        "faithfulness",
        "key_information_coverage",
        "retrieval_utility",
        "trend_relationship_quality",
        "conciseness_clarity",
    ):
        assert dimension in prompt
    for level in ("Excellent", "Good", "Usable", "Poor", "Reject"):
        assert f"- {level} profile" in prompt
    assert "Do not return an aggregate score" in prompt
