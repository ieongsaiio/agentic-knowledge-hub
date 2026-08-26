"""Unit tests for the LLM query-rewriting provider."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.settings import LLMSettings
from src.libs.llm import ChatResponse
from src.libs.query_rewriter import LLMQueryRewriter, QueryRewriterFactory


class FakeLLM:
    def __init__(self, content: str | Exception) -> None:
        self.content = content
        self.messages = None

    def chat(self, messages: list[Any], **kwargs: Any) -> ChatResponse:
        self.messages = messages
        if isinstance(self.content, Exception):
            raise self.content
        return ChatResponse(content=self.content, model="fake")


@dataclass(frozen=True)
class FakeSettings:
    llm: LLMSettings
    retrieval: Any


def make_settings(
    tmp_path: Path,
    *,
    fail_on_error: bool = False,
    max_queries: int = 2,
) -> FakeSettings:
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("Return a retrieval plan as JSON.", encoding="utf-8")
    config = SimpleNamespace(
        prompt_path=str(prompt),
        max_queries=max_queries,
        fail_on_error=fail_on_error,
        provider="llm",
        llm={},
    )
    return FakeSettings(
        llm=LLMSettings(
            provider="openai",
            model="fake",
            temperature=0.0,
            max_tokens=512,
            api_key="test",
        ),
        retrieval=SimpleNamespace(query_rewriter=config),
    )


def test_parses_structured_plan_and_limits_unique_rewrites(tmp_path: Path) -> None:
    payload = {
        "question_types": ["financial comparison"],
        "slots": [
            {
                "slot_type": "period",
                "value": "2023",
                "role": "reporting year",
                "must_preserve": True,
            }
        ],
        "source_hints": [
            {
                "document_type": "annual report",
                "section": "segment results",
                "content_form": "table",
                "confidence": "high",
            }
        ],
        "rewrite_decision": {
            "rewrite_needed": True,
            "strategy": "expand financial terminology",
            "reason": "The source may use formal labels.",
        },
        "rewritten_queries": [
            {"query_id": "q1", "purpose": "metric", "query": "2023 segment revenue"},
            {"query_id": "q2", "purpose": "duplicate", "query": "2023 segment revenue"},
            {"query_id": "q3", "purpose": "table", "query": "segment results 2023 table"},
            {"query_id": "q4", "purpose": "extra", "query": "ignored by limit"},
        ],
    }
    llm = FakeLLM(f"```json\n{json.dumps(payload)}\n```")
    rewriter = LLMQueryRewriter(make_settings(tmp_path), llm=llm)

    plan = rewriter.rewrite("Compare segment revenue in 2023", context="Company A")

    assert plan.rewrite_needed is True
    assert plan.question_types == ["financial comparison"]
    assert plan.slots[0].value == "2023"
    assert plan.source_hints[0].content_form == "table"
    assert plan.queries == ["2023 segment revenue", "segment results 2023 table"]
    assert "<context>\nCompany A\n</context>" in llm.messages[1].content


def test_false_decision_discards_generated_queries(tmp_path: Path) -> None:
    payload = {
        "question_types": ["lookup"],
        "slots": [],
        "source_hints": [],
        "rewrite_decision": {
            "rewrite_needed": False,
            "strategy": "direct retrieval",
            "reason": "Already specific.",
        },
        "rewritten_queries": [
            {"query_id": "q1", "purpose": "unused", "query": "other query"}
        ],
    }
    rewriter = LLMQueryRewriter(
        make_settings(tmp_path),
        llm=FakeLLM(json.dumps(payload)),
    )

    plan = rewriter.rewrite("What is net income in 2023?")

    assert plan.rewrite_needed is False
    assert plan.rewritten_queries == []


def test_invalid_response_falls_back_to_original_query(tmp_path: Path) -> None:
    rewriter = LLMQueryRewriter(
        make_settings(tmp_path),
        llm=FakeLLM("not-json"),
    )

    plan = rewriter.rewrite("original")

    assert plan.original_query == "original"
    assert plan.rewrite_needed is False
    assert "rewriter_fallback" in plan.reason


def test_fail_on_error_raises(tmp_path: Path) -> None:
    rewriter = LLMQueryRewriter(
        make_settings(tmp_path, fail_on_error=True),
        llm=FakeLLM(RuntimeError("provider unavailable")),
    )

    with pytest.raises(RuntimeError, match="Query rewriting failed"):
        rewriter.rewrite("original")


def test_factory_lists_llm_provider() -> None:
    assert "llm" in QueryRewriterFactory.list_providers()

