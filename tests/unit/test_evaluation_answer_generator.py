"""Unit tests for provider-driven evaluation answer generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.libs.llm import BaseLLM, ChatResponse, Message
from src.observability.evaluation.answer_generator import (
    EvaluationAnswerGenerator,
    GeneratedAnswer,
)


class FakeLLM(BaseLLM):
    """Record chat input and return a deterministic provider response."""

    def __init__(
        self,
        content: str = "Grounded answer [1].",
        model: str = "fake-model",
        usage: dict[str, int] | None = None,
    ) -> None:
        self.response = ChatResponse(
            content=content,
            model=model,
            usage=usage
            or {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        )
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        trace: Any = None,
        **kwargs: Any,
    ) -> ChatResponse:
        del trace, kwargs
        self.calls.append(messages)
        return self.response


def test_injected_provider_avoids_factory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    def fail_if_called(settings: Any) -> BaseLLM:
        del settings
        raise AssertionError("LLMFactory.create must not run for an injected provider")

    monkeypatch.setattr(
        "src.observability.evaluation.answer_generator.LLMFactory.create",
        fail_if_called,
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Use only supplied evidence.", encoding="utf-8")

    fake = FakeLLM()
    generator = EvaluationAnswerGenerator(
        settings=object(),
        llm=fake,
        prompt_path=prompt_path,
    )

    assert generator.llm is fake


def test_generate_formats_messages_and_returns_provider_metadata(tmp_path: Any) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("System evaluation prompt.", encoding="utf-8")
    usage = {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29}
    fake = FakeLLM(
        content="Revenue increased [1], while costs fell [2].",
        model="fake-evaluator-v1",
        usage=usage,
    )
    generator = EvaluationAnswerGenerator(object(), llm=fake, prompt_path=prompt_path)
    contexts = [
        {
            "text": "Revenue increased to USD 12 million.",
            "metadata": {"source_path": "reports/annual.pdf", "page": 7},
        },
        SimpleNamespace(
            page_content="Operating costs fell by 5 percent.",
            metadata={
                "document_name": "cost review.pdf",
                "page_start": 10,
                "page_end": 12,
            },
        ),
    ]

    result = generator.generate("  What changed this year?  ", contexts)

    assert result == GeneratedAnswer(
        content="Revenue increased [1], while costs fell [2].",
        model="fake-evaluator-v1",
        usage=usage,
    )
    assert result.usage is not usage
    assert len(fake.calls) == 1
    messages = fake.calls[0]
    assert [(message.role, message.content) for message in messages[:1]] == [
        ("system", "System evaluation prompt.")
    ]
    assert messages[1].role == "user"
    assert "Question:\nWhat changed this year?" in messages[1].content
    assert (
        "[1] (Source: reports/annual.pdf; Page: 7)\nRevenue increased to USD 12 million."
    ) in messages[1].content
    assert (
        "[2] (Source: cost review.pdf; Page: 10-12)\nOperating costs fell by 5 percent."
    ) in messages[1].content
    assert "Answer using only these contexts and cite them as [n]." in messages[1].content


def test_call_returns_answer_text_for_string_compatibility(tmp_path: Any) -> None:
    fake = FakeLLM(content="Compatible string answer.")
    generator = EvaluationAnswerGenerator(
        object(),
        llm=fake,
        prompt_path=tmp_path / "missing-prompt.txt",
    )

    answer = generator("What is supported?", ["The supplied evidence."])

    assert answer == "Compatible string answer."
    assert isinstance(answer, str)


def test_empty_contexts_request_an_insufficient_evidence_answer(tmp_path: Any) -> None:
    fake = FakeLLM(content="There is insufficient evidence to answer.")
    generator = EvaluationAnswerGenerator(
        object(),
        llm=fake,
        prompt_path=tmp_path / "missing-prompt.txt",
    )

    result = generator.generate("What happened?", [])

    assert result.content == "There is insufficient evidence to answer."
    user_message = fake.calls[0][1].content
    assert "Contexts:\n(No contexts were retrieved.)" in user_message
    assert "insufficient evidence to answer the question" in user_message


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_blank_query_is_rejected(query: str, tmp_path: Any) -> None:
    fake = FakeLLM()
    generator = EvaluationAnswerGenerator(
        object(),
        llm=fake,
        prompt_path=tmp_path / "missing-prompt.txt",
    )

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        generator.generate(query, ["context"])

    assert fake.calls == []


def test_prompt_is_loaded_from_file_and_sent_as_system_message(tmp_path: Any) -> None:
    prompt_path = tmp_path / "evaluation-prompt.txt"
    prompt_path.write_text(
        "\nCustom prompt with UTF-8: evidence only.\n",
        encoding="utf-8",
    )
    fake = FakeLLM()
    generator = EvaluationAnswerGenerator(
        object(),
        llm=fake,
        prompt_path=prompt_path,
    )

    generator.generate("Question?", ["Context."])

    assert generator.prompt_path == prompt_path
    assert generator.prompt == "Custom prompt with UTF-8: evidence only."
    assert fake.calls[0][0] == Message(
        role="system",
        content="Custom prompt with UTF-8: evidence only.",
    )
