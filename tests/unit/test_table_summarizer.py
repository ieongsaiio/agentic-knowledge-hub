"""Tests for the independently configured table-summary LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.libs.splitter.structured_markdown_splitter import _LLMTableSummarizer


@dataclass(frozen=True)
class _LLMConfig:
    provider: str = "openai"
    model: str = "main-model"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str = "shared-key"
    base_url: str = "https://example.test/v1"
    extra_chat_configs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _Settings:
    llm: _LLMConfig
    ingestion: object


def test_table_summarizer_uses_nested_llm_and_external_prompt(tmp_path) -> None:
    prompt_path = tmp_path / "table-summary.txt"
    prompt_path.write_text("Write a compact retrieval summary.", encoding="utf-8")
    settings = _Settings(
        llm=_LLMConfig(),
        ingestion=SimpleNamespace(
            structured_chunking={
                "table_summary": {
                    "prompt_path": str(prompt_path),
                    "llm": {
                        "model": "cheap-summary-model",
                        "temperature": 0.0,
                        "max_tokens": 256,
                    },
                }
            }
        ),
    )
    llm = MagicMock()
    llm.chat.return_value = "  Revenue was 100 in FY2024.  "

    with patch(
        "src.libs.llm.llm_factory.LLMFactory.create",
        return_value=llm,
    ) as create:
        summarizer = _LLMTableSummarizer(settings)
        result = summarizer.summarize(
            "<table><tr><td>Revenue</td><td>100</td></tr></table>",
            table_title="Annual results",
            footnotes=["Amounts are in millions."],
            previous_context="Results improved.",
            next_context="Outlook follows.",
        )

    effective_settings = create.call_args.args[0]
    assert effective_settings.llm.model == "cheap-summary-model"
    assert effective_settings.llm.temperature == 0.0
    assert effective_settings.llm.max_tokens == 256
    assert effective_settings.llm.api_key == "shared-key"
    messages = llm.chat.call_args.args[0]
    assert messages[0].content == "Write a compact retrieval summary."
    assert "<table_title>Annual results</table_title>" in messages[1].content
    assert "<previous_context>Results improved.</previous_context>" in messages[1].content
    assert "<footnote>Amounts are in millions.</footnote>" in messages[1].content
    assert "<next_context>Outlook follows.</next_context>" in messages[1].content
    assert result == "Revenue was 100 in FY2024."
