"""Tests for the independently configured table-summary LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
            document_name="ACME_2024_10K",
            section_path="Financial Statements > Revenue",
            page_range="10-11",
            table_units=[
                {"caption": "Revenue"},
                {"caption": "Expenses"},
            ],
            table_unit_count=2,
        )

    effective_settings = create.call_args.args[0]
    assert effective_settings.llm.model == "cheap-summary-model"
    assert effective_settings.llm.temperature == 0.0
    assert effective_settings.llm.max_tokens == 256
    assert effective_settings.llm.api_key == "shared-key"
    messages = llm.chat.call_args.args[0]
    assert messages[0].content == "Write a compact retrieval summary."
    assert "<table_title>Annual results</table_title>" in messages[1].content
    assert "<document_name>ACME_2024_10K</document_name>" in messages[1].content
    assert (
        "<section_path>Financial Statements > Revenue</section_path>"
        in messages[1].content
    )
    assert "<page_range>10-11</page_range>" in messages[1].content
    assert "<table_unit_count>2</table_unit_count>" in messages[1].content
    assert '<unit index="1" caption="Revenue" />' in messages[1].content
    assert '<unit index="2" caption="Expenses" />' in messages[1].content
    assert "<table_source><table>" in messages[1].content
    assert "</table></table_source>" in messages[1].content
    assert "<previous_context>Results improved.</previous_context>" in messages[1].content
    assert "<footnote>Amounts are in millions.</footnote>" in messages[1].content
    assert "<next_context>Outlook follows.</next_context>" in messages[1].content
    assert result == "Revenue was 100 in FY2024."


def test_table_summarizer_marks_a_single_table_explicitly(tmp_path) -> None:
    prompt_path = tmp_path / "table-summary.txt"
    prompt_path.write_text("Summarize the table.", encoding="utf-8")
    settings = _Settings(
        llm=_LLMConfig(),
        ingestion=SimpleNamespace(
            structured_chunking={
                "table_summary": {"prompt_path": str(prompt_path)}
            }
        ),
    )
    llm = MagicMock()
    llm.chat.return_value = "Table summary: Revenue increased."

    with patch(
        "src.libs.llm.llm_factory.LLMFactory.create",
        return_value=llm,
    ):
        summarizer = _LLMTableSummarizer(settings)
        summarizer.summarize(
            "<table><tr><td>Revenue</td><td>100</td></tr></table>",
        )

    user_prompt = llm.chat.call_args.args[0][1].content
    assert "<table_unit_count>1</table_unit_count>" in user_prompt
    assert "<table_units>" not in user_prompt


def test_default_table_summary_prompt_has_single_and_multi_table_contracts() -> None:
    prompt = Path("config/prompts/table_summary.txt").read_text(encoding="utf-8")

    assert "For one table unit:" in prompt
    assert "Table summary:" in prompt
    assert "Observed trends and relationships:" in prompt
    assert "For multiple table units:" in prompt
    assert "Table group summary:" in prompt
    assert "Observed relationships across units:" in prompt
    assert "Do not calculate new metrics" in prompt


def test_summarize_with_response_retains_provider_usage(tmp_path) -> None:
    prompt_path = tmp_path / "table-summary.txt"
    prompt_path.write_text("Summarize the table.", encoding="utf-8")
    settings = _Settings(
        llm=_LLMConfig(),
        ingestion=SimpleNamespace(
            structured_chunking={
                "table_summary": {"prompt_path": str(prompt_path)}
            }
        ),
    )
    response = SimpleNamespace(
        content="  Table summary: Revenue was 100.  ",
        usage={"prompt_tokens": 20, "completion_tokens": 8},
    )
    llm = MagicMock()
    llm.chat.return_value = response

    with patch(
        "src.libs.llm.llm_factory.LLMFactory.create",
        return_value=llm,
    ):
        summarizer = _LLMTableSummarizer(settings)
        summary, raw_response = summarizer.summarize_with_response(
            "<table><tr><td>Revenue</td><td>100</td></tr></table>",
        )

    assert summary == "Table summary: Revenue was 100."
    assert raw_response is response
    assert raw_response.usage["prompt_tokens"] == 20
