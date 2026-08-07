"""Real API integration tests for the configured OpenAI text provider."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.core.settings import load_settings
from src.libs.llm.base_llm import Message
from src.libs.llm.openai_llm import OpenAILLM


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
def test_openai_chat_completions_real_api_from_settings() -> None:
    """Force Chat Completions while reusing the configured real endpoint."""
    settings = load_settings("config/settings.yaml")
    if settings.llm.provider != "openai":
        pytest.skip("This test requires llm.provider to be 'openai'")
    if _is_placeholder(settings.llm.api_key):
        pytest.skip("llm.api_key is missing or still uses a placeholder")

    chat_settings = replace(
        settings,
        llm=replace(
            settings.llm,
            api_mode="chat_completions",
            temperature=0.0,
            max_tokens=32,
        ),
    )
    response = OpenAILLM(chat_settings).chat(
        [
            Message(
                role="user",
                content="Reply with exactly this token: OPENAI_CHAT_OK",
            )
        ]
    )

    assert "OPENAI_CHAT_OK" in response.content
    assert response.model
    assert response.raw_response
