"""Tests for OpenAI-compatible Vision LLM implementation."""

from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.libs.llm.base_vision_llm import ImageInput
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


def _configured_api_key_is_placeholder(api_key: str | None) -> bool:
    if not api_key:
        return True
    normalized = api_key.strip().upper()
    return normalized in {"", "YOUR_API_KEY_HERE", "YOUR_OPENAI_API_KEY"}


def _apple_image_fixture() -> tuple[Path, str]:
    image_path = Path("tests/fixtures/images/theapple.png")
    if not image_path.exists():
        pytest.fail(f"Apple image fixture not found: {image_path}")
    return image_path, "image/png"


@pytest.mark.integration
@pytest.mark.llm
def test_openai_vision_real_api_call_from_settings():
    """Call the configured OpenAI-compatible Vision API with the apple fixture."""
    settings = load_settings("config/settings.yaml")

    if not settings.vision_llm or not settings.vision_llm.enabled:
        pytest.skip("Vision LLM is not enabled in config/settings.yaml")

    if settings.vision_llm.provider != "openai":
        pytest.skip("This test requires vision_llm.provider to be 'openai'")

    if _configured_api_key_is_placeholder(settings.vision_llm.api_key):
        pytest.skip("vision_llm.api_key is missing or still uses a placeholder")

    image_path, mime_type = _apple_image_fixture()
    llm = OpenAIVisionLLM(settings)

    response = llm.chat_with_image(
        text=(
            "What fruit is shown in this image? "
            "Answer with one short English sentence."
        ),
        image=ImageInput(path=image_path, mime_type=mime_type),
        temperature=0.0,
        max_tokens=64,
    )

    assert response.content
    assert len(response.content.strip()) > 0
    assert "apple" in response.content.lower()
    assert response.raw_response
