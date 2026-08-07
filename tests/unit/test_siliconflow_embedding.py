"""Tests for the SiliconFlow embedding provider."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.libs.embedding.siliconflow_embedding import (
    SiliconFlowEmbedding,
    SiliconFlowEmbeddingError,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        embedding=SimpleNamespace(
            model="Qwen/Qwen3-Embedding-0.6B",
            dimensions=2,
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )


def test_embed_retries_when_response_data_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SiliconFlowEmbedding(_settings(), max_retries=2)
    null_response = Mock(status_code=200)
    null_response.json.return_value = {"data": None}
    valid_response = Mock(status_code=200)
    valid_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.1, 0.2]}]
    }
    post = Mock(side_effect=[null_response, valid_response])
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("src.libs.embedding.siliconflow_embedding.time.sleep", Mock())

    assert provider.embed(["hello"]) == [[0.1, 0.2]]
    assert post.call_count == 2


def test_embed_reports_null_data_after_retries_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SiliconFlowEmbedding(_settings(), max_retries=2)
    null_response = Mock(status_code=200)
    null_response.json.return_value = {"data": None}
    post = Mock(return_value=null_response)
    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("src.libs.embedding.siliconflow_embedding.time.sleep", Mock())

    with pytest.raises(SiliconFlowEmbeddingError, match="after 2 attempts"):
        provider.embed(["hello"])

    assert post.call_count == 2
