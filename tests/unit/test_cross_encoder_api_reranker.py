"""Tests for API-based Cross-Encoder Reranker implementation."""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.cross_encoder_api_reranker import (
    CrossEncoderAPIRerankError,
    CrossEncoderAPIReranker,
)


class MockResponse:
    def __init__(
        self,
        status_code: int = 200,
        data: Dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._data


@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.rerank = Mock(spec=RerankSettings)
    settings.rerank.enabled = True
    settings.rerank.provider = "cross_encoder_api"
    settings.rerank.model = ""
    settings.rerank.top_k = 2
    settings.rerank.api_args = {
        "base_url": "https://api.siliconflow.com/v1",
        "api_key": "test-key",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "return_documents": True,
    }
    return settings


@pytest.fixture
def sample_candidates() -> List[Dict[str, Any]]:
    return [
        {"id": "chunk_1", "text": "Python is a programming language.", "score": 0.8},
        {"id": "chunk_2", "text": "Machine learning uses neural networks.", "score": 0.7},
        {"id": "chunk_3", "content": "RAG combines retrieval and generation.", "score": 0.9},
    ]


class TestCrossEncoderAPIRerankerInit:
    def test_init_uses_api_args(self, mock_settings):
        reranker = CrossEncoderAPIReranker(mock_settings)

        assert reranker.model == "Qwen/Qwen3-Reranker-0.6B"
        assert reranker.api_key == "test-key"
        assert reranker.url == "https://api.siliconflow.com/v1/rerank"

    def test_init_model_falls_back_to_rerank_model(self, mock_settings):
        mock_settings.rerank.model = "fallback-model"
        mock_settings.rerank.api_args.pop("model")

        reranker = CrossEncoderAPIReranker(mock_settings)

        assert reranker.model == "fallback-model"

    def test_init_missing_model_raises(self, mock_settings):
        mock_settings.rerank.model = ""
        mock_settings.rerank.api_args.pop("model")

        with pytest.raises(CrossEncoderAPIRerankError, match="Missing rerank model"):
            CrossEncoderAPIReranker(mock_settings)

    def test_init_missing_api_key_raises(self, mock_settings, monkeypatch):
        mock_settings.rerank.api_args.pop("api_key")
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

        with pytest.raises(CrossEncoderAPIRerankError, match="Missing rerank API key"):
            CrossEncoderAPIReranker(mock_settings)

    def test_init_reads_api_key_from_env(self, mock_settings, monkeypatch):
        mock_settings.rerank.api_args.pop("api_key")
        monkeypatch.setenv("SILICONFLOW_API_KEY", "env-key")

        reranker = CrossEncoderAPIReranker(mock_settings)

        assert reranker.api_key == "env-key"


class TestCrossEncoderAPIRerankerCall:
    def test_rerank_posts_payload_and_maps_results(self, mock_settings, sample_candidates):
        api_response = MockResponse(
            data={
                "results": [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.72},
                ],
                "tokens": {"input_tokens": 10, "output_tokens": 0},
            }
        )

        with patch(
            "src.libs.reranker.cross_encoder_api_reranker.requests.post",
            return_value=api_response,
        ) as mock_post:
            reranker = CrossEncoderAPIReranker(mock_settings)
            result = reranker.rerank("what is rag?", sample_candidates, top_k=2)

        assert [item["id"] for item in result] == ["chunk_3", "chunk_1"]
        assert result[0]["rerank_score"] == 0.95
        assert "rerank_score" not in sample_candidates[2]

        _, call_kwargs = mock_post.call_args
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert call_kwargs["json"] == {
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "query": "what is rag?",
            "documents": [
                "Python is a programming language.",
                "Machine learning uses neural networks.",
                "RAG combines retrieval and generation.",
            ],
            "top_n": 2,
            "return_documents": True,
        }

    def test_rerank_forwards_extra_api_args(self, mock_settings, sample_candidates):
        mock_settings.rerank.api_args["max_chunks_per_doc"] = 4
        mock_settings.rerank.api_args["overlap_tokens"] = 32
        api_response = MockResponse(data={"results": [{"index": 0, "relevance_score": 0.8}]})

        with patch(
            "src.libs.reranker.cross_encoder_api_reranker.requests.post",
            return_value=api_response,
        ) as mock_post:
            reranker = CrossEncoderAPIReranker(mock_settings)
            reranker.rerank("python", sample_candidates, top_k=1)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["max_chunks_per_doc"] == 4
        assert payload["overlap_tokens"] == 32

    def test_rerank_threshold_filters_and_keeps_top_one_if_empty(
        self,
        mock_settings,
        sample_candidates,
    ):
        mock_settings.rerank.api_args["rerank_threshold"] = 0.99
        api_response = MockResponse(
            data={
                "results": [
                    {"index": 1, "relevance_score": 0.6},
                    {"index": 0, "relevance_score": 0.5},
                ]
            }
        )

        with patch(
            "src.libs.reranker.cross_encoder_api_reranker.requests.post",
            return_value=api_response,
        ):
            reranker = CrossEncoderAPIReranker(mock_settings)
            result = reranker.rerank("machine learning", sample_candidates)

        assert len(result) == 1
        assert result[0]["id"] == "chunk_2"
        assert result[0]["rerank_score"] == 0.6

    def test_rerank_http_error_raises(self, mock_settings, sample_candidates):
        api_response = MockResponse(
            status_code=401,
            data={"message": "Invalid token"},
            text="Invalid token",
        )

        with patch(
            "src.libs.reranker.cross_encoder_api_reranker.requests.post",
            return_value=api_response,
        ):
            reranker = CrossEncoderAPIReranker(mock_settings)
            with pytest.raises(CrossEncoderAPIRerankError, match="HTTP 401"):
                reranker.rerank("query", sample_candidates)

    def test_rerank_invalid_index_raises(self, mock_settings, sample_candidates):
        api_response = MockResponse(data={"results": [{"index": 99, "relevance_score": 0.8}]})

        with patch(
            "src.libs.reranker.cross_encoder_api_reranker.requests.post",
            return_value=api_response,
        ):
            reranker = CrossEncoderAPIReranker(mock_settings)
            with pytest.raises(CrossEncoderAPIRerankError, match="invalid document index"):
                reranker.rerank("query", sample_candidates)

    def test_rerank_invalid_top_k_raises(self, mock_settings, sample_candidates):
        reranker = CrossEncoderAPIReranker(mock_settings)

        with pytest.raises(ValueError, match="top_k"):
            reranker.rerank("query", sample_candidates, top_k=0)
