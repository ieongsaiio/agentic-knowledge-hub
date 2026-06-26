"""API-based Cross-Encoder reranker implementation.

This module keeps remote reranking separate from the local sentence-transformers
CrossEncoder implementation. It targets OpenAI-style HTTP rerank endpoints such
as SiliconFlow's /rerank API.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderAPIRerankError(RuntimeError):
    """Raised when API-based Cross-Encoder reranking fails."""


class CrossEncoderAPIReranker(BaseReranker):
    """Cross-Encoder reranker backed by a remote HTTP API.

    Expected settings shape:

        rerank:
          enabled: true
          provider: cross_encoder_api
          model: Qwen/Qwen3-Reranker-0.6B
          top_k: 5
          api_args:
            base_url: https://api.siliconflow.com/v1
            api_key_env: SILICONFLOW_API_KEY
            return_documents: true

    ``api_args.model`` may also be used and takes precedence over
    ``rerank.model``. Additional non-internal keys in ``api_args`` are forwarded
    into the request payload, which supports SiliconFlow fields such as
    ``max_chunks_per_doc`` and ``overlap_tokens``.
    """

    DEFAULT_BASE_URL = "https://api.siliconflow.com/v1"
    DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"

    _INTERNAL_API_ARG_KEYS = {
        "api_key",
        "api_key_env",
        "base_url",
        "endpoint",
        "url",
        "timeout",
        "rerank_threshold",
        "model",
        "top_n",
    }

    def __init__(
        self,
        settings: Any,
        api_args: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.api_args = self._collect_api_args(settings, api_args)
        self.api_args.update(kwargs.pop("api_args", {}) or {})
        self.api_args.update(kwargs)
        self.kwargs = kwargs

        self.model = self._get_model_name(settings)
        self.url = self._get_rerank_url()
        self.api_key = self._get_api_key()
        self.timeout = float(
            timeout
            or self.api_args.get("timeout")
            or getattr(getattr(settings, "rerank", None), "timeout", 30.0)
            or 30.0
        )

        if not self.model:
            raise CrossEncoderAPIRerankError(
                "Missing rerank model. Set rerank.model or rerank.api_args.model."
            )
        if not self.api_key:
            raise CrossEncoderAPIRerankError(
                "Missing rerank API key. Set rerank.api_args.api_key or "
                f"environment variable {self.api_args.get('api_key_env', self.DEFAULT_API_KEY_ENV)}."
            )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using the configured remote rerank API."""
        self.validate_query(query)
        self.validate_candidates(candidates)

        top_k = self._get_top_k(candidates, kwargs)
        rerank_threshold = kwargs.get(
            "rerank_threshold",
            self.api_args.get("rerank_threshold"),
        )

        try:
            documents = self._candidate_texts(candidates)
            response_data = self._call_api(
                query=query,
                documents=documents,
                top_n=top_k,
            )
            ranked = self._parse_ranked_results(response_data, candidates)
            ranked = self._apply_threshold(ranked, rerank_threshold)

            if trace:
                self._log_trace(trace, query, len(candidates), len(ranked))

            return ranked
        except Exception as e:
            logger.error("Cross-Encoder API reranking failed: %s", e, exc_info=True)
            if isinstance(e, CrossEncoderAPIRerankError):
                raise
            raise CrossEncoderAPIRerankError(
                f"Cross-Encoder API reranking failed: {e}"
            ) from e

    def _collect_api_args(
        self,
        settings: Any,
        api_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        configured = getattr(getattr(settings, "rerank", None), "api_args", None)
        if not isinstance(configured, dict):
            configured = {}

        merged = dict(configured)
        if api_args:
            merged.update(api_args)
        return merged

    def _get_model_name(self, settings: Any) -> str:
        model = self.api_args.get("model") or getattr(
            getattr(settings, "rerank", None),
            "model",
            "",
        )
        if not isinstance(model, str):
            raise CrossEncoderAPIRerankError("rerank model must be a string")
        return model.strip()

    def _get_rerank_url(self) -> str:
        if self.api_args.get("url"):
            return str(self.api_args["url"]).rstrip("/")
        if self.api_args.get("endpoint"):
            return str(self.api_args["endpoint"]).rstrip("/")

        base_url = str(self.api_args.get("base_url") or self.DEFAULT_BASE_URL).rstrip("/")
        if base_url.endswith("/rerank"):
            return base_url
        return f"{base_url}/rerank"

    def _get_api_key(self) -> Optional[str]:
        api_key = self.api_args.get("api_key")
        if api_key:
            return str(api_key)

        env_name = str(self.api_args.get("api_key_env") or self.DEFAULT_API_KEY_ENV)
        return os.environ.get(env_name)

    def _get_top_k(self, candidates: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> int:
        if "top_k" in kwargs:
            raw_top_k = kwargs["top_k"]
        elif "top_n" in self.api_args:
            raw_top_k = self.api_args["top_n"]
        else:
            raw_top_k = (
                getattr(getattr(self.settings, "rerank", None), "top_k", None)
                or len(candidates)
            )
        if not isinstance(raw_top_k, int) or raw_top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {raw_top_k}")
        return min(raw_top_k, len(candidates))

    def _candidate_texts(self, candidates: List[Dict[str, Any]]) -> List[str]:
        documents = []
        for candidate in candidates:
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            documents.append(text)
        return documents

    def _call_api(self, query: str, documents: List[str], top_n: int) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": self.api_args.get("return_documents", True),
        }
        payload.update(self._extra_payload_args())

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise CrossEncoderAPIRerankError(
                f"Rerank API error (HTTP {response.status_code}): "
                f"{self._parse_error_response(response)}"
            )

        try:
            return response.json()
        except Exception as e:
            raise CrossEncoderAPIRerankError(f"Rerank API returned invalid JSON: {e}") from e

    def _extra_payload_args(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.api_args.items()
            if key not in self._INTERNAL_API_ARG_KEYS and value is not None
        }

    def _parse_ranked_results(
        self,
        response_data: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked_items = response_data.get("results")
        if not isinstance(ranked_items, list):
            raise CrossEncoderAPIRerankError("Rerank API response missing 'results' list")

        ranked_candidates = []
        for item in ranked_items:
            if not isinstance(item, dict):
                raise CrossEncoderAPIRerankError("Rerank API result item must be an object")

            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                raise CrossEncoderAPIRerankError(
                    f"Rerank API returned invalid document index: {index}"
                )

            score = item.get("relevance_score", 0.0)
            try:
                score = float(score)
            except (TypeError, ValueError) as e:
                raise CrossEncoderAPIRerankError(
                    f"Rerank API returned non-numeric relevance_score: {score}"
                ) from e

            candidate_copy = candidates[index].copy()
            candidate_copy["rerank_score"] = score
            ranked_candidates.append(candidate_copy)

        return ranked_candidates

    def _apply_threshold(
        self,
        ranked: List[Dict[str, Any]],
        rerank_threshold: Optional[Any],
    ) -> List[Dict[str, Any]]:
        if rerank_threshold is None:
            return ranked

        try:
            threshold = float(rerank_threshold)
        except (TypeError, ValueError) as e:
            raise ValueError(f"rerank_threshold must be numeric, got {rerank_threshold}") from e

        filtered = [
            item
            for item in ranked
            if item.get("rerank_score", 0.0) >= threshold
        ]
        if not filtered and ranked:
            return ranked[:1]
        return filtered

    def _parse_error_response(self, response: Any) -> str:
        try:
            data = response.json()
            if isinstance(data, dict) and "message" in data:
                return str(data["message"])
            return str(data)
        except Exception:
            return getattr(response, "text", "") or "Unknown error"

    def _log_trace(
        self,
        trace: Any,
        query: str,
        input_count: int,
        output_count: int,
    ) -> None:
        logger.debug(
            "Cross-Encoder API rerank: query='%s...', input=%s, output=%s",
            query[:50],
            input_count,
            output_count,
        )
