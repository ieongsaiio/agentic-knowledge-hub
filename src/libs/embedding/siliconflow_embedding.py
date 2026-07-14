"""SiliconFlow Embedding implementation.

This module provides a SiliconFlow Embedding provider implementation that works
with SiliconFlow's OpenAI-compatible embeddings endpoint. The provider is model
agnostic: Qwen embedding models are configured through settings.embedding.model,
while the provider name remains "siliconflow".
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class SiliconFlowEmbeddingError(RuntimeError):
    """Raised when SiliconFlow Embeddings API call fails."""


class SiliconFlowEmbedding(BaseEmbedding):
    """SiliconFlow Embedding provider implementation.

    This class implements the BaseEmbedding interface for SiliconFlow's
    embeddings API. It batches requests to respect provider limits and preserves
    the original input order by sorting response items by their ``index`` field.

    Attributes:
        api_key: The SiliconFlow API key for authentication.
        model: The embedding model identifier to use.
        dimensions: Optional embedding dimension override.
        url: Full SiliconFlow embeddings endpoint URL.
        max_batch_size: Maximum number of texts sent in one request.

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> settings.embedding.provider = "siliconflow"  # doctest: +SKIP
        >>> settings.embedding.model = "Qwen/Qwen3-Embedding-0.6B"  # doctest: +SKIP
        >>> embedding = SiliconFlowEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """

    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_DIMENSIONS = 1024
    DEFAULT_BASE_URL = "https://api.siliconflow.com/v1"
    DEFAULT_MAX_BATCH_SIZE = 32
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SiliconFlow Embedding provider.

        Args:
            settings: Application settings containing Embedding configuration.
            api_key: Optional API key override (falls back to settings or env var).
            base_url: Optional base URL override. The embeddings path is appended
                when the value does not already end with ``/embeddings``.
            max_batch_size: Optional request batch size override.
            timeout: Request timeout in seconds.
            max_retries: Number of attempts for transient request/API failures.
            **kwargs: Additional configuration overrides.

        Raises:
            ValueError: If API key is not provided or batch size is invalid.
        """
        self.model = getattr(settings.embedding, "model", None) or self.DEFAULT_MODEL
        self.dimensions = (
            getattr(settings.embedding, "dimensions", None)
            or kwargs.get("dimensions")
            or self.DEFAULT_DIMENSIONS
        )

        self.api_key = (
            api_key
            or getattr(settings.embedding, "api_key", None)
            or os.environ.get("SILICONFLOW_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "SiliconFlow API key not provided. Set in settings.yaml "
                "(embedding.api_key), SILICONFLOW_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        configured_base_url = (
            base_url
            or getattr(settings.embedding, "base_url", None)
            or os.environ.get("SILICONFLOW_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        self.url = self._build_embeddings_url(configured_base_url)

        self.max_batch_size = max_batch_size or kwargs.get(
            "max_batch_size",
            self.DEFAULT_MAX_BATCH_SIZE,
        )
        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be a positive integer, got {self.max_batch_size}"
            )

        self.timeout = (
            timeout
            if timeout is not None
            else kwargs.get("timeout", self.DEFAULT_TIMEOUT)
        )
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or self.timeout <= 0
        ):
            raise ValueError(
                f"timeout must be a positive number, got {self.timeout}"
            )

        self.max_retries = (
            max_retries
            if max_retries is not None
            else kwargs.get("max_retries", self.DEFAULT_MAX_RETRIES)
        )
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries <= 0
        ):
            raise ValueError(
                f"max_retries must be a positive integer, got {self.max_retries}"
            )

        self._extra_config = kwargs

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts using SiliconFlow API.

        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability.
            **kwargs: Override parameters such as dimensions or max_batch_size.

        Returns:
            List of embedding vectors in the same order as input texts.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            SiliconFlowEmbeddingError: If API call or response parsing fails.
        """
        self.validate_texts(texts)

        try:
            import requests
        except ImportError as e:
            raise SiliconFlowEmbeddingError(
                "requests library is required for SiliconFlow Embedding. "
                "Install with: pip install requests"
            ) from e

        dimensions = kwargs.get("dimensions", self.dimensions)
        max_batch_size = kwargs.get("max_batch_size", self.max_batch_size)
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be a positive integer, got {max_batch_size}")

        all_embeddings: List[List[float]] = []

        for batch_start in range(0, len(texts), max_batch_size):
            batch = texts[batch_start: batch_start + max_batch_size]
            payload = {
                "model": self.model,
                "input": batch,
                "dimensions": dimensions,
            }

            result = self._post_with_retries(requests, payload, batch_start)

            try:
                sorted_data = sorted(result["data"], key=lambda item: item["index"])
                batch_embeddings = [item["embedding"] for item in sorted_data]
            except (KeyError, TypeError) as e:
                raise SiliconFlowEmbeddingError(
                    f"Failed to extract embeddings from SiliconFlow response: {e}"
                ) from e

            if len(batch_embeddings) != len(batch):
                raise SiliconFlowEmbeddingError(
                    f"Output length mismatch for batch starting at {batch_start}: "
                    f"expected {len(batch)}, got {len(batch_embeddings)}"
                )

            all_embeddings.extend(batch_embeddings)

        if len(all_embeddings) != len(texts):
            raise SiliconFlowEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(all_embeddings)}"
            )

        return all_embeddings

    def _post_with_retries(
        self,
        requests_module: Any,
        payload: dict[str, Any],
        batch_start: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests_module.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                try:
                    result = response.json()
                except Exception as e:
                    raise SiliconFlowEmbeddingError(
                        "Failed to parse SiliconFlow Embeddings API response "
                        f"for batch starting at {batch_start}: {e}"
                    ) from e

                if "data" not in result:
                    status_code = getattr(response, "status_code", "unknown")
                    raise SiliconFlowEmbeddingError(
                        "Unexpected response from SiliconFlow API for batch "
                        f"starting at {batch_start} (status={status_code}): {result}"
                    )
                return result
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))

        raise SiliconFlowEmbeddingError(
            "SiliconFlow Embeddings API request failed after "
            f"{self.max_retries} attempts for batch starting at {batch_start}: "
            f"{last_error}"
        ) from last_error

    def get_dimension(self) -> int:
        """Get the configured embedding dimension."""
        return int(self.dimensions)

    @staticmethod
    def _build_embeddings_url(base_url: str) -> str:
        """Build the full embeddings endpoint URL from a base URL."""
        url = base_url.rstrip("/")
        if url.endswith("/embeddings"):
            return url
        return f"{url}/embeddings"
