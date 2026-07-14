"""Recursive Splitter implementation using LangChain.

This module provides a recursive character-based text splitting strategy
that respects document structure (headers, code blocks) and splits text
hierarchically to maintain semantic coherence. Chunk size can be measured
in either characters or model-specific tokens.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Recursive text splitter with character- or token-based length limits.

    This splitter uses LangChain's RecursiveCharacterTextSplitter to split text
    by trying different separators in order (paragraphs, sentences, words) while
    respecting Markdown structure elements like headers and code blocks.

    Design Principles Applied:
    - Pluggable: Implements BaseSplitter interface for factory instantiation.
    - Config-Driven: Reads chunk size, overlap, and length unit from settings.
    - Fail-Fast: Raises ImportError if langchain-text-splitters is not installed.
    - Graceful Degradation: Validates inputs and provides clear error messages.

    Attributes:
        chunk_size: Maximum size of each chunk in the configured length unit.
        chunk_overlap: Overlap between chunks in the configured length unit.
        length_unit: Either ``characters`` or ``tokens``.
        tokenizer_model: Hugging Face tokenizer used for token length.
        separators: List of separators to try in order (defaults to Markdown-aware).

    Raises:
        ImportError: If langchain-text-splitters package is not installed.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # Double newline (paragraphs)
        "\n",  # Single newline
        ". ",  # Sentence endings
        "! ",
        "? ",
        "; ",
        ", ",
        " ",  # Spaces
        "",  # Characters
    ]

    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        length_unit: Optional[str] = None,
        tokenizer_model: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RecursiveSplitter.

        Args:
            settings: Application settings containing ingestion configuration.
            chunk_size: Optional chunk-size override.
            chunk_overlap: Optional overlap override.
            separators: Optional list of separator strings (defaults to Markdown-aware separators).
            length_unit: Optional ``characters`` or ``tokens`` override.
            tokenizer_model: Optional Hugging Face tokenizer name override.
            tokenizer: Optional preloaded tokenizer, primarily for dependency
                injection and tests.
            **kwargs: Additional parameters passed to LangChain splitter.

        Raises:
            ImportError: If langchain-text-splitters is not installed.
            ValueError: If chunk_size or chunk_overlap are invalid.
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )

        self.settings = settings

        # Extract configuration from settings with overrides
        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = (
                chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
            )
            configured_length_unit = (
                length_unit
                if length_unit is not None
                else getattr(ingestion_config, "length_unit", "characters")
            )
            if not isinstance(configured_length_unit, str):
                configured_length_unit = "characters"
            self.length_unit = configured_length_unit.strip().lower()

            configured_tokenizer_model = (
                tokenizer_model
                if tokenizer_model is not None
                else getattr(ingestion_config, "tokenizer_model", None)
            )
            self.tokenizer_model = (
                configured_tokenizer_model.strip()
                if isinstance(configured_tokenizer_model, str)
                else None
            )
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e

        # Validate configuration
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        if self.length_unit not in {"characters", "tokens"}:
            raise ValueError(
                "length_unit must be one of: characters, tokens; "
                f"got: {self.length_unit!r}"
            )

        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        self.tokenizer = tokenizer
        length_function = len
        if self.length_unit == "tokens":
            if not self.tokenizer_model:
                raise ValueError("tokenizer_model is required when length_unit is 'tokens'")
            if self.tokenizer is None:
                self.tokenizer = self._load_tokenizer(self.tokenizer_model)
            length_function = self._token_length

        # Initialize LangChain splitter
        kwargs.setdefault("add_start_index", True)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=length_function,
            is_separator_regex=False,
            **kwargs,
        )

    @staticmethod
    def _load_tokenizer(model_name: str) -> Any:
        """Load a fast Hugging Face tokenizer without loading model weights."""
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Token-based splitting requires transformers>=4.51.0. "
                "Install it with: pip install 'transformers>=4.51.0'"
            ) from e

        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer {model_name!r}: {e}"
            ) from e

    def _token_length(self, text: str) -> int:
        """Count model tokens without adding generation-specific tokens."""
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return len(token_ids)

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Split text into chunks recursively.

        This method splits text by trying different separators hierarchically,
        preserving document structure like Markdown headers and code blocks.

        Args:
            text: Input text to split. Must be a non-empty string.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Additional parameters (currently unused, reserved for future extensions).

        Returns:
            A list of text chunks. Each chunk respects the configured chunk_size
            and chunk_overlap. Order preserves the original text sequence.

        Raises:
            ValueError: If input text is invalid (empty, wrong type).
            RuntimeError: If splitting fails unexpectedly.

        Example:
            >>> splitter = RecursiveSplitter(settings)
            >>> chunks = splitter.split_text("# Header\\n\\nParagraph 1.\\n\\nParagraph 2.")
            >>> len(chunks)
            1  # If text fits in chunk_size
        """
        # Validate input
        self.validate_text(text)

        try:
            # Perform splitting
            chunks = self._splitter.split_text(text)

            # Handle edge case: LangChain may return empty list for very short text
            if not chunks:
                chunks = [text]

            # Validate output
            self.validate_chunks(chunks)

            return chunks

        except Exception as e:
            # Catch any LangChain errors and provide context
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e

    def split_text_with_offsets(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[tuple[str, int, int]]:
        """Split text and retain each chunk's half-open character range."""
        self.validate_text(text)

        try:
            documents = self._splitter.create_documents([text])
            chunks_with_offsets: List[tuple[str, int, int]] = []
            previous_start = -1
            normalized_text, normalized_offsets = (
                self._normalize_whitespace_with_offsets(text)
            )
            for document in documents:
                chunk_text = document.page_content
                start_offset = text.find(chunk_text, previous_start + 1)
                end_offset: int | None = None
                if start_offset < 0:
                    start_offset, end_offset = self._locate_normalized_chunk(
                        chunk_text,
                        normalized_text,
                        normalized_offsets,
                        previous_start,
                    )
                if start_offset < 0 or end_offset is not None and end_offset < 0:
                    raise RuntimeError(
                        "Unable to locate a split chunk in the original text"
                    )
                if end_offset is None:
                    end_offset = start_offset + len(chunk_text)
                chunks_with_offsets.append(
                    (
                        chunk_text,
                        start_offset,
                        end_offset,
                    )
                )
                previous_start = start_offset

            self.validate_chunks([chunk_text for chunk_text, _, _ in chunks_with_offsets])
            return chunks_with_offsets
        except Exception as e:
            raise RuntimeError(
                f"RecursiveSplitter failed to split text with offsets: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e

    @staticmethod
    def _normalize_whitespace_with_offsets(
        text: str,
    ) -> tuple[str, list[int]]:
        """Collapse whitespace while retaining source offsets for each character."""
        normalized: list[str] = []
        offsets: list[int] = []
        inside_whitespace = False

        for index, character in enumerate(text):
            if character.isspace():
                if not inside_whitespace:
                    normalized.append(" ")
                    offsets.append(index)
                inside_whitespace = True
                continue
            normalized.append(character)
            offsets.append(index)
            inside_whitespace = False

        return "".join(normalized), offsets

    @classmethod
    def _locate_normalized_chunk(
        cls,
        chunk_text: str,
        normalized_text: str,
        normalized_offsets: list[int],
        previous_start: int,
    ) -> tuple[int, int]:
        """Locate a whitespace-normalized chunk and map it to source offsets."""
        normalized_chunk, _ = cls._normalize_whitespace_with_offsets(chunk_text)
        normalized_chunk = normalized_chunk.strip()
        if not normalized_chunk or not normalized_offsets:
            return -1, -1

        search_from = bisect_right(normalized_offsets, previous_start)
        normalized_start = normalized_text.find(normalized_chunk, search_from)
        if normalized_start < 0:
            return -1, -1

        normalized_end = normalized_start + len(normalized_chunk) - 1
        if normalized_end >= len(normalized_offsets):
            return -1, -1
        return (
            normalized_offsets[normalized_start],
            normalized_offsets[normalized_end] + 1,
        )
