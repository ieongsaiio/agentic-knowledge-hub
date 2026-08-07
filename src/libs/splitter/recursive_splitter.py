"""Recursive Splitter implementation using LangChain.

This module provides a recursive character-based text splitting strategy
that respects document structure (headers, code blocks) and splits text
hierarchically to maintain semantic coherence. Chunk size can be measured
in either characters or model-specific tokens.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

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
        "\n\n",  # Markdown blocks and paragraphs
        "\n###### ",
        "\n##### ",
        "\n#### ",
        "\n### ",
        "\n## ",
        "\n# ",
        "\n---\n",  # Markdown horizontal rules
        ".\n",  # Sentence endings before PDF line-wrap boundaries
        "!\n",
        "?\n",
        ";\n",
        "! ",
        "? ",
        "; ",
        "\n",  # PDF line wraps and Markdown rows/list items
        ", ",
        " ",  # Words
        "",  # Characters, used only as a final fallback
    ]

    # MarkItDown can emit table cells across several physical lines. Keep a
    # small gap between pipe-bearing lines inside one protected table block.
    _MAX_TABLE_MARKER_GAP_LINES = 8
    _MAX_SHORT_HEADING_CHARS = 200
    _MAX_SHORT_HEADING_TOKENS = 50
    _PAGE_NUMBER_ONLY = re.compile(
        r"^(?:page[\s\u00a0]*)?(?:\d{1,3}|[ivxlcdm]+)$",
        re.IGNORECASE,
    )
    _HEADING_PHRASES = (
        "table of contents",
        "consolidated statement",
        "balance sheet",
        "cash flows",
        "income statement",
        "stockholders' equity",
        "shareholders' equity",
        "financial condition",
        "net revenues by segment",
    )

    def __init__(
        self,
        settings: Any,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        length_unit: str | None = None,
        tokenizer_model: str | None = None,
        tokenizer: Any | None = None,
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
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
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
            chunks = [
                chunk_text
                for chunk_text, _, _ in self.split_text_with_offsets(
                    text,
                    trace=trace,
                )
            ]

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
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, int, int]]:
        """Split text and retain each chunk's half-open character range."""
        self.validate_text(text)

        try:
            table_spans = self._find_markdown_table_spans(text)
            if table_spans:
                chunks_with_offsets = self._split_markdown_with_offsets(
                    text,
                    table_spans,
                )
                chunks_with_offsets = self._postprocess_chunks_with_offsets(
                    text,
                    chunks_with_offsets,
                )
                self.validate_chunks(
                    [chunk_text for chunk_text, _, _ in chunks_with_offsets]
                )
                return chunks_with_offsets

            documents = self._splitter.create_documents([text])
            chunks_with_offsets: list[tuple[str, int, int]] = []
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

            chunks_with_offsets = self._postprocess_chunks_with_offsets(
                text,
                chunks_with_offsets,
            )
            self.validate_chunks([chunk_text for chunk_text, _, _ in chunks_with_offsets])
            return chunks_with_offsets
        except Exception as e:
            raise RuntimeError(
                f"RecursiveSplitter failed to split text with offsets: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e

    def _postprocess_chunks_with_offsets(
        self,
        source_text: str,
        chunks: list[tuple[str, int, int]],
    ) -> list[tuple[str, int, int]]:
        """Remove standalone page markers and merge short headings forward."""
        ordered = sorted(chunks, key=lambda item: (item[1], item[2]))
        removed_page_ranges = [
            (start, end)
            for chunk_text, start, end in ordered
            if self._is_page_number_only(chunk_text)
        ]
        retained = [
            item for item in ordered if not self._is_page_number_only(item[0])
        ]
        if not retained:
            return ordered

        merged: list[tuple[str, int, int]] = []
        for item in reversed(retained):
            chunk_text, start, end = item
            if not merged or not self._is_short_heading(chunk_text):
                merged.insert(0, item)
                continue

            next_text, next_start, next_end = merged[0]
            lower_bound = min(end, next_start)
            upper_bound = max(end, next_start)
            crosses_removed_page = any(
                removed_start < upper_bound and removed_end > lower_bound
                for removed_start, removed_end in removed_page_ranges
            )
            gap_has_content = (
                end <= next_start and bool(source_text[end:next_start].strip())
            )
            if crosses_removed_page or gap_has_content:
                merged.insert(0, item)
                continue

            combined_text, combined_start, combined_end = self._trim_source_range(
                source_text,
                start,
                next_end,
            )
            if self._measure(combined_text) > self.chunk_size:
                merged.insert(0, item)
                continue

            merged[0] = (combined_text, combined_start, combined_end)

        return merged

    @classmethod
    def _is_page_number_only(cls, text: str) -> bool:
        normalized = " ".join(text.replace("\u00a0", " ").split())
        return bool(normalized and cls._PAGE_NUMBER_ONLY.fullmatch(normalized))

    def _is_short_heading(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped or len(stripped) > self._MAX_SHORT_HEADING_CHARS:
            return False
        if self._measure(stripped) > self._MAX_SHORT_HEADING_TOKENS:
            return False
        if self._find_markdown_table_spans(stripped):
            return False

        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        meaningful_lines = [
            line
            for line in lines
            if line.casefold() != "table of contents"
            and not self._is_page_number_only(line)
        ]
        if not meaningful_lines:
            return False

        meaningful = " ".join(meaningful_lines)
        lowered = meaningful.casefold()
        words = re.findall(r"[A-Za-z][A-Za-z0-9'&.-]*", meaningful)
        if not words:
            return False
        if meaningful_lines[0].startswith("#"):
            return True
        if any(phrase in lowered for phrase in self._HEADING_PHRASES):
            return True
        if re.match(r"^note\s+\d+\b", lowered):
            return True
        if meaningful.rstrip().endswith(":") and len(words) <= 30:
            return True

        letters = [character for character in meaningful if character.isalpha()]
        uppercase_ratio = (
            sum(character.isupper() for character in letters) / len(letters)
            if letters
            else 0.0
        )
        if uppercase_ratio >= 0.8 and len(words) <= 20:
            return True

        title_case_ratio = sum(word[0].isupper() for word in words) / len(words)
        if title_case_ratio >= 0.75 and len(words) <= 12:
            return True

        sentence_endings = len(re.findall(r"[!?;]|\.(?:\s|$)", meaningful))
        return len(words) <= 12 and sentence_endings == 0

    def _split_markdown_with_offsets(
        self,
        text: str,
        table_spans: list[tuple[int, int]],
    ) -> list[tuple[str, int, int]]:
        """Split prose normally while treating Markdown tables as blocks."""
        chunks: list[tuple[str, int, int]] = []
        cursor = 0

        for table_start, table_end in table_spans:
            if cursor < table_start:
                chunks.extend(
                    self._split_segment_with_offsets(
                        text[cursor:table_start],
                        cursor,
                    )
                )

            table_text, start, end = self._trim_source_range(
                text,
                table_start,
                table_end,
            )
            if table_text:
                if self._measure(table_text) <= self.chunk_size:
                    chunks.append((table_text, start, end))
                else:
                    chunks.extend(
                        self._split_oversized_table_with_offsets(
                            text,
                            start,
                            end,
                        )
                    )
            cursor = table_end

        if cursor < len(text):
            chunks.extend(
                self._split_segment_with_offsets(text[cursor:], cursor)
            )

        return sorted(chunks, key=lambda item: (item[1], item[2]))

    def _split_segment_with_offsets(
        self,
        segment: str,
        base_offset: int,
    ) -> list[tuple[str, int, int]]:
        """Use the configured recursive splitter on one non-table segment."""
        if not segment or not segment.strip():
            return []

        documents = self._splitter.create_documents([segment])
        normalized_text, normalized_offsets = self._normalize_whitespace_with_offsets(
            segment
        )
        previous_start = -1
        chunks: list[tuple[str, int, int]] = []

        for document in documents:
            chunk_text = document.page_content
            start_offset = segment.find(chunk_text, previous_start + 1)
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
                    "Unable to locate a split Markdown segment in the source text"
                )
            if end_offset is None:
                end_offset = start_offset + len(chunk_text)
            chunks.append(
                (
                    chunk_text,
                    base_offset + start_offset,
                    base_offset + end_offset,
                )
            )
            previous_start = start_offset

        return chunks

    def _split_oversized_table_with_offsets(
        self,
        source_text: str,
        table_start: int,
        table_end: int,
    ) -> list[tuple[str, int, int]]:
        """Split an oversized table at complete physical-line boundaries."""
        table_text = source_text[table_start:table_end]
        line_ranges: list[tuple[int, int]] = []
        cursor = 0
        for line in table_text.splitlines(keepends=True):
            line_ranges.append((cursor, cursor + len(line)))
            cursor += len(line)
        if cursor < len(table_text):
            line_ranges.append((cursor, len(table_text)))

        chunks: list[tuple[str, int, int]] = []
        line_index = 0
        while line_index < len(line_ranges):
            candidate_end = line_index
            best_end: int | None = None

            while candidate_end < len(line_ranges):
                relative_start = line_ranges[line_index][0]
                relative_end = line_ranges[candidate_end][1]
                candidate = table_text[relative_start:relative_end].strip()
                if candidate and self._measure(candidate) <= self.chunk_size:
                    best_end = candidate_end
                    candidate_end += 1
                    continue
                break

            if best_end is None:
                relative_start, relative_end = line_ranges[line_index]
                chunks.extend(
                    self._split_segment_with_offsets(
                        table_text[relative_start:relative_end],
                        table_start + relative_start,
                    )
                )
                line_index += 1
                continue

            relative_start = line_ranges[line_index][0]
            relative_end = line_ranges[best_end][1]
            chunk_text, start, end = self._trim_source_range(
                source_text,
                table_start + relative_start,
                table_start + relative_end,
            )
            if chunk_text:
                chunks.append((chunk_text, start, end))

            next_line = best_end + 1
            if self.chunk_overlap > 0 and best_end > line_index:
                overlap_start = best_end
                while overlap_start > line_index:
                    proposed_start = line_ranges[overlap_start - 1][0]
                    overlap_text = table_text[
                        proposed_start:line_ranges[best_end][1]
                    ].strip()
                    if self._measure(overlap_text) > self.chunk_overlap:
                        break
                    overlap_start -= 1
                next_line = max(line_index + 1, overlap_start)
            line_index = next_line

        return chunks

    def _measure(self, text: str) -> int:
        """Measure text in the configured character or token unit."""
        return self._token_length(text) if self.length_unit == "tokens" else len(text)

    @classmethod
    def _find_markdown_table_spans(cls, text: str) -> list[tuple[int, int]]:
        """Locate Markdown table regions, including MarkItDown multiline cells."""
        lines: list[tuple[int, int, str]] = []
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            end = cursor + len(raw_line)
            lines.append((cursor, end, raw_line.rstrip("\r\n")))
            cursor = end
        if cursor < len(text):
            lines.append((cursor, len(text), text[cursor:]))

        marker_indexes = [
            index
            for index, (_, _, line) in enumerate(lines)
            if cls._is_markdown_table_marker(line)
        ]
        if len(marker_indexes) < 2:
            return []

        clusters: list[list[int]] = []
        current = [marker_indexes[0]]
        for marker_index in marker_indexes[1:]:
            if marker_index - current[-1] <= cls._MAX_TABLE_MARKER_GAP_LINES:
                current.append(marker_index)
            else:
                clusters.append(current)
                current = [marker_index]
        clusters.append(current)

        spans: list[tuple[int, int]] = []
        for cluster in clusters:
            cluster_lines = [lines[index][2] for index in cluster]
            has_delimiter = any(
                cls._is_markdown_table_delimiter(line) for line in cluster_lines
            )
            if len(cluster) < 3 and not has_delimiter:
                continue

            # MarkItDown may place one logical cell across physical lines, for
            # example ``| Deferred`` followed by ``income`` and ``taxes |``.
            # Expand to blank-line boundaries so those edge lines stay inside
            # the protected table block even when they contain only one pipe.
            first_line = cluster[0]
            while first_line > 0 and lines[first_line - 1][2].strip():
                first_line -= 1
            last_line = cluster[-1]
            while (
                last_line + 1 < len(lines)
                and lines[last_line + 1][2].strip()
            ):
                last_line += 1
            spans.append((lines[first_line][0], lines[last_line][1]))

        return cls._merge_overlapping_spans(spans)

    @staticmethod
    def _is_markdown_table_marker(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and stripped.count("|") >= 2

    @staticmethod
    def _is_markdown_table_delimiter(line: str) -> bool:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        return len(cells) >= 2 and all(
            bool(cell) and set(cell) <= {"-", ":"} and cell.count("-") >= 3
            for cell in cells
        )

    @staticmethod
    def _merge_overlapping_spans(
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not spans:
            return []
        merged = [spans[0]]
        for start, end in spans[1:]:
            previous_start, previous_end = merged[-1]
            if start <= previous_end:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _trim_source_range(
        text: str,
        start: int,
        end: int,
    ) -> tuple[str, int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return text[start:end], start, end

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
