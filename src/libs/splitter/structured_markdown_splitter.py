"""Section-aware Markdown splitter that protects semantic block units."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Any

from markdown_it import MarkdownIt

from src.core.types import Document
from src.libs.loader.markdown_section_tree import build_markdown_section_tree
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.html_table_chunker import HTMLTableChunker
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.table_linearizer import TableLinearizer

logger = logging.getLogger(__name__)


def strip_table_child_section_path(text: str) -> str:
    """Remove only a leading generated ``Section:`` paragraph."""
    if not text.startswith("Section:"):
        return text
    parts = re.split(r"\r?\n[ \t]*\r?\n", text, maxsplit=1)
    return parts[1] if len(parts) == 2 else text


class _LLMTableSummarizer:
    def __init__(self, settings: Any) -> None:
        from src.core.settings import resolve_path
        from src.libs.llm.llm_factory import LLMFactory

        config = dict(
            settings.ingestion.structured_chunking.get("table_summary") or {}
        )
        llm_overrides = config.get("llm") or {}
        if not isinstance(llm_overrides, dict):
            raise ValueError(
                "ingestion.structured_chunking.table_summary.llm must be a mapping"
            )
        summary_llm = replace(settings.llm, **llm_overrides)
        self._llm = LLMFactory.create(replace(settings, llm=summary_llm))
        prompt_path = str(
            config.get("prompt_path", "config/prompts/table_summary.txt")
        )
        self._system_prompt = resolve_path(prompt_path).read_text(
            encoding="utf-8"
        ).strip()

    def summarize(
        self,
        table_text: str,
        *,
        table_title: str | None = None,
        footnotes: list[str] | None = None,
        previous_context: str | None = None,
        next_context: str | None = None,
    ) -> str:
        from src.libs.llm.base_llm import Message

        context_parts = [
            f"<table_title>{table_title}</table_title>"
            if table_title
            else "",
            f"<previous_context>{previous_context}</previous_context>"
            if previous_context
            else "",
            f"<table>{table_text}</table>",
            *[
                f"<footnote>{footnote}</footnote>"
                for footnote in (footnotes or [])
                if footnote.strip()
            ],
            f"<next_context>{next_context}</next_context>"
            if next_context
            else "",
        ]
        user_prompt = "\n".join(
            part for part in context_parts if part
        )
        response = self._llm.chat(
            [
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=user_prompt),
            ],
        )
        summary = response if isinstance(response, str) else response.content
        summary = summary.strip()
        if not summary:
            raise RuntimeError("Table summary LLM returned empty content")
        return summary


@dataclass
class SplitFragment:
    """A source-exact chunk plus retrieval-specific representations."""

    text: str
    start_offset: int
    end_offset: int
    dense_index_text: str
    sparse_index_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Unit:
    text: str
    start: int
    end: int
    unit_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredMarkdownSplitter(BaseSplitter):
    """Split Section direct content while keeping special Markdown blocks whole."""

    _SPECIAL_TYPES = {
        "table_open": "table",
        "bullet_list_open": "list",
        "ordered_list_open": "list",
        "fence": "code",
        "code_block": "code",
        "blockquote_open": "blockquote",
    }
    _HEADING_LINE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+.*(?:\r?\n|$)")

    def __init__(
        self,
        settings: Any,
        *,
        tokenizer: Any | None = None,
        table_summarizer: Any | None = None,
        **_: Any,
    ) -> None:
        ingestion = settings.ingestion
        config = getattr(ingestion, "structured_chunking", None) or {}
        self.chunk_size = int(ingestion.chunk_size)
        model_max_tokens = int(
            getattr(settings.embedding, "max_tokens", self.chunk_size)
        )
        self.embedding_max_tokens = int(
            config.get("embedding_max_tokens", model_max_tokens)
        )
        self.embedding_safety_margin = int(
            config.get("embedding_safety_margin", 0)
        )
        self.table_dense_representation = str(
            config.get("table_dense_representation", "linearized")
        ).strip().lower()
        if self.table_dense_representation not in {"linearized", "original"}:
            raise ValueError(
                "ingestion.structured_chunking.table_dense_representation "
                "must be one of: linearized, original"
            )
        self.table_context_tokens = int(config.get("table_context_tokens", 80))
        if self.table_context_tokens < 0:
            raise ValueError(
                "ingestion.structured_chunking.table_context_tokens "
                "cannot be negative"
            )
        self._table_summary_enabled = bool(
            config.get("table_summary", {}).get("enabled", False)
        )
        table_summary_config = config.get("table_summary", {}) or {}
        self._table_summary_config = table_summary_config
        self._table_summary_max_workers = int(
            table_summary_config.get("max_workers", 5)
        )
        self._table_summary_fail_on_error = bool(
            table_summary_config.get("fail_on_error", False)
        )
        table_child_config = config.get("table_child_chunking", {}) or {}
        self.table_child_enabled = bool(table_child_config.get("enabled", False))
        self.table_child_max_tokens = int(
            table_child_config.get("max_tokens", 768)
        )
        self._table_child_chunker = HTMLTableChunker(
            max_tokens=self.table_child_max_tokens,
            overlap_rows=int(table_child_config.get("overlap_rows", 1)),
            repeated_context_rows=int(
                table_child_config.get("repeated_context_rows", 2)
            ),
            length_function=self._length,
        )
        self._table_summarizer = table_summarizer
        self._settings = settings
        self._linearizer = TableLinearizer()
        self._text_splitter = RecursiveSplitter(
            settings,
            tokenizer=tokenizer,
        )
        self._tokenizer = tokenizer or self._text_splitter.tokenizer
        self._text_splitters = {self.chunk_size: self._text_splitter}
        self._context_splitters: dict[int, RecursiveSplitter] = {}
        self._parser = MarkdownIt("commonmark", {"html": True}).enable("table")

    def split_text(
        self,
        text: str,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
        del trace, kwargs
        document = Document(
            id="ad_hoc_document",
            text=text,
            metadata={"source_path": "<memory>"},
        )
        return [fragment.text for fragment in self.split_document(document)]

    def split_document(
        self,
        document: Document,
        trace: Any | None = None,
    ) -> list[SplitFragment]:
        del trace
        self.validate_text(document.text)
        tree = document.metadata.get("section_tree")
        if not isinstance(tree, dict):
            tree = build_markdown_section_tree(
                document.text,
                document_id=document.id,
                page_spans=document.metadata.get("page_spans"),
            )

        fragments: list[SplitFragment] = []
        for section in self._walk_sections(tree["root"]):
            content = section.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            section_start = int(section["content_start_offset"])
            header_path = list(section.get("path") or [])
            section_prefix = (
                f"Section: {' > '.join(header_path)}" if header_path else ""
            )
            prefix_length = self._length(section_prefix)
            content_limit = (
                self.chunk_size - prefix_length
                if prefix_length < self.chunk_size
                else self.chunk_size
            )
            units = self._section_units(
                content,
                section_start,
                document=document,
                max_content_length=content_limit,
            )
            fragments.extend(
                self._pack_units(
                    units,
                    header_path=header_path,
                    section_id=str(section["id"]),
                    document=document,
                    max_content_length=content_limit,
                )
            )
        if self._table_summary_enabled:
            fragments = self._materialize_table_summaries(fragments)
        retained: list[SplitFragment] = []
        for fragment in fragments:
            if self._is_heading_only_fragment(fragment):
                continue
            retained.append(fragment)
        return retained

    def _is_heading_only_fragment(self, fragment: SplitFragment) -> bool:
        """Return true when a retrieval fragment contains one heading only."""
        if fragment.metadata.get("unit_types") != ["text"]:
            return False
        tokens = self._parser.parse(fragment.text.strip())
        return (
            len(tokens) == 3
            and tokens[0].type == "heading_open"
            and tokens[1].type == "inline"
            and tokens[2].type == "heading_close"
        )

    @staticmethod
    def _walk_sections(root: dict[str, Any]) -> Iterable[dict[str, Any]]:
        yield root
        for child in root.get("subsections", []):
            yield from StructuredMarkdownSplitter._walk_sections(child)

    def _section_units(
        self,
        content: str,
        base_offset: int,
        document: Document | None = None,
        max_content_length: int | None = None,
    ) -> list[_Unit]:
        line_offsets = self._line_offsets(content)
        tokens = self._parser.parse(content)
        spans: list[tuple[int, int, str, dict[str, Any]]] = []
        for token in tokens:
            unit_type = self._SPECIAL_TYPES.get(token.type)
            if unit_type is None and token.type == "html_block":
                if "<table" in token.content.lower():
                    unit_type = "table"
            if unit_type is None or token.map is None:
                continue
            start = line_offsets[token.map[0]]
            end = line_offsets[token.map[1]]
            if any(
                existing_start <= start and end <= existing_end
                for existing_start, existing_end, _, _ in spans
            ):
                continue
            spans.append((start, end, unit_type, {}))

        spans.sort(key=lambda item: item[0])
        if document is not None:
            spans = self._claim_table_context(
                content,
                base_offset,
                tokens,
                line_offsets,
                spans,
                document,
            )

        units: list[_Unit] = []
        cursor = 0
        for start, end, unit_type, unit_metadata in spans:
            if start < cursor:
                continue
            units.extend(
                self._text_units(
                    content[cursor:start],
                    base_offset + cursor,
                    max_content_length=max_content_length,
                )
            )
            raw = content[start:end].rstrip()
            raw_end = start + len(raw)
            if raw.strip():
                units.append(
                    _Unit(
                        text=raw,
                        start=base_offset + start,
                        end=base_offset + raw_end,
                        unit_type=unit_type,
                        metadata=unit_metadata,
                    )
                )
            cursor = end
        units.extend(
            self._text_units(
                content[cursor:],
                base_offset + cursor,
                max_content_length=max_content_length,
            )
        )
        if document is not None and self.table_context_tokens:
            self._attach_table_context(units, document)
        return units

    def _claim_table_context(
        self,
        content: str,
        base_offset: int,
        tokens: list[Any],
        line_offsets: list[int],
        spans: list[tuple[int, int, str, dict[str, Any]]],
        document: Document,
    ) -> list[tuple[int, int, str, dict[str, Any]]]:
        """Assign adjacent parsed captions and footnotes to their table span."""
        blocks = self._top_level_blocks(tokens, line_offsets)
        claimed: list[tuple[int, int, str, dict[str, Any]]] = []
        for start, end, unit_type, unit_metadata in spans:
            if unit_type != "table":
                claimed.append((start, end, unit_type, unit_metadata))
                continue

            table_text = content[start:end].rstrip()
            table_end = start + len(table_text)
            table_unit = _Unit(
                text=table_text,
                start=base_offset + start,
                end=base_offset + table_end,
                unit_type="table",
            )
            metadata = self._table_metadata(document, table_unit)
            expanded_start = start
            expanded_end = table_end

            title = metadata.get("table_title")
            if isinstance(title, str) and title.strip():
                previous = self._adjacent_block_before(
                    content,
                    blocks,
                    start,
                )
                if (
                    previous is not None
                    and self._block_matches_text(
                        content[previous[0] : previous[1]],
                        title,
                    )
                ):
                    expanded_start = previous[0]

            footnotes = [
                note
                for note in metadata.get("vision_footnotes", [])
                if isinstance(note, str) and note.strip()
            ]
            cursor = end
            for footnote in footnotes:
                following = self._adjacent_block_after(
                    content,
                    blocks,
                    cursor,
                )
                if following is None:
                    break
                if not self._block_matches_text(
                    content[following[0] : following[1]],
                    footnote,
                ):
                    break
                expanded_end = self._rstrip_end(content, following[0], following[1])
                cursor = following[1]

            internal_metadata = dict(metadata)
            internal_metadata["_table_source_text"] = table_text
            internal_metadata["_table_start_offset"] = base_offset + start
            internal_metadata["_table_end_offset"] = base_offset + table_end
            claimed.append(
                (
                    expanded_start,
                    expanded_end,
                    unit_type,
                    internal_metadata,
                )
            )
        claimed.sort(key=lambda item: item[0])
        return claimed

    @staticmethod
    def _top_level_blocks(
        tokens: list[Any],
        line_offsets: list[int],
    ) -> list[tuple[int, int, str]]:
        blocks: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for token in tokens:
            if token.map is None or token.level != 0 or token.nesting < 0:
                continue
            start = line_offsets[token.map[0]]
            end = line_offsets[token.map[1]]
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            blocks.append((start, end, token.type))
        blocks.sort(key=lambda item: item[0])
        return blocks

    @staticmethod
    def _adjacent_block_before(
        content: str,
        blocks: list[tuple[int, int, str]],
        boundary: int,
    ) -> tuple[int, int, str] | None:
        candidates = [
            block
            for block in blocks
            if block[1] <= boundary
            and block[2] in {"paragraph_open", "heading_open", "html_block"}
        ]
        if not candidates:
            return None
        candidate = max(candidates, key=lambda block: block[1])
        return candidate if not content[candidate[1] : boundary].strip() else None

    @staticmethod
    def _adjacent_block_after(
        content: str,
        blocks: list[tuple[int, int, str]],
        boundary: int,
    ) -> tuple[int, int, str] | None:
        candidates = [
            block
            for block in blocks
            if block[0] >= boundary
            and block[2] in {"paragraph_open", "heading_open", "html_block"}
        ]
        if not candidates:
            return None
        candidate = min(candidates, key=lambda block: block[0])
        return candidate if not content[boundary : candidate[0]].strip() else None

    @classmethod
    def _block_matches_text(cls, raw_block: str, expected: str) -> bool:
        plain = re.sub(r"<[^>]+>", " ", raw_block)
        plain = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+", "", plain)
        return cls._normalized(plain) == cls._normalized(expected)

    @staticmethod
    def _rstrip_end(content: str, start: int, end: int) -> int:
        return start + len(content[start:end].rstrip())

    def _text_units(
        self,
        text: str,
        base_offset: int,
        *,
        max_content_length: int | None = None,
    ) -> list[_Unit]:
        if not text.strip():
            return []
        splitter = self._text_splitter_for(
            max_content_length or self.chunk_size,
        )
        fragments = splitter.split_text_with_offsets(text)
        return [
            _Unit(
                text=fragment,
                start=base_offset + start,
                end=base_offset + end,
                unit_type="text",
                metadata={"_flush_after": index < len(fragments) - 1},
            )
            for index, (fragment, start, end) in enumerate(fragments)
            if fragment.strip()
        ]

    def _text_splitter_for(self, chunk_size: int) -> RecursiveSplitter:
        cached = self._text_splitters.get(chunk_size)
        if cached is not None:
            return cached
        overlap = min(self._text_splitter.chunk_overlap, max(0, chunk_size - 1))
        splitter = RecursiveSplitter(
            self._settings,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=self._text_splitter.separators,
            tokenizer=self._tokenizer,
        )
        self._text_splitters[chunk_size] = splitter
        return splitter

    def _attach_table_context(
        self,
        units: list[_Unit],
        document: Document,
    ) -> None:
        """Copy bounded adjacent text into table retrieval units."""
        for index, unit in enumerate(units):
            if unit.unit_type != "table":
                continue

            previous_context = ""
            next_context = ""
            expanded_start = unit.start
            expanded_end = unit.end

            if index > 0 and units[index - 1].unit_type == "text":
                previous_index = index - 1
                while (
                    previous_index > 0
                    and units[previous_index - 1].unit_type == "text"
                ):
                    previous_index -= 1
                previous_start = units[previous_index].start
                previous_end = units[index - 1].end
                previous_text = document.text[previous_start:previous_end]
                previous_context, relative_start, _ = self._context_window(
                    previous_text,
                    from_end=True,
                )
                expanded_start = previous_start + relative_start

            if index + 1 < len(units) and units[index + 1].unit_type == "text":
                following_index = index + 1
                while (
                    following_index + 1 < len(units)
                    and units[following_index + 1].unit_type == "text"
                ):
                    following_index += 1
                following_start = units[index + 1].start
                following_end = units[following_index].end
                following_text = document.text[following_start:following_end]
                next_context, _, relative_end = self._context_window(
                    following_text,
                    from_end=False,
                )
                expanded_end = following_start + relative_end

            if previous_context:
                unit.metadata["previous_context"] = previous_context
            if next_context:
                unit.metadata["next_context"] = next_context
            unit.start = expanded_start
            unit.end = expanded_end
            unit.text = document.text[expanded_start:expanded_end]

    def _context_window(
        self,
        text: str,
        *,
        from_end: bool,
    ) -> tuple[str, int, int]:
        """Return source-exact text and half-open offsets for one context side."""
        if not text or self.table_context_tokens <= 0:
            return "", 0, 0
        if self._length(text) <= self.table_context_tokens:
            return text.strip(), 0, len(text)

        fragments = self._context_splitter_for(
            self.table_context_tokens,
        ).split_text_with_offsets(text)
        fragment, start, end = fragments[-1] if from_end else fragments[0]
        return fragment.strip(), start, end

    def _context_splitter_for(self, chunk_size: int) -> RecursiveSplitter:
        cached = self._context_splitters.get(chunk_size)
        if cached is not None:
            return cached
        splitter = RecursiveSplitter(
            self._settings,
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=self._text_splitter.separators,
            tokenizer=self._tokenizer,
        )
        self._context_splitters[chunk_size] = splitter
        return splitter

    def _pack_units(
        self,
        units: list[_Unit],
        *,
        header_path: list[str],
        section_id: str,
        document: Document,
        max_content_length: int | None = None,
    ) -> list[SplitFragment]:
        fragments: list[SplitFragment] = []
        buffer: list[_Unit] = []
        buffer_length = 0
        content_limit = max_content_length or self.chunk_size

        def flush() -> None:
            nonlocal buffer, buffer_length
            if not buffer:
                return
            fragments.append(
                self._build_fragment(
                    buffer,
                    header_path=header_path,
                    section_id=section_id,
                    document=document,
                )
            )
            buffer = []
            buffer_length = 0

        for unit in units:
            if unit.unit_type == "table":
                flush()
                table_text = str(
                    unit.metadata.get("_table_source_text", unit.text)
                )
                if self.table_child_enabled and "<table" in table_text.casefold():
                    fragments.extend(
                        self._build_table_child_fragments(
                            unit,
                            header_path=header_path,
                            section_id=section_id,
                            document=document,
                        )
                    )
                else:
                    fragments.append(
                        self._build_fragment(
                            [unit],
                            header_path=header_path,
                            section_id=section_id,
                            document=document,
                        )
                    )
                if self._table_summary_enabled:
                    fragments.append(
                        self._build_table_summary_fragment(
                            unit,
                            header_path=header_path,
                            section_id=section_id,
                            document=document,
                        )
                    )
                continue

            unit_length = self._unit_index_length(unit)
            if buffer and buffer_length + unit_length > content_limit:
                flush()

            buffer.append(unit)
            buffer_length += unit_length

            if unit.unit_type == "text" and unit.metadata.get("_flush_after"):
                flush()

        flush()
        return fragments

    @staticmethod
    def _table_parent_id(
        document: Document,
        table_text: str,
        table_start: int,
        table_end: int,
    ) -> str:
        parent_digest = hashlib.sha256(
            (
                f"{document.id}:{table_start}:{table_end}:" + table_text
            ).encode("utf-8")
        ).hexdigest()[:12]
        return f"{document.id}_table_{parent_digest}"

    def _build_table_summary_fragment(
        self,
        unit: _Unit,
        *,
        header_path: list[str],
        section_id: str,
        document: Document,
    ) -> SplitFragment:
        """Create one dense-only summary alias for a complete source table."""
        table_text = str(unit.metadata.get("_table_source_text", unit.text))
        public_metadata = {
            key: value
            for key, value in unit.metadata.items()
            if not key.startswith("_")
        }
        if not public_metadata:
            public_metadata = self._table_metadata(document, unit)

        parent_start = unit.start
        parent_end = unit.end
        table_start = int(unit.metadata.get("_table_start_offset", parent_start))
        table_end = int(unit.metadata.get("_table_end_offset", parent_end))
        parent_chunk_id = self._table_parent_id(
            document,
            table_text,
            table_start,
            table_end,
        )
        raw_parent = document.text[parent_start:parent_end]
        metadata = {
            **public_metadata,
            "section_id": section_id,
            "header_path": header_path,
            "unit_types": ["table"],
            "chunk_role": "table_summary",
            "parent_chunk_id": parent_chunk_id,
            "retrieval_group_id": parent_chunk_id,
            "retrieval_returns_parent": True,
            "parent_start_offset": parent_start,
            "parent_end_offset": parent_end,
            "table_start_offset": table_start,
            "table_end_offset": table_end,
            "preserve_raw_content": True,
            "source_exact": True,
            "embedding_source_type": "llm_table_summary_supplement",
            "sparse_index_enabled": False,
            "table_summary_prompt_version": str(
                self._table_summary_config.get("prompt_version", "v1")
            ),
            "table_summary_model": str(
                (self._table_summary_config.get("llm") or {}).get(
                    "model",
                    getattr(getattr(self._settings, "llm", None), "model", ""),
                )
            ),
            "_summary_table_text": table_text,
        }
        return SplitFragment(
            text=raw_parent,
            start_offset=parent_start,
            end_offset=parent_end,
            dense_index_text="__pending_table_summary__",
            sparse_index_text=raw_parent,
            metadata=metadata,
        )

    def _materialize_table_summaries(
        self,
        fragments: list[SplitFragment],
    ) -> list[SplitFragment]:
        """Generate table summaries concurrently and drop only failed aliases."""
        pending = [
            fragment
            for fragment in fragments
            if fragment.metadata.get("chunk_role") == "table_summary"
        ]
        if not pending:
            return fragments
        if self._table_summarizer is None:
            self._table_summarizer = _LLMTableSummarizer(self._settings)

        def summarize(fragment: SplitFragment) -> str:
            metadata = fragment.metadata
            return self._table_summarizer.summarize(
                str(metadata["_summary_table_text"]),
                table_title=str(metadata.get("table_title") or "") or None,
                footnotes=[
                    footnote
                    for footnote in metadata.get("vision_footnotes", [])
                    if isinstance(footnote, str)
                ],
                previous_context=str(metadata.get("previous_context") or "")
                or None,
                next_context=str(metadata.get("next_context") or "") or None,
            )

        summaries: dict[int, str] = {}
        failures: dict[int, Exception] = {}
        max_workers = min(self._table_summary_max_workers, len(pending))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fragment = {
                executor.submit(summarize, fragment): fragment
                for fragment in pending
            }
            for future in as_completed(future_to_fragment):
                fragment = future_to_fragment[future]
                key = id(fragment)
                try:
                    summaries[key] = future.result()
                except Exception as exc:
                    failures[key] = exc

        if failures and self._table_summary_fail_on_error:
            first_error = next(iter(failures.values()))
            raise RuntimeError(
                f"Failed to summarize {len(failures)} table(s)"
            ) from first_error

        materialized: list[SplitFragment] = []
        for fragment in fragments:
            if fragment.metadata.get("chunk_role") != "table_summary":
                materialized.append(fragment)
                continue
            summary = summaries.get(id(fragment))
            if summary is None:
                logger.warning(
                    "Skipping table summary supplement after LLM failure: %s",
                    failures.get(id(fragment)),
                )
                continue
            summary_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:8]
            metadata = {
                key: value
                for key, value in fragment.metadata.items()
                if key != "_summary_table_text"
            }
            metadata["storage_id"] = (
                f"{metadata['parent_chunk_id']}_summary_{summary_hash}"
            )
            metadata["table_summary"] = summary
            fragment.metadata = metadata
            fragment.dense_index_text = summary
            materialized.append(fragment)
        return materialized

    def _build_table_child_fragments(
        self,
        unit: _Unit,
        *,
        header_path: list[str],
        section_id: str,
        document: Document,
    ) -> list[SplitFragment]:
        """Build retrievable row children while retaining a parsed-cache parent."""
        table_text = str(unit.metadata.get("_table_source_text", unit.text))
        public_metadata = {
            key: value
            for key, value in unit.metadata.items()
            if not key.startswith("_")
        }
        if not public_metadata:
            public_metadata = self._table_metadata(document, unit)

        prefix_parts: list[str] = []
        if header_path:
            prefix_parts.append(f"Section: {' > '.join(header_path)}")
        previous_context = public_metadata.get("previous_context")
        if previous_context:
            prefix_parts.append(str(previous_context))

        suffix_parts = [
            f"Footnote: {footnote}"
            for footnote in public_metadata.get("vision_footnotes", [])
            if isinstance(footnote, str) and footnote.strip()
        ]
        next_context = public_metadata.get("next_context")
        if next_context:
            suffix_parts.append(str(next_context))

        title = str(public_metadata.get("table_title") or "")
        children = self._table_child_chunker.split(
            table_text,
            title=title,
            prefix_text="\n\n".join(prefix_parts),
            suffix_text="\n\n".join(suffix_parts),
        )
        parsed_parent = self._table_child_chunker.parse(table_text)
        parent_start = unit.start
        parent_end = unit.end
        table_start = int(
            unit.metadata.get("_table_start_offset", parent_start)
        )
        table_end = int(unit.metadata.get("_table_end_offset", parent_end))
        parent_chunk_id = self._table_parent_id(
            document,
            table_text,
            table_start,
            table_end,
        )

        fragments: list[SplitFragment] = []
        for child in children:
            content_hash = hashlib.sha256(
                child.index_text.encode("utf-8")
            ).hexdigest()[:8]
            storage_id = (
                f"{parent_chunk_id}_child_{child.child_index:04d}_{content_hash}"
            )
            metadata = {
                **public_metadata,
                "section_id": section_id,
                "header_path": header_path,
                "unit_types": ["table"],
                "chunk_role": "table_child",
                "parent_chunk_id": parent_chunk_id,
                "table_child_index": child.child_index,
                "table_child_count": len(children),
                "source_row_indices": list(child.source_row_indices),
                "overlap_row_indices": list(child.overlap_row_indices),
                "repeated_prefix_row_indices": list(
                    child.repeated_context_row_indices
                ),
                "table_child_token_count": child.token_count,
                "parent_table_row_count": len(parsed_parent.rows),
                "parent_table_column_count": parsed_parent.width,
                "parent_start_offset": parent_start,
                "parent_end_offset": parent_end,
                "table_start_offset": table_start,
                "table_end_offset": table_end,
                "storage_id": storage_id,
                "preserve_raw_content": True,
                "source_exact": False,
                "embedding_source_type": "original_table_child_no_section",
            }
            fragments.append(
                SplitFragment(
                    text=child.index_text,
                    start_offset=parent_start,
                    end_offset=parent_end,
                    dense_index_text=strip_table_child_section_path(
                        child.index_text
                    ),
                    sparse_index_text=child.index_text,
                    metadata=metadata,
                )
            )
        return fragments

    def _unit_index_length(self, unit: _Unit) -> int:
        if unit.unit_type == "table":
            table_text = unit.metadata.get("_table_source_text", unit.text)
            parts = [
                str(unit.metadata.get("table_title", "")),
                *[
                    str(note)
                    for note in unit.metadata.get("vision_footnotes", [])
                ],
                self._linearizer.linearize(table_text),
            ]
            return self._length("\n\n".join(part for part in parts if part))
        return self._length(unit.text)

    def _build_fragment(
        self,
        units: list[_Unit],
        *,
        header_path: list[str],
        section_id: str,
        document: Document,
    ) -> SplitFragment:
        start = units[0].start
        end = units[-1].end
        raw = document.text[start:end]
        unit_types = [unit.unit_type for unit in units]
        metadata: dict[str, Any] = {
            "section_id": section_id,
            "header_path": header_path,
            "unit_types": unit_types,
            "preserve_raw_content": True,
            "embedding_source_type": "raw_text",
        }

        dense_parts: list[str] = []
        if header_path:
            dense_parts.append(f"Section: {' > '.join(header_path)}")
        for unit in units:
            if unit.unit_type == "table":
                table_metadata = {
                    key: value
                    for key, value in unit.metadata.items()
                    if not key.startswith("_")
                }
                if not table_metadata:
                    table_metadata = self._table_metadata(document, unit)
                metadata.update(table_metadata)
                table_text = unit.metadata.get("_table_source_text", unit.text)
                dense_parts.extend(self._table_dense_parts(table_text, metadata))
            else:
                body = unit.text
                if unit.unit_type == "text" and header_path:
                    body = self._HEADING_LINE.sub("", body, count=1).strip()
                if body:
                    dense_parts.append(body)

        sparse_index_text = raw
        if "table" in unit_types:
            sparse_context: list[str] = []
            table_title = metadata.get("table_title")
            if isinstance(table_title, str) and table_title.strip():
                sparse_context.append(table_title.strip())
            sparse_context.extend(
                note.strip()
                for note in metadata.get("vision_footnotes", [])
                if isinstance(note, str) and note.strip()
            )
            missing_context = [
                value
                for value in sparse_context
                if self._normalized(value) not in self._normalized(raw)
            ]
            if missing_context:
                sparse_index_text = "\n\n".join([raw, *missing_context])
        return SplitFragment(
            text=raw,
            start_offset=start,
            end_offset=end,
            dense_index_text="\n\n".join(part for part in dense_parts if part),
            sparse_index_text=sparse_index_text,
            metadata=metadata,
        )

    def _table_dense_parts(
        self,
        table_text: str,
        metadata: dict[str, Any],
    ) -> list[str]:
        parts: list[str] = []
        previous_context = metadata.get("previous_context")
        if previous_context:
            parts.append(str(previous_context))
        title = metadata.get("table_title")
        if title:
            parts.append(f"Table title: {title}")

        table_index_text = (
            table_text
            if self.table_dense_representation == "original"
            else self._linearizer.linearize(table_text)
        )
        limit = max(1, self.embedding_max_tokens - self.embedding_safety_margin)
        if self._length(table_index_text) <= limit:
            parts.append(table_index_text)
            for footnote in metadata.get("vision_footnotes", []):
                parts.append(f"Footnote: {footnote}")
            next_context = metadata.get("next_context")
            if next_context:
                parts.append(str(next_context))
            metadata["embedding_source_type"] = (
                f"{self.table_dense_representation}_table"
            )
            return parts

        parts.append(table_index_text)
        for footnote in metadata.get("vision_footnotes", []):
            parts.append(f"Footnote: {footnote}")
        next_context = metadata.get("next_context")
        if next_context:
            parts.append(str(next_context))
        metadata["embedding_source_type"] = (
            f"oversized_{self.table_dense_representation}_table"
        )
        return parts

    def _table_metadata(
        self,
        document: Document,
        table_unit: _Unit,
    ) -> dict[str, Any]:
        artifact = document.metadata.get("parsed_artifact")
        if not isinstance(artifact, dict):
            return {}
        page = self._page_for_offset(
            document.metadata.get("page_spans"),
            table_unit.start,
        )
        blocks: list[dict[str, Any]] = []
        for page_item in artifact.get("pages", []):
            res = page_item.get("res", {}) if isinstance(page_item, dict) else {}
            page_index = res.get("page_index")
            if page is not None and page_index != page - 1:
                continue
            pruned = res.get("prunedResult", {})
            blocks.extend(
                block
                for block in pruned.get("parsing_res_list", [])
                if isinstance(block, dict)
            )

        table_indexes = [
            index
            for index, block in enumerate(blocks)
            if str(block.get("block_label", "")).lower() == "table"
        ]
        if not table_indexes:
            return {}
        table_index = min(
            table_indexes,
            key=lambda index: (
                -self._table_match_score(
                    blocks[index].get("block_content", ""),
                    table_unit.text,
                ),
                index,
            ),
        )
        result: dict[str, Any] = {}
        for index in range(table_index - 1, -1, -1):
            label = str(blocks[index].get("block_label", "")).lower()
            if label in {"table_title", "figure_title"}:
                result["table_title"] = str(
                    blocks[index].get("block_content", "")
                ).strip()
                break
            if label == "table":
                break
        footnotes: list[str] = []
        for index in range(table_index + 1, len(blocks)):
            label = str(blocks[index].get("block_label", "")).lower()
            if label in {"vision_footnote", "footnote"}:
                value = str(blocks[index].get("block_content", "")).strip()
                if value:
                    footnotes.append(value)
                continue
            if label == "table":
                break
            if footnotes:
                break
        if footnotes:
            result["vision_footnotes"] = footnotes
        return result

    def _length(self, text: str) -> int:
        if self._tokenizer is None:
            measure = getattr(self._text_splitter, "_measure", None)
            if callable(measure):
                return int(measure(text))
            return len(text)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _line_offsets(text: str) -> list[int]:
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        if offsets[-1] != len(text):
            offsets.append(len(text))
        return offsets

    @staticmethod
    def _normalized(value: Any) -> str:
        return re.sub(r"\s+", "", str(value)).lower()

    def _table_match_score(self, candidate: Any, source_table: str) -> float:
        candidate_text = self._linearizer.linearize(str(candidate))
        source_text = self._linearizer.linearize(source_table)
        candidate_terms = set(re.findall(r"[a-z0-9]+", candidate_text.lower()))
        source_terms = set(re.findall(r"[a-z0-9]+", source_text.lower()))
        if not candidate_terms:
            return 0.0
        return len(candidate_terms & source_terms) / len(candidate_terms)

    @staticmethod
    def _page_for_offset(page_spans: Any, offset: int) -> int | None:
        if not isinstance(page_spans, list):
            return None
        for span in page_spans:
            if not isinstance(span, dict):
                continue
            if span.get("start_offset", -1) <= offset < span.get("end_offset", -1):
                page = span.get("page")
                return page if isinstance(page, int) else None
        return None
