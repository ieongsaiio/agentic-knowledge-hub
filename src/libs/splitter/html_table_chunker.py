"""Build overlapping, standalone HTML row groups from a source table."""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable


@dataclass(frozen=True)
class ParsedHTMLTable:
    """Rectangular text grid obtained after expanding declared cell spans."""

    rows: tuple[tuple[str, ...], ...]
    width: int


@dataclass(frozen=True)
class HTMLTableChild:
    """A standalone table child and its source-row provenance."""

    html: str
    index_text: str
    token_count: int
    child_index: int
    source_row_indices: tuple[int, ...]
    repeated_context_row_indices: tuple[int, ...]
    overlap_row_indices: tuple[int, ...]


@dataclass(frozen=True)
class _SourceCell:
    text: str
    rowspan: int
    colspan: int


class _SourceTableParser(HTMLParser):
    """Read the first top-level HTML table without relying on regex slicing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_SourceCell]] = []
        self._table_depth = 0
        self._capturing = False
        self._row: list[_SourceCell] | None = None
        self._cell_text: list[str] | None = None
        self._cell_attrs: dict[str, str | None] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized = tag.lower()
        if normalized == "table":
            if not self._capturing and not self.rows:
                self._capturing = True
                self._table_depth = 1
            elif self._capturing:
                self._table_depth += 1
            return
        if not self._capturing or self._table_depth != 1:
            return
        if normalized == "tr":
            self._row = []
        elif normalized in {"td", "th"} and self._row is not None:
            self._cell_text = []
            self._cell_attrs = dict(attrs)
        elif normalized == "br" and self._cell_text is not None:
            self._cell_text.append(" ")

    def handle_data(self, data: str) -> None:
        if self._capturing and self._table_depth == 1 and self._cell_text is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "table" and self._capturing:
            self._table_depth -= 1
            if self._table_depth == 0:
                self._capturing = False
            return
        if not self._capturing or self._table_depth != 1:
            return
        if normalized in {"td", "th"} and self._cell_text is not None:
            assert self._row is not None
            self._row.append(
                _SourceCell(
                    text=_clean_text(" ".join(self._cell_text)),
                    rowspan=_positive_span(self._cell_attrs.get("rowspan")),
                    colspan=_positive_span(self._cell_attrs.get("colspan")),
                )
            )
            self._cell_text = None
            self._cell_attrs = {}
        elif normalized == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


class _PrefixTextParser(HTMLParser):
    """Collect visible prefix text and complete div blocks."""

    _BLOCK_TAGS = {"div", "p", "section", "header", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.divs: list[str] = []
        self._div_depth = 0
        self._div_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in self._BLOCK_TAGS:
            self.parts.append("\n")
        if normalized == "br":
            self.parts.append("\n")
        if normalized == "div":
            self._div_depth += 1

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self._div_depth:
            self._div_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized == "div" and self._div_depth:
            self._div_depth -= 1
            if self._div_depth == 0:
                value = _clean_text(" ".join(self._div_text))
                if value:
                    self.divs.append(value)
                self._div_text = []
        if normalized in self._BLOCK_TAGS:
            self.parts.append("\n")

    @property
    def lines(self) -> list[str]:
        return [
            cleaned
            for line in "".join(self.parts).splitlines()
            if (cleaned := _clean_text(line))
        ]


class HTMLTableChunker:
    """Split table body rows while repeating context rows and preserving provenance."""

    def __init__(
        self,
        *,
        target_children: int = 4,
        overlap_rows: int = 1,
        repeated_context_rows: int = 2,
        minimum_rows_per_child: int = 2,
        max_tokens: int | None = None,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        if target_children < 1:
            raise ValueError("target_children must be at least 1")
        if overlap_rows < 0:
            raise ValueError("overlap_rows cannot be negative")
        if repeated_context_rows < 0:
            raise ValueError("repeated_context_rows cannot be negative")
        if minimum_rows_per_child < 1:
            raise ValueError("minimum_rows_per_child must be at least 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.target_children = target_children
        self.overlap_rows = overlap_rows
        self.repeated_context_rows = repeated_context_rows
        self.minimum_rows_per_child = minimum_rows_per_child
        self.max_tokens = max_tokens
        self._length_function = length_function or len

    def parse(self, table_html: str) -> ParsedHTMLTable:
        """Parse the first HTML table and expand rowspan/colspan into a grid."""
        parsed, _ = self._parse_source(table_html)
        return parsed

    def extract_caption(
        self,
        table_html: str,
        *,
        preferred_title: str = "",
    ) -> str:
        """Extract a conservative caption without using document section paths."""
        preferred = self._clean_caption_candidate(preferred_title)
        if preferred:
            return preferred

        table_match = re.search(r"<table\b", table_html, flags=re.IGNORECASE)
        if table_match is None:
            return ""
        prefix = table_html[: table_match.start()]
        headings = re.findall(
            r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$",
            prefix,
        )
        if headings:
            return self._clean_caption_candidate(headings[-1])

        parser = _PrefixTextParser()
        parser.feed(prefix)
        parser.close()
        if parser.divs:
            candidate = self._clean_caption_candidate(parser.divs[-1])
            if candidate:
                return candidate

        if not parser.lines:
            return ""
        candidate = parser.lines[-1]
        word_count = len(candidate.split())
        if len(candidate) <= 160 and word_count <= 16:
            if candidate.endswith(":") or not candidate.endswith((".", ";")):
                return candidate
        return ""

    @staticmethod
    def _clean_caption_candidate(value: str) -> str:
        if not value or "<table" in value.casefold():
            return ""
        parser = _PrefixTextParser()
        parser.feed(value)
        parser.close()
        return _clean_text(" ".join(parser.lines))

    def _parse_source(
        self,
        table_html: str,
    ) -> tuple[ParsedHTMLTable, list[list[_SourceCell]]]:
        parser = _SourceTableParser()
        parser.feed(table_html)
        parser.close()
        if not parser.rows:
            raise ValueError("Input does not contain a non-empty HTML table")
        return self._expand_spans(parser.rows), parser.rows

    def split(
        self,
        table_html: str,
        *,
        title: str = "",
        prefix_text: str = "",
        suffix_text: str = "",
    ) -> list[HTMLTableChild]:
        """Return independently valid child tables with overlapping source rows."""
        parsed, source_rows = self._parse_source(table_html)
        resolved_title = self.extract_caption(
            table_html,
            preferred_title=title,
        )
        blocks = self._table_blocks(parsed, source_rows)
        if not any(data_indices for _, data_indices in blocks):
            context_indices = tuple(range(len(parsed.rows)))
            return [
                self._build_child(
                    parsed,
                    title=resolved_title,
                    child_index=0,
                    context_indices=context_indices,
                    source_indices=(),
                    overlap_indices=(),
                    prefix_text=prefix_text,
                    suffix_text=suffix_text,
                )
            ]

        populated_blocks = [block for block in blocks if block[1]]
        if self.max_tokens is not None:
            return self._split_by_token_budget(
                parsed,
                populated_blocks,
                title=resolved_title,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
            )

        total_data_rows = sum(len(data_indices) for _, data_indices in populated_blocks)
        child_count = max(
            len(populated_blocks),
            min(
                self.target_children,
                max(1, math.ceil(total_data_rows / self.minimum_rows_per_child)),
            ),
        )
        allocations = self._allocate_children(populated_blocks, child_count)
        children: list[HTMLTableChild] = []
        for (context_indices, data_indices), allocation in zip(
            populated_blocks,
            allocations,
        ):
            core_groups = self._balanced_groups(data_indices, allocation)
            consumed = 0
            for core in core_groups:
                overlap_start = max(0, consumed - self.overlap_rows)
                overlap = tuple(data_indices[overlap_start:consumed])
                source_indices = overlap + tuple(core)
                children.append(
                    self._build_child(
                        parsed,
                        title=resolved_title,
                        child_index=len(children),
                        context_indices=context_indices,
                        source_indices=source_indices,
                        overlap_indices=overlap,
                        prefix_text=prefix_text,
                        suffix_text=suffix_text,
                    )
                )
                consumed += len(core)
        return children

    def _split_by_token_budget(
        self,
        parsed: ParsedHTMLTable,
        blocks: list[tuple[tuple[int, ...], list[int]]],
        *,
        title: str,
        prefix_text: str,
        suffix_text: str,
    ) -> list[HTMLTableChild]:
        """Greedily pack complete rows while accounting for all index text."""
        assert self.max_tokens is not None
        children: list[HTMLTableChild] = []
        for context_indices, data_indices in blocks:
            core: list[int] = []
            overlap: tuple[int, ...] = ()
            for row_index in data_indices:
                candidate_core = (*core, row_index)
                candidate = self._build_child(
                    parsed,
                    title=title,
                    child_index=len(children),
                    context_indices=context_indices,
                    source_indices=overlap + candidate_core,
                    overlap_indices=overlap,
                    prefix_text=prefix_text,
                    suffix_text=suffix_text,
                )
                if not core or candidate.token_count <= self.max_tokens:
                    core.append(row_index)
                    continue

                children.append(
                    self._build_child(
                        parsed,
                        title=title,
                        child_index=len(children),
                        context_indices=context_indices,
                        source_indices=overlap + tuple(core),
                        overlap_indices=overlap,
                        prefix_text=prefix_text,
                        suffix_text=suffix_text,
                    )
                )
                overlap = tuple(core[-self.overlap_rows :]) if self.overlap_rows else ()
                core = [row_index]
                while overlap:
                    next_child = self._build_child(
                        parsed,
                        title=title,
                        child_index=len(children),
                        context_indices=context_indices,
                        source_indices=overlap + tuple(core),
                        overlap_indices=overlap,
                        prefix_text=prefix_text,
                        suffix_text=suffix_text,
                    )
                    if next_child.token_count <= self.max_tokens:
                        break
                    overlap = overlap[1:]

            if core:
                children.append(
                    self._build_child(
                        parsed,
                        title=title,
                        child_index=len(children),
                        context_indices=context_indices,
                        source_indices=overlap + tuple(core),
                        overlap_indices=overlap,
                        prefix_text=prefix_text,
                        suffix_text=suffix_text,
                    )
                )
        return children

    def _table_blocks(
        self,
        parsed: ParsedHTMLTable,
        source_rows: list[list[_SourceCell]],
    ) -> list[tuple[tuple[int, ...], list[int]]]:
        starts = [0]
        starts.extend(
            row_index
            for row_index, row in enumerate(source_rows[1:], start=1)
            if any(cell.colspan > 1 for cell in row)
            and row_index + 1 < len(source_rows)
            and self._looks_like_followup_header(parsed.rows[row_index + 1])
        )
        starts = sorted(set(starts))
        raw_blocks: list[tuple[tuple[int, ...], list[int]]] = []
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(source_rows)
            context_length = 1
            if (
                self.repeated_context_rows > 1
                and start + 1 < end
                and self._looks_like_followup_header(parsed.rows[start + 1])
            ):
                context_length = min(self.repeated_context_rows, end - start)
            context_end = start + context_length
            raw_blocks.append(
                (
                    tuple(range(start, context_end)),
                    list(range(context_end, end)),
                )
            )
        blocks: list[tuple[tuple[int, ...], list[int]]] = []
        pending_context: list[int] = []
        for context_indices, data_indices in raw_blocks:
            if not data_indices:
                pending_context.extend(context_indices)
                continue
            merged_context = tuple(pending_context) + context_indices
            blocks.append((merged_context, data_indices))
            pending_context = []
        if pending_context:
            if blocks:
                context_indices, data_indices = blocks[-1]
                blocks[-1] = (context_indices, data_indices + pending_context)
            else:
                blocks.append((tuple(pending_context), []))
        return blocks

    @staticmethod
    def _looks_like_followup_header(row: tuple[str, ...]) -> bool:
        if not row:
            return False
        first = row[0].strip().casefold()
        if first in {"", "metric", "item", "description"}:
            return True
        year_cells = sum(bool(re.search(r"\b(?:19|20)\d{2}\b", cell)) for cell in row[1:])
        return year_cells > 0

    @staticmethod
    def _allocate_children(
        blocks: list[tuple[tuple[int, ...], list[int]]],
        child_count: int,
    ) -> list[int]:
        allocations = [1] * len(blocks)
        while sum(allocations) < child_count:
            candidates = [
                (
                    len(data_indices) / (allocations[index] + 1),
                    index,
                )
                for index, (_, data_indices) in enumerate(blocks)
                if allocations[index] < len(data_indices)
            ]
            if not candidates:
                break
            _, selected = max(candidates)
            allocations[selected] += 1
        return allocations

    @staticmethod
    def _balanced_groups(values: list[int], count: int) -> list[list[int]]:
        base_size, larger_groups = divmod(len(values), count)
        groups: list[list[int]] = []
        offset = 0
        for group_index in range(count):
            size = base_size + (1 if group_index < larger_groups else 0)
            groups.append(values[offset : offset + size])
            offset += size
        return groups

    @staticmethod
    def _expand_spans(rows: list[list[_SourceCell]]) -> ParsedHTMLTable:
        occupied: dict[tuple[int, int], str] = {}
        width = 0
        row_count = len(rows)
        for row_index, row in enumerate(rows):
            column = 0
            for cell in row:
                while any(
                    (row_index, covered_column) in occupied
                    for covered_column in range(column, column + cell.colspan)
                ):
                    column += 1
                for covered_row in range(
                    row_index,
                    min(row_count, row_index + cell.rowspan),
                ):
                    for covered_column in range(column, column + cell.colspan):
                        occupied[(covered_row, covered_column)] = cell.text
                width = max(width, column + cell.colspan)
                column += cell.colspan
        expanded = tuple(
            tuple(occupied.get((row_index, column), "") for column in range(width))
            for row_index in range(row_count)
        )
        return ParsedHTMLTable(rows=expanded, width=width)

    def _build_child(
        self,
        parsed: ParsedHTMLTable,
        *,
        title: str,
        child_index: int,
        context_indices: tuple[int, ...],
        source_indices: tuple[int, ...],
        overlap_indices: tuple[int, ...],
        prefix_text: str = "",
        suffix_text: str = "",
    ) -> HTMLTableChild:
        rendered_indices = context_indices + source_indices
        rendered_rows = [parsed.rows[index] for index in rendered_indices]
        table_rows = [
            "  <tr>"
            + "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
            + "</tr>"
            for row in rendered_rows
        ]
        parts: list[str] = []
        if title:
            parts.append(f'<div class="table-title">{html.escape(title)}</div>')
        parts.extend(["<table>", "<tbody>", *table_rows, "</tbody>", "</table>"])
        rendered_html = "\n".join(parts)
        index_text = "\n\n".join(
            part.strip()
            for part in (prefix_text, rendered_html, suffix_text)
            if part.strip()
        )
        return HTMLTableChild(
            html=rendered_html,
            index_text=index_text,
            token_count=self._length_function(index_text),
            child_index=child_index,
            source_row_indices=source_indices,
            repeated_context_row_indices=context_indices,
            overlap_row_indices=overlap_indices,
        )


def _positive_span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()
