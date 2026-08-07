"""Deterministic Markdown and HTML table linearization for retrieval."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass(frozen=True)
class _HTMLCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass(frozen=True)
class _PlacedCell:
    cell: _HTMLCell
    row: int
    column: int


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_HTMLCell]] = []
        self._row: list[_HTMLCell] | None = None
        self._cell_text: list[str] | None = None
        self._cell_attrs: dict[str, str | None] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized = tag.lower()
        if normalized == "tr":
            self._row = []
        elif normalized in {"th", "td"} and self._row is not None:
            self._cell_text = []
            self._cell_attrs = dict(attrs)
        elif normalized == "br" and self._cell_text is not None:
            self._cell_text.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_text is not None:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"th", "td"} and self._cell_text is not None:
            assert self._row is not None
            self._row.append(
                _HTMLCell(
                    text=_clean_cell(" ".join(self._cell_text)),
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


class TableLinearizer:
    """Represent every table as physical rows, columns, and declared spans."""

    _ALIGNMENT_CELL = re.compile(r"^:?-{3,}:?$")

    def linearize(self, table_text: str) -> str:
        if "<table" in table_text.lower():
            parser = _HTMLTableParser()
            parser.feed(table_text)
            if not parser.rows:
                return _clean_cell(re.sub(r"<[^>]+>", " ", table_text))
            return self._format_html_rows(parser.rows)

        rows = self._parse_pipe_table(table_text)
        if not rows:
            return _clean_cell(table_text)
        width = max(len(row) for row in rows)
        html_rows = [
            [_HTMLCell(text=cell) for cell in row + [""] * (width - len(row))]
            for row in rows
        ]
        return self._format_html_rows(html_rows)

    def _parse_pipe_table(self, table_text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in table_text.splitlines():
            stripped = line.strip()
            if not stripped or "|" not in stripped:
                continue
            cells = [_clean_cell(cell) for cell in stripped.strip("|").split("|")]
            if cells and all(self._ALIGNMENT_CELL.fullmatch(cell) for cell in cells):
                continue
            rows.append(cells)
        return rows

    def _format_html_rows(self, rows: list[list[_HTMLCell]]) -> str:
        placed_rows, width = self._place_cells(rows)
        output = [f"Table columns: {width}"]
        for row_index, placed in enumerate(placed_rows):
            cells = "; ".join(self._describe_cell(item) for item in placed)
            output.append(f"Table row {row_index + 1}: {cells}")
        return "\n".join(output)

    @staticmethod
    def _place_cells(
        rows: list[list[_HTMLCell]],
    ) -> tuple[list[list[_PlacedCell]], int]:
        occupied: set[tuple[int, int]] = set()
        placed_rows: list[list[_PlacedCell]] = []
        width = 0
        for row_index, row in enumerate(rows):
            column = 0
            placed: list[_PlacedCell] = []
            for cell in row:
                while (row_index, column) in occupied:
                    column += 1
                item = _PlacedCell(cell=cell, row=row_index, column=column)
                placed.append(item)
                for covered_row in range(row_index, row_index + cell.rowspan):
                    for covered_column in range(column, column + cell.colspan):
                        occupied.add((covered_row, covered_column))
                width = max(width, column + cell.colspan)
                column += cell.colspan
            placed_rows.append(placed)
        return placed_rows, width

    @staticmethod
    def _describe_cell(item: _PlacedCell) -> str:
        cell = item.cell
        column_start = item.column + 1
        column_end = column_start + cell.colspan - 1
        if column_start == column_end:
            location = f"column {column_start}"
        else:
            location = f"columns {column_start}-{column_end}"
        if cell.rowspan > 1:
            row_start = item.row + 1
            row_end = row_start + cell.rowspan - 1
            location += f", rows {row_start}-{row_end}"
        escaped = cell.text.replace('"', '\\"')
        return f'{location}="{escaped}"'

def _positive_span(value: str | None) -> int:
    try:
        parsed = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


def _clean_cell(value: str) -> str:
    normalized = re.sub(r"\\[rn]", " ", html.unescape(value))
    return re.sub(r"\s+", " ", normalized).strip()
