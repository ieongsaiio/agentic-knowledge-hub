"""Parse an HTML table into a rectangular text grid."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass(frozen=True)
class ParsedHTMLTable:
    """Rectangular text grid obtained after expanding declared cell spans."""

    rows: tuple[tuple[str, ...], ...]
    width: int


@dataclass(frozen=True)
class _Cell:
    text: str
    rowspan: int
    colspan: int


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_Cell]] = []
        self._table_depth = 0
        self._capturing = False
        self._row: list[_Cell] | None = None
        self._cell_text: list[str] | None = None
        self._cell_attrs: dict[str, str | None] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        normalized = tag.casefold()
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
        normalized = tag.casefold()
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
                _Cell(
                    text=" ".join(" ".join(self._cell_text).split()),
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


def _positive_span(value: str | None) -> int:
    try:
        parsed = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, parsed)


class HTMLTableParser:
    """Read the first HTML table and expand rowspan/colspan values."""

    def parse(self, table_html: str) -> ParsedHTMLTable:
        parser = _Parser()
        parser.feed(table_html)
        parser.close()
        if not parser.rows:
            raise ValueError("Input does not contain a non-empty HTML table")

        grid: list[list[str]] = []
        active: dict[int, tuple[str, int]] = {}
        for source_row in parser.rows:
            row: list[str] = []
            column = 0

            def consume_active() -> None:
                nonlocal column
                while column in active:
                    value, remaining = active[column]
                    row.append(value)
                    if remaining <= 1:
                        del active[column]
                    else:
                        active[column] = (value, remaining - 1)
                    column += 1

            consume_active()
            for cell in source_row:
                consume_active()
                for _ in range(cell.colspan):
                    row.append(cell.text)
                    if cell.rowspan > 1:
                        active[column] = (cell.text, cell.rowspan - 1)
                    column += 1
                consume_active()
            grid.append(row)

        width = max(len(row) for row in grid)
        return ParsedHTMLTable(
            rows=tuple(tuple(row + [""] * (width - len(row))) for row in grid),
            width=width,
        )

    def visible_text(self, table_html: str) -> str:
        """Render table cells as stable row text without indexing HTML markup."""
        parsed = self.parse(table_html)
        lines: list[str] = []
        for index, row in enumerate(parsed.rows, start=1):
            cells = [cell.strip() for cell in row]
            if not any(cells):
                continue
            lines.append(f"Table row {index}: " + " | ".join(cells))
        return "\n".join(lines)


__all__ = ["HTMLTableParser", "ParsedHTMLTable"]
