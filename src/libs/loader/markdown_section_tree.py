"""Build a hierarchical Section tree from normalized Markdown text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from markdown_it import MarkdownIt

_SECTION_TREE_SCHEMA_VERSION = 1


def build_markdown_section_tree(
    text: str,
    *,
    document_id: str,
    page_spans: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse Markdown headings into Section nodes with exact source offsets.

    Each node's ``content`` contains only the heading and body that directly
    belong to that Section. Descendant Section text is represented only in
    ``subsections`` and in the node's wider ``start_offset:end_offset`` range.
    Retrieval children are intentionally outside this loader-stage contract.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")

    line_offsets = _line_offsets(text)
    heading_tokens = _heading_tokens(text)
    headings = [
        {
            "start": line_offsets[token.map[0]],
            "level": int(token.tag[1:]),
            "title": _heading_title(tokens, index),
        }
        for tokens, index, token in heading_tokens
    ]

    root_content_end = headings[0]["start"] if headings else len(text)
    root: dict[str, Any] = {
        "id": f"{document_id}_section_root",
        "parent_section_id": None,
        "heading_title": None,
        "heading_depth": 0,
        "sequence_number": 0,
        "path": [],
        "start_offset": 0,
        "end_offset": len(text),
        "content_start_offset": 0,
        "content_end_offset": root_content_end,
        "content": text[:root_content_end],
        "subsections": [],
    }
    _add_page_metadata(root, page_spans)

    stack: list[dict[str, Any]] = [root]
    flat_sections: list[dict[str, Any]] = []
    for sequence_number, heading in enumerate(headings, start=1):
        level = heading["level"]
        start = heading["start"]
        while len(stack) > 1 and stack[-1]["heading_depth"] >= level:
            stack.pop()

        parent = stack[-1]
        path = [*parent["path"], heading["title"]]
        section = {
            "id": f"{document_id}_section_{sequence_number:04d}",
            "parent_section_id": parent["id"],
            "heading_title": heading["title"],
            "heading_depth": level,
            "sequence_number": sequence_number,
            "path": path,
            "start_offset": start,
            "end_offset": len(text),
            "content_start_offset": start,
            "content_end_offset": len(text),
            "content": "",
            "subsections": [],
        }
        parent["subsections"].append(section)
        flat_sections.append(section)
        stack.append(section)

    for index, section in enumerate(flat_sections):
        next_start = (
            flat_sections[index + 1]["start_offset"]
            if index + 1 < len(flat_sections)
            else len(text)
        )
        section["content_end_offset"] = next_start
        section["content"] = text[section["content_start_offset"]:next_start]

        end_offset = len(text)
        for candidate in flat_sections[index + 1:]:
            if candidate["heading_depth"] <= section["heading_depth"]:
                end_offset = candidate["start_offset"]
                break
        section["end_offset"] = end_offset
        _add_page_metadata(section, page_spans)

    return {
        "schema_version": _SECTION_TREE_SCHEMA_VERSION,
        "document_id": document_id,
        "section_count": len(flat_sections),
        "root": root,
    }


def _heading_tokens(
    text: str,
) -> list[tuple[list[Any], int, Any]]:
    tokens = MarkdownIt("commonmark", {"html": True}).parse(text)
    return [
        (tokens, index, token)
        for index, token in enumerate(tokens)
        if token.type == "heading_open"
        and token.map is not None
        and token.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}
    ]


def _heading_title(tokens: list[Any], heading_index: int) -> str:
    if heading_index + 1 >= len(tokens):
        return ""
    inline = tokens[heading_index + 1]
    if inline.type != "inline":
        return ""
    return inline.content.strip()


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    cursor = 0
    for line in text.splitlines(keepends=True):
        cursor += len(line)
        offsets.append(cursor)
    if offsets[-1] != len(text):
        offsets.append(len(text))
    return offsets


def _add_page_metadata(
    section: dict[str, Any],
    page_spans: Sequence[Mapping[str, Any]] | None,
) -> None:
    if not page_spans:
        return
    start = section["start_offset"]
    end = section["end_offset"]
    pages: list[int] = []
    for span in page_spans:
        span_start = span.get("start_offset")
        span_end = span.get("end_offset")
        if (
            isinstance(span_start, int)
            and not isinstance(span_start, bool)
            and isinstance(span_end, int)
            and not isinstance(span_end, bool)
            and start < span_end
            and end > span_start
        ):
            page = span.get("page")
            if isinstance(page, int) and not isinstance(page, bool):
                pages.append(page)
                continue
            page_start = span.get("page_start")
            page_end = span.get("page_end")
            if (
                isinstance(page_start, int)
                and not isinstance(page_start, bool)
                and isinstance(page_end, int)
                and not isinstance(page_end, bool)
                and page_start <= page_end
            ):
                pages.extend(range(page_start, page_end + 1))

    if not pages:
        return
    section["page_start"] = min(pages)
    section["page_end"] = max(pages)
    if section["page_start"] == section["page_end"]:
        section["page_num"] = section["page_start"]
