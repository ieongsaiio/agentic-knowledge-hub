"""Build canonical Markdown Documents from provider-neutral parsed blocks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from src.core.types import Document
from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage


class CanonicalDocumentAssembler:
    """Render normalized parser output into one stable Markdown document."""

    def __init__(self, ignored_block_types: Iterable[str] | None = None) -> None:
        self.ignored_block_types = set(ignored_block_types or ())

    def assemble(
        self,
        parsed_document: ParsedDocument,
        *,
        source_path: str,
        doc_id: str,
        doc_hash: str | None = None,
        doc_type: str = "pdf",
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Assemble Markdown and exact source-page/block character spans."""
        if not isinstance(parsed_document, ParsedDocument):
            raise TypeError("parsed_document must be a ParsedDocument")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("source_path must be a non-empty string")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(doc_type, str) or not doc_type:
            raise ValueError("doc_type must be a non-empty string")

        output = ""
        page_spans: list[dict[str, int]] = []
        structure_pages: list[dict[str, Any]] = []
        structure_blocks: list[dict[str, Any]] = []

        pages = sorted(parsed_document.pages, key=lambda page: page.page_index)
        has_rendered_page = False
        for page in pages:
            rendered_blocks = self._render_page_blocks(page)
            page_text = "\n\n".join(text for _, text in rendered_blocks)

            if page_text and has_rendered_page:
                output += "\n\n"
            page_start = len(output)

            page_block_records: list[dict[str, Any]] = []
            for index, (block, rendered) in enumerate(rendered_blocks):
                if index:
                    output += "\n\n"
                block_start = len(output)
                output += rendered
                record = block.to_dict()
                record["start_offset"] = block_start
                record["end_offset"] = len(output)
                page_block_records.append(record)
                structure_blocks.append(record)

            page_end = len(output)
            page_spans.append(
                {
                    "page": page.page_index + 1,
                    "page_index": page.page_index,
                    "start_offset": page_start,
                    "end_offset": page_end,
                }
            )
            structure_pages.append(
                {
                    "page_index": page.page_index,
                    "width": page.width,
                    "height": page.height,
                    "start_offset": page_start,
                    "end_offset": page_end,
                    "blocks": page_block_records,
                }
            )
            has_rendered_page = has_rendered_page or bool(page_text)

        document_metadata = deepcopy(metadata or {})
        document_metadata.update(
            {
                "source_path": source_path,
                "doc_type": doc_type,
                "pdf": doc_type.lower() == "pdf",
                "page_count": len(pages),
                "page_spans": page_spans,
                "parsed_structure": {
                    "schema_version": parsed_document.schema_version,
                    "provider": parsed_document.provider,
                    "parser_version": parsed_document.parser_version,
                    "pages": structure_pages,
                    "blocks": structure_blocks,
                },
                "parser_provider": parsed_document.provider,
                "parser_version": parsed_document.parser_version,
            }
        )
        if doc_hash is not None:
            document_metadata["doc_hash"] = doc_hash
        if parsed_document.raw_artifact is not None:
            document_metadata["parsed_source_artifact"] = deepcopy(
                parsed_document.raw_artifact
            )
        if parsed_document.raw_markdown is not None:
            document_metadata["parsed_source_markdown"] = parsed_document.raw_markdown

        return Document(id=doc_id, text=output, metadata=document_metadata)

    def _render_page_blocks(
        self,
        page: ParsedPage,
    ) -> list[tuple[ParsedBlock, str]]:
        indexed = list(enumerate(page.blocks))
        indexed.sort(
            key=lambda item: (
                item[1].order is None,
                item[1].order if item[1].order is not None else item[0],
                item[0],
            )
        )
        rendered: list[tuple[ParsedBlock, str]] = []
        for _, block in indexed:
            if block.type in self.ignored_block_types:
                continue
            text = self._render_block(block)
            if text:
                rendered.append((block, text))
        return rendered

    @staticmethod
    def _render_block(block: ParsedBlock) -> str:
        content = block.content.strip("\n")
        if block.type == "title":
            if re.match(r"^#{1,6}(?:\s|$)", content):
                return content
            level = min(max(block.level or 1, 1), 6)
            return f"{'#' * level} {content}" if content else ""

        if block.type == "table":
            parts = [*block.caption, content, *block.footnotes]
            return "\n\n".join(part.strip("\n") for part in parts if part.strip("\n"))

        if block.type == "list":
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if not lines:
                return ""
            if all(re.match(r"^(?:[-+*]|\d+[.)])\s+", line) for line in lines):
                return "\n".join(lines)
            return "\n".join(f"- {line}" for line in lines)

        if block.type == "code":
            if content.lstrip().startswith("```"):
                return content
            language = block.metadata.get("language", "")
            language = language if isinstance(language, str) else ""
            return f"```{language}\n{content}\n```" if content else ""

        if block.type == "equation":
            if content.startswith("$$") and content.endswith("$$"):
                return content
            return f"$$\n{content}\n$$" if content else ""

        return content


def assemble_canonical_document(
    parsed_document: ParsedDocument,
    *,
    source_path: str,
    doc_id: str,
    doc_hash: str | None = None,
    doc_type: str = "pdf",
    metadata: dict[str, Any] | None = None,
    ignored_block_types: Iterable[str] | None = None,
) -> Document:
    """Convenience wrapper for one-off canonical document assembly."""
    return CanonicalDocumentAssembler(ignored_block_types).assemble(
        parsed_document,
        source_path=source_path,
        doc_id=doc_id,
        doc_hash=doc_hash,
        doc_type=doc_type,
        metadata=metadata,
    )
