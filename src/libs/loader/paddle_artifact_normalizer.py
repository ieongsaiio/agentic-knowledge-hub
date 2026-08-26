"""Normalize PaddleOCR artifacts into the provider-neutral parse contract."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage

_BLOCK_TYPE_MAP = {
    "paragraph": "text",
    "text": "text",
    "paragraph_title": "title",
    "doc_title": "title",
    "document_title": "title",
    "title": "title",
    "table_title": "title",
    "figure_title": "title",
    "table": "table",
    "list": "list",
    "ordered_list": "list",
    "unordered_list": "list",
    "code": "code",
    "code_block": "code",
    "formula": "equation",
    "equation": "equation",
    "display_formula": "equation",
    "inline_formula": "equation",
    "image": "image",
    "figure": "image",
    "chart": "image",
    "header_image": "image",
    "footer_image": "image",
    "header": "page_header",
    "page_header": "page_header",
    "footer": "page_footer",
    "page_footer": "page_footer",
    "number": "page_number",
    "page_number": "page_number",
    "aside": "page_aside_text",
    "aside_text": "page_aside_text",
    "page_aside_text": "page_aside_text",
    "vision_footnote": "page_footnote",
    "footnote": "page_footnote",
    "page_footnote": "page_footnote",
}
_CAPTION_LABELS = {"table_title", "figure_title"}
_FOOTNOTE_LABELS = {"vision_footnote", "footnote"}
_KNOWN_BLOCK_FIELDS = {
    "block_label",
    "block_content",
    "block_bbox",
    "block_order",
    "block_level",
    "level",
}


class PaddleArtifactNormalizer:
    """Convert the shared Paddle ``pages + restructured_pages`` artifact."""

    def normalize(self, artifact: Mapping[str, Any]) -> ParsedDocument:
        if not isinstance(artifact, Mapping):
            raise TypeError("Paddle artifact must be a mapping")

        restructured_pages = artifact.get("restructured_pages")
        if not isinstance(restructured_pages, list):
            raise ValueError("Paddle artifact restructured_pages must be a list")

        raw_pages = artifact.get("pages")
        if not isinstance(raw_pages, list):
            raw_pages = []
        raw_by_index = self._raw_pages_by_index(raw_pages)

        normalized_pages: list[ParsedPage] = []
        markdown_by_page: list[tuple[int, str]] = []
        for fallback_index, value in enumerate(restructured_pages):
            if not isinstance(value, Mapping):
                raise ValueError(f"Paddle restructured_pages[{fallback_index}] must be a mapping")
            page_index = self._page_index(value, fallback_index)
            markdown = value.get("markdown_text", "")
            if not isinstance(markdown, str):
                raise ValueError(f"Paddle page {page_index} markdown_text must be a string")

            raw_page = raw_by_index.get(page_index)
            if raw_page is None and fallback_index < len(raw_pages):
                candidate = raw_pages[fallback_index]
                raw_page = candidate if isinstance(candidate, Mapping) else None
            blocks = self._parsing_blocks(raw_page)
            normalized_blocks = self._normalize_blocks(blocks, page_index)
            if not normalized_blocks:
                normalized_blocks = [
                    ParsedBlock(
                        block_id=f"p{page_index}_fallback",
                        type="text",
                        content=markdown,
                        page_index=page_index,
                        order=0,
                        metadata={
                            "fallback": True,
                            "source_block_label": "markdown_page",
                        },
                    )
                ]

            width, height = self._page_dimensions(raw_page)
            normalized_pages.append(
                ParsedPage(
                    page_index=page_index,
                    width=width,
                    height=height,
                    blocks=normalized_blocks,
                )
            )
            markdown_by_page.append((page_index, markdown))

        normalized_pages.sort(key=lambda page: page.page_index)
        markdown_by_page.sort(key=lambda item: item[0])
        return ParsedDocument(
            schema_version=1,
            provider="paddle",
            parser_version=self._parser_version(artifact),
            pages=normalized_pages,
            raw_markdown="\n\n".join(markdown for _, markdown in markdown_by_page),
            raw_artifact=copy.deepcopy(dict(artifact)),
        )

    @staticmethod
    def _raw_pages_by_index(
        raw_pages: Sequence[Any],
    ) -> dict[int, Mapping[str, Any]]:
        indexed: dict[int, Mapping[str, Any]] = {}
        for fallback_index, page in enumerate(raw_pages):
            if not isinstance(page, Mapping):
                continue
            res = page.get("res")
            source = res if isinstance(res, Mapping) else page
            page_index = PaddleArtifactNormalizer._page_index(source, fallback_index)
            indexed.setdefault(page_index, page)
        return indexed

    @staticmethod
    def _page_index(page: Mapping[str, Any], fallback: int) -> int:
        for key in ("page_index", "pageIndex", "page_num", "pageNum"):
            value = page.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return fallback

    @staticmethod
    def _parsing_blocks(raw_page: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
        if raw_page is None:
            return []
        res = raw_page.get("res")
        if not isinstance(res, Mapping):
            return []
        pruned = res.get("prunedResult")
        if not isinstance(pruned, Mapping):
            return []
        blocks = pruned.get("parsing_res_list")
        if not isinstance(blocks, list):
            return []
        return [block for block in blocks if isinstance(block, Mapping)]

    def _normalize_blocks(
        self,
        blocks: Sequence[Mapping[str, Any]],
        page_index: int,
    ) -> list[ParsedBlock]:
        labels = [self._label(block) for block in blocks]
        claimed: set[int] = set()
        table_context: dict[int, tuple[list[str], list[str]]] = {}

        for index, label in enumerate(labels):
            if label != "table":
                continue

            caption_indices: list[int] = []
            cursor = index - 1
            while cursor >= 0 and labels[cursor] in _CAPTION_LABELS:
                caption_indices.append(cursor)
                cursor -= 1
            caption_indices.reverse()

            footnote_indices: list[int] = []
            cursor = index + 1
            while cursor < len(blocks) and labels[cursor] in _FOOTNOTE_LABELS:
                footnote_indices.append(cursor)
                cursor += 1

            captions = [self._content(blocks[item]) for item in caption_indices]
            footnotes = [self._content(blocks[item]) for item in footnote_indices]
            captions = [content for content in captions if content]
            footnotes = [content for content in footnotes if content]
            claimed.update(caption_indices)
            claimed.update(footnote_indices)
            table_context[index] = (captions, footnotes)

        normalized: list[ParsedBlock] = []
        for index, block in enumerate(blocks):
            if index in claimed:
                continue
            label = labels[index]
            block_type = _BLOCK_TYPE_MAP.get(label, "text")
            captions, footnotes = table_context.get(index, ([], []))
            metadata: dict[str, Any] = {
                "source_block_label": label or "unknown",
                "source_block_index": index,
            }
            source_fields = {
                key: copy.deepcopy(value)
                for key, value in block.items()
                if key not in _KNOWN_BLOCK_FIELDS
            }
            if source_fields:
                metadata["source_fields"] = source_fields

            normalized.append(
                ParsedBlock(
                    block_id=f"p{page_index}_b{index}",
                    type=block_type,
                    content=self._content(block),
                    page_index=page_index,
                    bbox=self._bbox(block.get("block_bbox")),
                    order=self._order(block.get("block_order"), index),
                    level=self._level(block),
                    caption=captions,
                    footnotes=footnotes,
                    metadata=metadata,
                )
            )
        return normalized

    @staticmethod
    def _label(block: Mapping[str, Any]) -> str:
        value = block.get("block_label", "")
        return str(value).strip().lower()

    @staticmethod
    def _content(block: Mapping[str, Any]) -> str:
        value = block.get("block_content", "")
        return value if isinstance(value, str) else str(value or "")

    @staticmethod
    def _bbox(value: Any) -> list[float] | None:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            return None
        return [float(item) for item in value]

    @staticmethod
    def _order(value: Any, fallback: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return fallback

    @staticmethod
    def _level(block: Mapping[str, Any]) -> int | None:
        for key in ("block_level", "level"):
            value = block.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    @staticmethod
    def _page_dimensions(
        raw_page: Mapping[str, Any] | None,
    ) -> tuple[float | None, float | None]:
        if raw_page is None:
            return None, None
        res = raw_page.get("res")
        source = res if isinstance(res, Mapping) else raw_page
        width = source.get("page_width", source.get("width"))
        height = source.get("page_height", source.get("height"))
        return (
            float(width)
            if isinstance(width, (int, float)) and not isinstance(width, bool)
            else None,
            float(height)
            if isinstance(height, (int, float)) and not isinstance(height, bool)
            else None,
        )

    @staticmethod
    def _parser_version(artifact: Mapping[str, Any]) -> str | None:
        direct = artifact.get("parser_version")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        config = artifact.get("config")
        if isinstance(config, Mapping):
            for key in ("model", "pipeline_version", "parser_version"):
                value = config.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            nested = config.get("api")
            if isinstance(nested, Mapping):
                model = nested.get("model")
                if isinstance(model, str) and model.strip():
                    return model.strip()
        return None


__all__ = ["PaddleArtifactNormalizer"]
