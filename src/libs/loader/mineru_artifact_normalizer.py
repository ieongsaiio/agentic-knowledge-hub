"""Normalize MinerU output artifacts into provider-neutral parsed documents."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage

_PageSize = tuple[float | None, float | None]
_NormalizedPage = tuple[int, list[Mapping[str, Any]], _PageSize | None]

_TYPE_MAP = {
    "title": "title",
    "doc_title": "title",
    "paragraph": "text",
    "text": "text",
    "table": "table",
    "list": "list",
    "index": "list",
    "code": "code",
    "algorithm": "code",
    "equation": "equation",
    "equation_interline": "equation",
    "interline_equation": "equation",
    "image": "image",
    "chart": "chart",
    "header": "page_header",
    "page_header": "page_header",
    "footer": "page_footer",
    "page_footer": "page_footer",
    "page_number": "page_number",
    "aside": "page_aside_text",
    "aside_text": "page_aside_text",
    "page_aside_text": "page_aside_text",
    "page_footnote": "page_footnote",
}

_CONTENT_KEYS = {
    "title": ("title_content", "text", "content"),
    "text": ("paragraph_content", "text", "content"),
    "table": (
        "html",
        "table_body",
        "table_content",
        "body",
        "markdown",
        "text",
    ),
    "list": ("list_items", "items", "text", "content"),
    "code": ("code_content", "code_body", "algorithm_content", "text"),
    "equation": ("math_content", "text", "content"),
    "image": ("image_content", "text", "content"),
    "chart": ("chart_content", "content", "table_body", "text"),
    "page_header": ("page_header_content", "header_content", "text", "content"),
    "page_footer": ("page_footer_content", "footer_content", "text", "content"),
    "page_number": ("page_number_content", "text", "content"),
    "page_aside_text": (
        "page_aside_text_content",
        "aside_text_content",
        "text",
        "content",
    ),
    "page_footnote": ("page_footnote_content", "text", "content"),
}


class MinerUArtifactNormalizer:
    """Convert MinerU v2 or legacy content lists to ``ParsedDocument``."""

    schema_version = 1

    def normalize(self, artifact: Mapping[str, Any]) -> ParsedDocument:
        if not isinstance(artifact, Mapping):
            raise TypeError("MinerU artifact must be a mapping")

        page_sizes = self._middle_page_sizes(artifact.get("middle_json"))
        normalized: list[_NormalizedPage] | None = self._normalize_v2(
            artifact.get("content_list_v2")
        )
        if normalized is None:
            normalized = self._normalize_legacy(artifact.get("content_list"))
        assert normalized is not None

        pages_by_index: dict[int, list[tuple[ParsedBlock, bool, int]]] = defaultdict(list)
        page_overrides: dict[int, tuple[float | None, float | None]] = {}
        for page_index, raw_blocks, page_size in normalized:
            if page_size is not None:
                page_overrides[page_index] = page_size
            converted: list[tuple[ParsedBlock, bool, int]] = []
            for sequence, raw_block in enumerate(raw_blocks):
                block, has_explicit_order = self._convert_block(
                    raw_block,
                    page_index=page_index,
                    sequence=sequence,
                )
                converted.append((block, has_explicit_order, sequence))
            if converted and all(explicit for _, explicit, _ in converted):
                converted.sort(key=lambda value: (value[0].order, value[2]))
            pages_by_index[page_index].extend(converted)

        all_page_indices = sorted(set(page_sizes) | set(page_overrides) | set(pages_by_index))
        pages: list[ParsedPage] = []
        for page_index in all_page_indices:
            width, height = page_overrides.get(
                page_index,
                page_sizes.get(page_index, (None, None)),
            )
            pages.append(
                ParsedPage(
                    page_index=page_index,
                    width=width,
                    height=height,
                    blocks=[item[0] for item in pages_by_index.get(page_index, [])],
                )
            )

        version = artifact.get("version")
        parser_version = str(version) if version is not None else None
        markdown = artifact.get("full_markdown")
        if markdown is not None and not isinstance(markdown, str):
            raise TypeError("MinerU artifact full_markdown must be a string or None")
        return ParsedDocument(
            schema_version=self.schema_version,
            provider="mineru",
            parser_version=parser_version,
            pages=pages,
            raw_markdown=markdown,
            raw_artifact=copy.deepcopy(dict(artifact)),
        )

    def _normalize_v2(
        self,
        value: Any,
    ) -> list[_NormalizedPage] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get("pages")
        if not isinstance(value, list) or not value:
            return None

        pages: list[_NormalizedPage] = []
        # Current v2 normally uses one nested block list per physical page.
        if all(isinstance(page, list) for page in value):
            for page_index, blocks in enumerate(value):
                pages.append((page_index, self._block_mappings(blocks), None))
            return pages

        # Some builds wrap each page in an object carrying page_idx/page_size.
        if all(isinstance(page, Mapping) and "type" not in page for page in value):
            for fallback_index, page in enumerate(value):
                page_index = self._page_index(page.get("page_idx", fallback_index))
                blocks = page.get("blocks", page.get("items", page.get("content", [])))
                pages.append(
                    (
                        page_index,
                        self._block_mappings(blocks),
                        self._page_size(page.get("page_size")),
                    )
                )
            return pages

        # Tolerate a flattened v2 list if every item supplies page_idx.
        if all(isinstance(block, Mapping) and "type" in block for block in value):
            grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for block in value:
                page_index = self._page_index(block.get("page_idx"))
                grouped[page_index].append(block)
            return [(index, grouped[index], None) for index in sorted(grouped)]
        return None

    def _normalize_legacy(
        self,
        value: Any,
    ) -> list[_NormalizedPage]:
        if not isinstance(value, list):
            raise ValueError(
                "MinerU artifact has no parseable content_list_v2 or content_list"
            )
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for block in self._block_mappings(value):
            page_index = self._page_index(block.get("page_idx"))
            grouped[page_index].append(block)
        return [(index, grouped[index], None) for index in sorted(grouped)]

    def _convert_block(
        self,
        raw: Mapping[str, Any],
        *,
        page_index: int,
        sequence: int,
    ) -> tuple[ParsedBlock, bool]:
        source_type = str(raw.get("type", "text")).strip().lower() or "text"
        normalized_type = _TYPE_MAP.get(source_type, "text")
        payload = raw.get("content")
        structured = payload if isinstance(payload, Mapping) else raw

        level = self._optional_non_negative_int(
            structured.get("level", raw.get("text_level")), "level"
        )
        if source_type == "text" and level is not None and level > 0:
            normalized_type = "title"

        content = self._block_content(normalized_type, raw, structured)
        caption = self._caption(normalized_type, raw, structured)
        footnotes = self._footnotes(normalized_type, raw, structured)
        images = self._images(raw, structured)
        explicit_order_value = raw.get("order", raw.get("index"))
        explicit_order = self._optional_non_negative_int(
            explicit_order_value, "order"
        )
        order = sequence if explicit_order is None else explicit_order
        metadata: dict[str, Any] = {"source_type": source_type}
        sub_type = raw.get("sub_type", structured.get("sub_type"))
        if source_type == "algorithm" and not sub_type:
            sub_type = "algorithm"
        if isinstance(sub_type, str) and sub_type:
            metadata["sub_type"] = sub_type
        anchor = raw.get("anchor")
        if isinstance(anchor, str) and anchor:
            metadata["anchor"] = anchor
        language = structured.get("code_language", raw.get("code_language"))
        if isinstance(language, str) and language:
            metadata["language"] = language

        block = ParsedBlock(
            block_id=f"mineru-p{page_index:04d}-b{sequence:04d}",
            type=normalized_type,
            content=content,
            page_index=page_index,
            bbox=self._bbox(raw.get("bbox")),
            order=order,
            level=level,
            caption=caption,
            footnotes=footnotes,
            images=images,
            metadata=metadata,
        )
        return block, explicit_order is not None

    def _block_content(
        self,
        block_type: str,
        raw: Mapping[str, Any],
        structured: Mapping[str, Any],
    ) -> str:
        if block_type == "list":
            value = self._first_value(structured, raw, _CONTENT_KEYS[block_type])
            if isinstance(value, list):
                return "\n".join(
                    text for text in (self._inline_text(item) for item in value) if text
                )
            return self._inline_text(value)
        value = self._first_value(
            structured,
            raw,
            _CONTENT_KEYS.get(block_type, ("text", "content")),
        )
        if value is None and not isinstance(raw.get("content"), Mapping):
            value = raw.get("content")
        return self._inline_text(value)

    def _caption(
        self,
        block_type: str,
        raw: Mapping[str, Any],
        structured: Mapping[str, Any],
    ) -> list[str]:
        prefixes = [block_type]
        if block_type == "code" and str(raw.get("type", "")).lower() == "algorithm":
            prefixes.insert(0, "algorithm")
        for prefix in prefixes:
            value = self._first_value(
                structured,
                raw,
                (f"{prefix}_caption", "caption"),
            )
            if value is not None:
                return self._text_list(value)
        return []

    def _footnotes(
        self,
        block_type: str,
        raw: Mapping[str, Any],
        structured: Mapping[str, Any],
    ) -> list[str]:
        prefixes = [block_type]
        if block_type == "code" and str(raw.get("type", "")).lower() == "algorithm":
            prefixes.insert(0, "algorithm")
        for prefix in prefixes:
            value = self._first_value(
                structured,
                raw,
                (f"{prefix}_footnote", "footnotes", "footnote"),
            )
            if value is not None:
                return self._text_list(value)
        return []

    @staticmethod
    def _images(
        raw: Mapping[str, Any], structured: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        image_source = structured.get("image_source", raw.get("image_source"))
        if isinstance(image_source, Mapping):
            path = image_source.get("path")
            if isinstance(path, str) and path:
                return [{"path": path}]
        for key in ("img_path", "image_path"):
            value = structured.get(key, raw.get(key))
            if isinstance(value, str) and value:
                return [{"path": value}]
        return []

    @classmethod
    def _inline_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, list):
            return "".join(cls._inline_text(item) for item in value)
        if isinstance(value, Mapping):
            content = value.get("content")
            if content is not None:
                return cls._inline_text(content)
            children = value.get("children")
            if children is not None:
                return cls._inline_text(children)
            for key in ("text", "value", "math_content"):
                if key in value:
                    return cls._inline_text(value[key])
        return ""

    @classmethod
    def _text_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if not isinstance(value, list):
            text = cls._inline_text(value)
            return [text] if text else []
        if value and all(
            isinstance(item, Mapping)
            and str(item.get("type", "")).lower()
            in {"text", "hyperlink", "inline_equation", "equation"}
            for item in value
        ):
            text = cls._inline_text(value)
            return [text] if text else []
        return [text for text in (cls._inline_text(item) for item in value) if text]

    @staticmethod
    def _first_value(
        primary: Mapping[str, Any],
        fallback: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Any:
        for key in keys:
            if key in primary and primary[key] is not None:
                return primary[key]
            if key in fallback and fallback[key] is not None:
                return fallback[key]
        return None

    @staticmethod
    def _block_mappings(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, list):
            raise ValueError("MinerU content page must contain a list of blocks")
        if any(not isinstance(block, Mapping) for block in value):
            raise ValueError("MinerU content blocks must be objects")
        return list(value)

    @staticmethod
    def _page_index(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("MinerU page_idx must be a non-negative integer")
        return value

    @staticmethod
    def _optional_non_negative_int(value: Any, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"MinerU {name} must be a non-negative integer")
        return value

    @staticmethod
    def _bbox(value: Any) -> list[float] | None:
        if value is None:
            return None
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
        ):
            raise ValueError("MinerU bbox must contain four numeric coordinates")
        return [float(item) for item in value]

    @classmethod
    def _page_size(cls, value: Any) -> tuple[float | None, float | None] | None:
        if value is None:
            return None
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
        ):
            raise ValueError("MinerU page_size must contain width and height")
        return float(value[0]), float(value[1])

    @classmethod
    def _middle_page_sizes(
        cls, middle: Any
    ) -> dict[int, tuple[float | None, float | None]]:
        if not isinstance(middle, Mapping):
            return {}
        pdf_info = middle.get("pdf_info")
        if not isinstance(pdf_info, list):
            return {}
        result: dict[int, tuple[float | None, float | None]] = {}
        for page in pdf_info:
            if not isinstance(page, Mapping):
                continue
            page_index = cls._page_index(page.get("page_idx"))
            result[page_index] = cls._page_size(page.get("page_size")) or (None, None)
        return result


__all__ = ["MinerUArtifactNormalizer"]
