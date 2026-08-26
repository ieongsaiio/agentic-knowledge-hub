"""Provider-neutral contracts for structured document parsing results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric or None")
    return float(value)


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list of strings")
    return list(value)


def _json_copy(value: Any, name: str) -> Any:
    copied = deepcopy(value)
    try:
        json.dumps(copied)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable") from exc
    return copied


@dataclass
class ParsedBlock:
    """A normalized parser block whose page index is always 0-based."""

    block_id: str
    type: str
    content: str
    page_index: int
    bbox: list[float] | None = None
    order: int | None = None
    level: int | None = None
    caption: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("block_id must be a non-empty string")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("type must be a non-empty string")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        self.page_index = _require_non_negative_int(self.page_index, "page_index")

        if self.bbox is not None:
            if not isinstance(self.bbox, list) or len(self.bbox) != 4:
                raise ValueError("bbox must contain exactly four coordinates")
            coordinates = [_optional_number(value, "bbox coordinate") for value in self.bbox]
            if any(value is None for value in coordinates):
                raise TypeError("bbox coordinates cannot be None")
            self.bbox = [float(value) for value in coordinates if value is not None]

        if self.order is not None:
            self.order = _require_non_negative_int(self.order, "order")
        if self.level is not None:
            self.level = _require_non_negative_int(self.level, "level")

        self.caption = _string_list(self.caption, "caption")
        self.footnotes = _string_list(self.footnotes, "footnotes")
        if not isinstance(self.images, list) or any(
            not isinstance(image, Mapping) for image in self.images
        ):
            raise TypeError("images must be a list of mappings")
        self.images = _json_copy(self.images, "images")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        self.metadata = _json_copy(dict(self.metadata), "metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "type": self.type,
            "content": self.content,
            "page_index": self.page_index,
            "bbox": deepcopy(self.bbox),
            "order": self.order,
            "level": self.level,
            "caption": list(self.caption),
            "footnotes": list(self.footnotes),
            "images": deepcopy(self.images),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParsedBlock:
        payload = _require_mapping(data, "ParsedBlock")
        allowed = {
            "block_id",
            "type",
            "content",
            "page_index",
            "bbox",
            "order",
            "level",
            "caption",
            "footnotes",
            "images",
            "metadata",
        }
        _reject_unknown(payload, allowed, "ParsedBlock")
        required = {"block_id", "type", "content", "page_index"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"ParsedBlock is missing required fields: {sorted(missing)}")
        return cls(
            block_id=payload["block_id"],
            type=payload["type"],
            content=payload["content"],
            page_index=payload["page_index"],
            bbox=payload.get("bbox"),
            order=payload.get("order"),
            level=payload.get("level"),
            caption=payload.get("caption", []),
            footnotes=payload.get("footnotes", []),
            images=payload.get("images", []),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class ParsedPage:
    """A physical source page containing normalized blocks."""

    page_index: int
    blocks: list[ParsedBlock] = field(default_factory=list)
    width: float | None = None
    height: float | None = None

    def __post_init__(self) -> None:
        self.page_index = _require_non_negative_int(self.page_index, "page_index")
        self.width = _optional_number(self.width, "width")
        self.height = _optional_number(self.height, "height")
        if not isinstance(self.blocks, list) or any(
            not isinstance(block, ParsedBlock) for block in self.blocks
        ):
            raise TypeError("blocks must be a list of ParsedBlock values")
        for block in self.blocks:
            if block.page_index != self.page_index:
                raise ValueError(
                    f"block {block.block_id!r} page_index does not match its ParsedPage"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParsedPage:
        payload = _require_mapping(data, "ParsedPage")
        allowed = {"page_index", "width", "height", "blocks"}
        _reject_unknown(payload, allowed, "ParsedPage")
        if "page_index" not in payload:
            raise ValueError("ParsedPage is missing required field: page_index")
        blocks = payload.get("blocks", [])
        if not isinstance(blocks, list):
            raise TypeError("blocks must be a list")
        return cls(
            page_index=payload["page_index"],
            width=payload.get("width"),
            height=payload.get("height"),
            blocks=[ParsedBlock.from_dict(block) for block in blocks],
        )


@dataclass
class ParsedDocument:
    """Provider-neutral structured output produced by a parser adapter."""

    schema_version: int
    provider: str
    pages: list[ParsedPage] = field(default_factory=list)
    parser_version: str | None = None
    raw_markdown: str | None = None
    raw_artifact: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.schema_version = _require_non_negative_int(self.schema_version, "schema_version")
        if self.schema_version == 0:
            raise ValueError("schema_version must be greater than zero")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty string")
        if self.parser_version is not None and not isinstance(self.parser_version, str):
            raise TypeError("parser_version must be a string or None")
        if self.raw_markdown is not None and not isinstance(self.raw_markdown, str):
            raise TypeError("raw_markdown must be a string or None")
        if not isinstance(self.pages, list) or any(
            not isinstance(page, ParsedPage) for page in self.pages
        ):
            raise TypeError("pages must be a list of ParsedPage values")
        indices = [page.page_index for page in self.pages]
        if len(indices) != len(set(indices)):
            raise ValueError("ParsedDocument page_index values must be unique")
        if self.raw_artifact is not None:
            if not isinstance(self.raw_artifact, Mapping):
                raise TypeError("raw_artifact must be a mapping or None")
            self.raw_artifact = _json_copy(dict(self.raw_artifact), "raw_artifact")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "parser_version": self.parser_version,
            "pages": [page.to_dict() for page in self.pages],
            "raw_markdown": self.raw_markdown,
            "raw_artifact": deepcopy(self.raw_artifact),
        }
        _json_copy(payload, "ParsedDocument")
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParsedDocument:
        payload = _require_mapping(data, "ParsedDocument")
        allowed = {
            "schema_version",
            "provider",
            "parser_version",
            "pages",
            "raw_markdown",
            "raw_artifact",
        }
        _reject_unknown(payload, allowed, "ParsedDocument")
        required = {"schema_version", "provider"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"ParsedDocument is missing required fields: {sorted(missing)}")
        pages = payload.get("pages", [])
        if not isinstance(pages, list):
            raise TypeError("pages must be a list")
        return cls(
            schema_version=payload["schema_version"],
            provider=payload["provider"],
            parser_version=payload.get("parser_version"),
            pages=[ParsedPage.from_dict(page) for page in pages],
            raw_markdown=payload.get("raw_markdown"),
            raw_artifact=payload.get("raw_artifact"),
        )
