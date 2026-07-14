"""Build document-level views from flat Chroma chunk records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentCatalogEntry:
    """A logical document reconstructed from chunk metadata."""

    doc_id: str
    title: str
    source_path: str | None = None
    doc_type: str | None = None
    doc_hash: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "chunk_count": self.chunk_count,
            "chunk_ids": self.chunk_ids,
        }
        if self.source_path:
            result["source_path"] = self.source_path
        if self.doc_type:
            result["doc_type"] = self.doc_type
        if self.doc_hash:
            result["doc_hash"] = self.doc_hash
        if self.summary:
            result["summary"] = self.summary
        if self.tags:
            result["tags"] = self.tags
        if self.pages:
            result["pages"] = self.pages
        return result


def build_document_catalog(collection: Any) -> list[DocumentCatalogEntry]:
    """Group a Chroma collection's chunk records by logical document ID."""
    raw = collection.get(include=["metadatas", "documents"])
    ids = raw.get("ids") or []
    metadatas = raw.get("metadatas") or []
    texts = raw.get("documents") or []
    grouped: dict[str, DocumentCatalogEntry] = {}

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        text = texts[index] if index < len(texts) and texts[index] else ""
        doc_id = _resolve_doc_id(metadata, str(chunk_id))

        entry = grouped.get(doc_id)
        if entry is None:
            source_path = _optional_string(metadata.get("source_path") or metadata.get("source"))
            entry = DocumentCatalogEntry(
                doc_id=doc_id,
                title=_resolve_title(metadata, text, source_path),
                source_path=source_path,
                doc_type=_optional_string(metadata.get("doc_type")),
                doc_hash=_optional_string(metadata.get("doc_hash")),
                summary=_optional_string(metadata.get("summary")) or _preview(text),
            )
            grouped[doc_id] = entry

        entry.chunk_ids.append(str(chunk_id))
        entry.tags = _merge_unique(entry.tags, _normalize_tags(metadata.get("tags")))

        page = metadata.get("page") or metadata.get("page_num")
        try:
            page_number = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_number = None
        if page_number is not None and page_number not in entry.pages:
            entry.pages.append(page_number)

    for entry in grouped.values():
        entry.pages.sort()

    return sorted(
        grouped.values(),
        key=lambda entry: (entry.title.lower(), entry.doc_id),
    )


def _resolve_doc_id(metadata: dict[str, Any], chunk_id: str) -> str:
    source_ref = _optional_string(metadata.get("source_ref"))
    if source_ref:
        return source_ref

    doc_hash = _optional_string(metadata.get("doc_hash"))
    if doc_hash:
        return f"doc_{doc_hash[:16]}"

    source_path = _optional_string(metadata.get("source_path") or metadata.get("source"))
    if source_path:
        import hashlib

        digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
        return f"doc_path_{digest}"

    return f"doc_chunk_{chunk_id}"


def _resolve_title(
    metadata: dict[str, Any],
    text: str,
    source_path: str | None,
) -> str:
    title = _optional_string(metadata.get("title"))
    if title:
        return title
    if source_path:
        return Path(source_path).stem
    first_line = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")
    return first_line or "Untitled Document"


def _preview(text: str, max_length: int = 240) -> str | None:
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= max_length else cleaned[: max_length - 3] + "..."


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if not isinstance(value, str) or not value.strip():
        return []

    stripped = value.strip()
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list):
        return [str(tag).strip() for tag in decoded if str(tag).strip()]
    return [tag.strip() for tag in stripped.split(",") if tag.strip()]


def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    merged = existing[:]
    for item in additions:
        if item not in merged:
            merged.append(item)
    return merged


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
