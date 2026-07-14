"""Persistent cache for deterministic parsed-loader outputs.

The cache stores the expensive PDF-to-Document result before chunking, so
benchmark ablations can reuse MarkItDown/PyMuPDF parsing across different
chunk sizes and retrieval settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

from src.core.settings import resolve_path
from src.core.types import Document

logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ParsedDocumentCache:
    """Store parsed Document objects keyed by file hash and loader config."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/parsed_documents",
        loader_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.cache_dir = resolve_path(cache_dir)
        self.loader_config = dict(loader_config or {})
        self.loader_config.setdefault("cache_schema_version", _CACHE_SCHEMA_VERSION)

    @property
    def config_hash(self) -> str:
        payload = _canonical_json(self.loader_config).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def cache_path(self, file_hash: str) -> Path:
        safe_hash = str(file_hash).strip().lower()
        if not safe_hash:
            raise ValueError("file_hash cannot be empty")
        return self.cache_dir / f"{safe_hash}_{self.config_hash}.json"

    def get(self, file_hash: str, source_path: str | Path | None = None) -> Document | None:
        """Return a cached Document, or None on cache miss/corruption."""
        path = self.cache_path(file_hash)
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("cache_schema_version") != _CACHE_SCHEMA_VERSION:
                return None
            if payload.get("file_hash") != file_hash:
                return None
            if payload.get("loader_config") != self.loader_config:
                return None

            document_data = payload.get("document")
            if not isinstance(document_data, dict):
                return None
            document = Document.from_dict(document_data)
            if source_path is not None:
                document.metadata["source_path"] = str(Path(source_path))
            logger.info("Parsed document cache hit: %s", path)
            return document
        except Exception as exc:
            logger.warning("Ignoring unreadable parsed document cache %s: %s", path, exc)
            return None

    def put(self, file_hash: str, document: Document) -> Path:
        """Persist a parsed Document and return the cache path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(file_hash)
        payload = {
            "cache_schema_version": _CACHE_SCHEMA_VERSION,
            "file_hash": file_hash,
            "loader_config": self.loader_config,
            "document": document.to_dict(),
        }

        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
        logger.info("Parsed document cache stored: %s", path)
        return path
