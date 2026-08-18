"""PDF loader backed by MinerU's provider-neutral structured artifact."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import re
from pathlib import Path
from typing import Any

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.canonical_document_assembler import CanonicalDocumentAssembler
from src.libs.loader.mineru_artifact_normalizer import MinerUArtifactNormalizer
from src.libs.loader.table_continuation_merger import TableContinuationMerger

_DEFAULT_IGNORED_BLOCK_TYPES = {
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
}
_RUNTIME_API_KEYS = {
    "api_key",
    "token_env",
    "poll_interval_seconds",
    "timeout_seconds",
    "request_timeout_seconds",
    "callback",
    "seed",
}


class MineruPdfLoader(BaseLoader):
    """Parse a PDF with MinerU and assemble one canonical Markdown Document."""

    def __init__(
        self,
        mineru_config: dict[str, Any] | None = None,
        extract_images: bool = False,
        image_storage_dir: str | Path = "data/images",
        client: Any | None = None,
    ) -> None:
        self.mineru_config = copy.deepcopy(mineru_config or {})
        backend = str(self.mineru_config.get("backend", "api")).strip().lower()
        if backend != "api":
            raise ValueError("MinerU loader currently supports only the api backend")
        api_config = self.mineru_config.get("api", {})
        if not isinstance(api_config, dict):
            raise ValueError("MinerU mineru.api must be a mapping")
        self.api_config = copy.deepcopy(api_config)
        grouping = self.mineru_config.get("table_grouping", {}) or {}
        if not isinstance(grouping, dict):
            raise ValueError("MinerU table_grouping must be a mapping")
        self.table_grouping_enabled = bool(grouping.get("enabled", True))
        self.table_grouping_config = {
            "minimum_horizontal_overlap": float(
                grouping.get("minimum_horizontal_overlap", 0.85)
            ),
            "maximum_vertical_gap_ratio": float(
                grouping.get("maximum_vertical_gap_ratio", 0.08)
            ),
        }
        ignored = self.mineru_config.get("ignored_block_types")
        if ignored is None:
            self.ignored_block_types = set(_DEFAULT_IGNORED_BLOCK_TYPES)
        elif not isinstance(ignored, list) or any(
            not isinstance(value, str) for value in ignored
        ):
            raise ValueError("MinerU ignored_block_types must be a list of strings")
        else:
            self.ignored_block_types = {value.strip() for value in ignored if value.strip()}
        self.extract_images = bool(extract_images)
        if not self.extract_images:
            self.ignored_block_types.add("image")
        self.image_storage_dir = Path(image_storage_dir)
        self._client = client
        self._normalizer = MinerUArtifactNormalizer()

    def cache_config(self) -> dict[str, Any]:
        """Return output-affecting MinerU settings without secrets or timings."""
        api = {
            key: copy.deepcopy(value)
            for key, value in self.api_config.items()
            if key not in _RUNTIME_API_KEYS and key != "base_url"
        }
        api.setdefault("model_version", "vlm")
        return {
            "loader": "MineruPdfLoader",
            "loader_schema_version": 2,
            "provider": "mineru",
            "backend": "api",
            "extract_images": self.extract_images,
            "ignored_block_types": sorted(self.ignored_block_types),
            "table_grouping": {
                "enabled": self.table_grouping_enabled,
                **self.table_grouping_config,
            },
            "api": api,
        }

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")
        doc_hash = self._compute_file_hash(path)
        artifact = self._client_instance().run(path)
        parsed = self._normalizer.normalize(artifact)
        if self.table_grouping_enabled:
            parsed = TableContinuationMerger(**self.table_grouping_config).process(
                parsed
            )
        document = CanonicalDocumentAssembler(self.ignored_block_types).assemble(
            parsed,
            source_path=str(path),
            doc_id=f"doc_{doc_hash[:16]}",
            doc_hash=doc_hash,
        )
        title = self._extract_title(document.text)
        if title:
            document.metadata["title"] = title
        return self._attach_section_tree(document)

    async def aload(self, file_path: str | Path) -> Document:
        return await asyncio.to_thread(self.load, file_path)

    def _client_instance(self) -> Any:
        if self._client is None:
            from src.libs.loader.mineru_api_client import MinerUApiClient

            self._client = MinerUApiClient(self.api_config)
        return self._client

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _extract_title(text: str) -> str | None:
        match = re.search(r"^[ \t]{0,3}#{1,6}[ \t]+(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return next((line.strip() for line in text.splitlines() if line.strip()), None)


__all__ = ["MineruPdfLoader"]
