"""MCP tool for listing logical documents in a Chroma collection."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp import types

from src.mcp_server.tools.document_catalog import (
    DocumentCatalogEntry,
    build_document_catalog,
)

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.mcp_server.protocol_handler import ProtocolHandler

logger = logging.getLogger(__name__)

TOOL_NAME = "list_documents"
TOOL_DESCRIPTION = """List documents in a knowledge-base collection.

Returns each logical document's document ID, title, source, type, page numbers,
and chunk count. Use the returned document ID with get_document_summary.
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "collection": {
            "type": "string",
            "description": "Collection whose documents should be listed.",
        },
    },
    "required": ["collection"],
}


@dataclass
class ListDocumentsConfig:
    persist_directory: str = "./data/db/chroma"


class ListDocumentsTool:
    """Build a document-level catalog from flat Chroma chunk records."""

    def __init__(
        self,
        settings: Settings | None = None,
        config: ListDocumentsConfig | None = None,
    ) -> None:
        self._settings = settings
        self._config = config

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings

            self._settings = load_settings()
        return self._settings

    @property
    def config(self) -> ListDocumentsConfig:
        if self._config is None:
            persist_directory = getattr(
                self.settings.vector_store,
                "persist_directory",
                "./data/db/chroma",
            )
            self._config = ListDocumentsConfig(persist_directory=persist_directory)
        return self._config

    def _get_chroma_client(self) -> Any:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise ImportError(
                "chromadb package is required for list_documents. "
                "Install it with: pip install chromadb"
            ) from exc

        persist_path = Path(self.config.persist_directory).resolve()
        if not persist_path.exists():
            raise ValueError(f"ChromaDB directory does not exist: {persist_path}")

        return chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    def list_documents(self, collection: str) -> list[DocumentCatalogEntry]:
        if not collection or not collection.strip():
            raise ValueError("collection cannot be empty")
        client = self._get_chroma_client()
        try:
            chroma_collection = client.get_collection(name=collection)
        except Exception as exc:
            raise ValueError(f"Collection '{collection}' was not found") from exc
        return build_document_catalog(chroma_collection)

    @staticmethod
    def format_response(
        collection: str,
        documents: list[DocumentCatalogEntry],
    ) -> str:
        if not documents:
            return f"No documents found in collection `{collection}`."

        total_chunks = sum(document.chunk_count for document in documents)
        lines = [
            f"## Documents in `{collection}` ({len(documents)} total, {total_chunks} chunks)",
            "",
        ]
        for index, document in enumerate(documents, start=1):
            lines.append(f"{index}. **{document.title}**")
            lines.append(f"   - Document ID: `{document.doc_id}`")
            if document.source_path:
                lines.append(f"   - Source: `{document.source_path}`")
            if document.doc_type:
                lines.append(f"   - Type: {document.doc_type.upper()}")
            lines.append(f"   - Chunks: {document.chunk_count}")
            if document.pages:
                pages = ", ".join(str(page) for page in document.pages)
                lines.append(f"   - Pages: {pages}")
            if document.summary:
                lines.append(f"   - Summary: {document.summary}")
            lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def format_json(
        collection: str,
        documents: list[DocumentCatalogEntry],
    ) -> str:
        payload = {
            "collection": collection,
            "document_count": len(documents),
            "chunk_count": sum(document.chunk_count for document in documents),
            "documents": [document.to_dict() for document in documents],
        }
        return (
            "\n---\n**Documents (JSON):**\n```json\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
        )

    async def execute(self, collection: str) -> types.CallToolResult:
        try:
            documents = await asyncio.to_thread(self.list_documents, collection)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_response(collection, documents),
                    ),
                    types.TextContent(
                        type="text",
                        text=self.format_json(collection, documents),
                    ),
                ],
                isError=False,
            )
        except Exception as exc:
            logger.exception("Error executing list_documents")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error listing documents: {exc}",
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    tool = ListDocumentsTool()

    async def handler(collection: str) -> types.CallToolResult:
        return await tool.execute(collection=collection)

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
