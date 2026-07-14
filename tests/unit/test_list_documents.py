"""Unit tests for the list_documents MCP tool."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
from mcp import types

from src.mcp_server.tools.document_catalog import build_document_catalog
from src.mcp_server.tools.list_documents import (
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    ListDocumentsConfig,
    ListDocumentsTool,
    register_tool,
)


@pytest.fixture
def collection() -> Mock:
    mock = Mock()
    mock.get.return_value = {
        "ids": ["chunk_a0", "chunk_a1", "chunk_b0"],
        "documents": ["First document text", "Second chunk", "Another document"],
        "metadatas": [
            {
                "source_ref": "doc_a",
                "source_path": "docs/a.pdf",
                "doc_hash": "a" * 64,
                "doc_type": "pdf",
                "title": "Document A",
                "summary": "Summary A",
                "tags": "rag,mcp",
                "chunk_index": 0,
                "page_num": 1,
            },
            {
                "source_ref": "doc_a",
                "source_path": "docs/a.pdf",
                "doc_hash": "a" * 64,
                "doc_type": "pdf",
                "title": "Section title",
                "tags": "mcp,agent",
                "chunk_index": 1,
                "page_num": 2,
            },
            {
                "source_ref": "doc_b",
                "source_path": "docs/b.pdf",
                "doc_hash": "b" * 64,
                "doc_type": "pdf",
                "title": "Document B",
                "chunk_index": 0,
                "page_num": 3,
            },
        ],
    }
    return mock


def test_build_document_catalog_groups_chunks(collection: Mock) -> None:
    documents = build_document_catalog(collection)

    assert len(documents) == 2
    doc_a = next(document for document in documents if document.doc_id == "doc_a")
    assert doc_a.chunk_count == 2
    assert doc_a.chunk_ids == ["chunk_a0", "chunk_a1"]
    assert doc_a.pages == [1, 2]
    assert doc_a.tags == ["rag", "mcp", "agent"]
    assert doc_a.title == "Document A"
    assert doc_a.summary == "Summary A"


def test_build_document_catalog_derives_doc_id_from_hash() -> None:
    collection = Mock()
    collection.get.return_value = {
        "ids": ["chunk_1"],
        "documents": ["Text"],
        "metadatas": [{"doc_hash": "1234567890abcdef" + "0" * 48}],
    }

    documents = build_document_catalog(collection)

    assert documents[0].doc_id == "doc_1234567890abcdef"


def test_format_response_and_json(collection: Mock) -> None:
    documents = build_document_catalog(collection)

    text = ListDocumentsTool.format_response("knowledge", documents)
    json_text = ListDocumentsTool.format_json("knowledge", documents)
    payload = json.loads(json_text.split("```json\n", 1)[1].rsplit("\n```", 1)[0])

    assert "## Documents in `knowledge` (2 total, 3 chunks)" in text
    assert "Document ID: `doc_a`" in text
    assert payload["document_count"] == 2
    assert payload["chunk_count"] == 3
    assert payload["documents"][0]["doc_id"] in {"doc_a", "doc_b"}


@pytest.mark.asyncio
async def test_execute_returns_text_and_json(collection: Mock) -> None:
    tool = ListDocumentsTool(
        config=ListDocumentsConfig(persist_directory="./data/db/chroma")
    )
    documents = build_document_catalog(collection)

    with patch.object(tool, "list_documents", return_value=documents):
        result = await tool.execute("knowledge")

    assert result.isError is False
    assert len(result.content) == 2
    assert all(isinstance(block, types.TextContent) for block in result.content)
    assert "Documents in `knowledge`" in result.content[0].text
    assert "Documents (JSON)" in result.content[1].text


@pytest.mark.asyncio
async def test_execute_returns_error() -> None:
    tool = ListDocumentsTool(
        config=ListDocumentsConfig(persist_directory="./data/db/chroma")
    )

    with patch.object(tool, "list_documents", side_effect=ValueError("missing")):
        result = await tool.execute("missing")

    assert result.isError is True
    assert "Error listing documents" in result.content[0].text


def test_register_tool() -> None:
    handler = Mock()

    register_tool(handler)

    call = handler.register_tool.call_args.kwargs
    assert call["name"] == "list_documents"
    assert call["input_schema"]["required"] == ["collection"]


def test_tool_contract() -> None:
    assert TOOL_NAME == "list_documents"
    assert TOOL_INPUT_SCHEMA["properties"]["collection"]["type"] == "string"
