"""Response contract tests for the query_knowledge_hub MCP handler."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.response.response_builder import MCPToolResponse
from src.mcp_server.tools.query_knowledge_hub import (
    TOOL_INPUT_SCHEMA,
    TOOL_OUTPUT_SCHEMA,
    query_knowledge_hub_handler,
)


def test_query_tool_declares_xml_default_and_structured_output() -> None:
    response_format = TOOL_INPUT_SCHEMA["properties"]["response_format"]

    assert response_format["default"] == "xml"
    assert response_format["enum"] == ["xml", "json", "markdown"]
    assert "results" in TOOL_OUTPUT_SCHEMA["properties"]
    assert "text" in (
        TOOL_OUTPUT_SCHEMA["properties"]["results"]["items"]["properties"]
    )


@pytest.mark.asyncio
async def test_handler_returns_xml_and_native_structured_content() -> None:
    payload = {
        "query": "revenue",
        "collection": "financebench",
        "result_count": 1,
        "results": [
            {
                "rank": 1,
                "chunk_id": "chunk_1",
                "text": "complete chunk text",
                "source": {"doc_id": "doc_1", "page_start": 48, "page_end": 48},
                "scores": {"final": 0.9},
                "metadata": {},
            }
        ],
        "has_images": False,
        "image_count": 0,
        "is_empty": False,
    }
    tool = Mock()
    tool.execute = AsyncMock(
        return_value=MCPToolResponse(
            content="<retrieval_results><result /></retrieval_results>",
            structured_content=payload,
        )
    )

    with patch(
        "src.mcp_server.tools.query_knowledge_hub.get_tool_instance",
        return_value=tool,
    ):
        result = await query_knowledge_hub_handler(
            query="revenue",
            top_k=1,
            collection="financebench",
        )

    assert result.content[0].text.startswith("<retrieval_results")
    assert result.structuredContent == payload
    assert result.isError is False
    tool.execute.assert_awaited_once_with(
        query="revenue",
        top_k=1,
        collection="financebench",
        response_format="xml",
    )
