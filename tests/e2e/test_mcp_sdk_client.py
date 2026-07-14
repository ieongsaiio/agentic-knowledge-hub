"""E2E coverage using the official MCP Python SDK as the client.

This complements ``test_mcp_client.py``, which intentionally writes raw
JSON-RPC messages to the server's stdin to validate the wire protocol.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_official_sdk_stdio_client_session() -> None:
    """Discover and call every exposed tool through an SDK client session."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        cwd=PROJECT_ROOT,
        env=env,
        encoding="utf-8",
        encoding_error_handler="replace",
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialize_result = await session.initialize()
            assert initialize_result.serverInfo.name == "agentic-knowledge-hub"
            assert initialize_result.serverInfo.version == "0.1.0"

            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}
            assert tool_names == {
                "query_knowledge_hub",
                "list_collections",
                "list_documents",
                "get_document_summary",
            }

            collections_result = await session.call_tool(
                "list_collections",
                {"include_stats": True},
            )
            assert collections_result.isError is not True
            assert collections_result.content

            documents_result = await session.call_tool(
                "list_documents",
                {"collection": "default"},
            )
            assert documents_result.isError is not True
            assert len(documents_result.content) == 2

            query_result = await session.call_tool(
                "query_knowledge_hub",
                {"query": "test query", "top_k": 2},
            )
            assert query_result.content

            summary_result = await session.call_tool(
                "get_document_summary",
                {"doc_id": "does_not_exist"},
            )
            assert summary_result.isError is True
            assert summary_result.content
