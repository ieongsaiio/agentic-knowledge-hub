"""Response Builder for constructing MCP-formatted responses.

This module builds structured responses for MCP tools, combining:
- Human-readable Markdown content with citation markers
- Structured citation data for machine consumption
- Multimodal content (text + images) support
- Proper handling of empty results and error cases
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from xml.etree import ElementTree

from mcp import types

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.response.multimodal_assembler import MultimodalAssembler


@dataclass
class MCPToolResponse:
    """Structured response for MCP tools.
    
    Attributes:
        content: LLM-facing XML, JSON, or Markdown text.
        structured_content: Canonical JSON payload for MCP structuredContent.
        citations: List of structured citations for reference
        metadata: Additional response metadata (query, result_count, etc.)
        is_empty: Whether the search returned no results
        image_contents: List of MCP ImageContent blocks for multimodal responses
    """
    content: str
    structured_content: Dict[str, Any] = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_empty: bool = False
    image_contents: List[types.ImageContent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP protocol.
        
        Returns:
            Dictionary with 'content' and 'structuredContent' fields.
        """
        structured_content = self.structured_content or {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "isEmpty": self.is_empty,
            }
        return {
            "content": self.content,
            "structuredContent": structured_content,
        }
    
    def to_mcp_content(self) -> List[Union[types.TextContent, types.ImageContent]]:
        """Convert to MCP content blocks format.
        
        Returns:
            List of content blocks for MCP CallToolResult.
            Includes TextContent and optionally ImageContent blocks.
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = [
            types.TextContent(
                type="text",
                text=self.content,
            )
        ]
        
        # Add image blocks if present (multimodal response)
        if self.image_contents:
            blocks.extend(self.image_contents)
        
        return blocks
    
    @property
    def has_images(self) -> bool:
        """Check if response contains images.
        
        Returns:
            True if response has image content, False otherwise.
        """
        return len(self.image_contents) > 0


class ResponseBuilder:
    """Builds MCP-formatted responses from retrieval results.
    
    This class transforms retrieval results into structured MCP responses,
    including human-readable Markdown with inline citations and structured
    citation data for machine consumption.
    
    Supports multimodal responses with images when results contain image
    references in their metadata.
    
    Example:
        >>> builder = ResponseBuilder()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> response = builder.build(results, "What is Azure OpenAI?")
        >>> print(response.content)  # Markdown with [1], [2] markers
        >>> print(response.citations[0].source)  # "docs/guide.pdf"
        >>> print(response.has_images)  # True if images found
    """
    
    SUPPORTED_RESPONSE_FORMATS = frozenset({"xml", "json", "markdown"})

    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional["MultimodalAssembler"] = None,
        max_results_in_content: int = 5,
        snippet_max_length: int = 300,
        enable_multimodal: bool = True,
        default_response_format: str = "xml",
    ) -> None:
        """Initialize ResponseBuilder.
        
        Args:
            citation_generator: Optional CitationGenerator instance.
                If None, creates a default one.
            multimodal_assembler: Optional MultimodalAssembler for image handling.
                If None and enable_multimodal=True, creates a default one.
            max_results_in_content: Deprecated compatibility argument. All
                retrieved results are rendered.
            snippet_max_length: Deprecated compatibility argument. Chunk text
                is always rendered in full.
            enable_multimodal: Whether to include images in response (default: True).
            default_response_format: TextContent rendering format.
        """
        self.citation_generator = citation_generator or CitationGenerator()
        self.max_results_in_content = max_results_in_content
        self.snippet_max_length = snippet_max_length
        self.enable_multimodal = enable_multimodal
        self.default_response_format = self._validate_response_format(
            default_response_format
        )
        
        # Lazy-load multimodal assembler to avoid circular imports
        self._multimodal_assembler = multimodal_assembler
    
    @property
    def multimodal_assembler(self) -> "MultimodalAssembler":
        """Get or create MultimodalAssembler instance."""
        if self._multimodal_assembler is None:
            from src.core.response.multimodal_assembler import MultimodalAssembler
            self._multimodal_assembler = MultimodalAssembler()
        return self._multimodal_assembler
    
    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        collection: Optional[str] = None,
        include_images: bool = True,
        response_format: Optional[str] = None,
    ) -> MCPToolResponse:
        """Build MCP response from retrieval results.
        
        Args:
            results: List of RetrievalResult from search.
            query: Original user query.
            collection: Optional collection name.
            include_images: Whether to include images in response (default: True).
            response_format: TextContent format: xml, json, or markdown.
            
        Returns:
            MCPToolResponse with formatted content, citations, and optional images.
        """
        effective_format = self._validate_response_format(
            response_format or self.default_response_format
        )

        # Handle empty results
        if not results:
            return self._build_empty_response(query, collection, effective_format)
        
        # Generate citations
        citations = self.citation_generator.generate(results)
        
        # Build metadata
        metadata = self._build_metadata(query, collection, len(results))
        
        # Assemble image content if enabled
        image_contents: List[types.ImageContent] = []
        if self.enable_multimodal and include_images:
            image_blocks = self.multimodal_assembler.assemble(results, collection)
            # Filter to only ImageContent blocks
            image_contents = [
                block for block in image_blocks
                if isinstance(block, types.ImageContent)
            ]
            if image_contents:
                metadata["has_images"] = True
                metadata["image_count"] = len(image_contents)

        structured_content = self._build_structured_content(
            results=results,
            query=query,
            collection=collection,
            image_count=len(image_contents),
        )
        content = self._render_content(structured_content, effective_format)

        return MCPToolResponse(
            content=content,
            structured_content=structured_content,
            citations=citations,
            metadata=metadata,
            is_empty=False,
            image_contents=image_contents,
        )

    def build_error(
        self,
        query: str,
        collection: Optional[str],
        error_message: str,
        response_format: Optional[str] = None,
    ) -> MCPToolResponse:
        """Build a protocol-safe error response in the requested format."""
        effective_format = self._validate_response_format(
            response_format or self.default_response_format
        )
        structured_content = {
            "query": query,
            "collection": collection,
            "result_count": 0,
            "results": [],
            "has_images": False,
            "image_count": 0,
            "is_empty": True,
            "error": error_message,
        }
        return MCPToolResponse(
            content=self._render_content(structured_content, effective_format),
            structured_content=structured_content,
            metadata={
                "query": query,
                "collection": collection,
                "error": error_message,
            },
            is_empty=True,
        )
    
    def _build_empty_response(
        self,
        query: str,
        collection: Optional[str] = None,
        response_format: str = "xml",
    ) -> MCPToolResponse:
        """Build response for empty results.
        
        Args:
            query: Original user query.
            collection: Optional collection name.
            
        Returns:
            MCPToolResponse indicating no results found.
        """
        metadata = self._build_metadata(query, collection, 0)
        structured_content = {
            "query": query,
            "collection": collection,
            "result_count": 0,
            "results": [],
            "has_images": False,
            "image_count": 0,
            "is_empty": True,
        }
        content = self._render_content(structured_content, response_format)

        return MCPToolResponse(
            content=content,
            structured_content=structured_content,
            citations=[],
            metadata=metadata,
            is_empty=True,
        )
    
    def _build_structured_content(
        self,
        results: List[RetrievalResult],
        query: str,
        collection: Optional[str],
        image_count: int,
    ) -> Dict[str, Any]:
        """Build the canonical machine-readable retrieval payload."""
        return {
            "query": query,
            "collection": collection,
            "result_count": len(results),
            "results": [
                self._serialize_result(rank, result)
                for rank, result in enumerate(results, start=1)
            ],
            "has_images": image_count > 0,
            "image_count": image_count,
            "is_empty": not results,
        }

    def _serialize_result(
        self,
        rank: int,
        result: RetrievalResult,
    ) -> Dict[str, Any]:
        """Serialize one result without truncating its text."""
        public_metadata = {
            key: value
            for key, value in (result.metadata or {}).items()
            if key != "text"
        }
        metadata = self._json_safe(public_metadata)
        page_start = self._first_present(
            result.metadata,
            "page_start",
            "page_num",
            "page",
        )
        page_end = self._first_present(
            result.metadata,
            "page_end",
            "page_num",
            "page",
        )
        source = {
            "doc_id": self._first_present(result.metadata, "doc_id", "source_ref"),
            "path": self._first_present(result.metadata, "source_path", "source"),
            "page_start": page_start,
            "page_end": page_end,
        }
        source = {key: value for key, value in source.items() if value is not None}

        scores: Dict[str, float] = {"final": round(float(result.score), 6)}
        score_fields = {
            "dense_score": "dense",
            "sparse_score": "sparse",
            "fusion_score": "fusion",
            "rrf_score": "rrf",
            "original_score": "original",
            "rerank_score": "rerank",
        }
        for metadata_key, output_key in score_fields.items():
            score = result.metadata.get(metadata_key)
            if isinstance(score, (int, float)):
                scores[output_key] = round(float(score), 6)

        return {
            "rank": rank,
            "chunk_id": result.chunk_id,
            "text": result.text or "",
            "source": source,
            "scores": scores,
            "metadata": metadata,
        }

    def _render_content(
        self,
        payload: Dict[str, Any],
        response_format: str,
    ) -> str:
        """Render the canonical payload for the MCP TextContent block."""
        if response_format == "json":
            return json.dumps(payload, ensure_ascii=False, indent=2)
        if response_format == "markdown":
            return self._render_markdown(payload)
        return self._render_xml(payload)

    def _render_xml(self, payload: Dict[str, Any]) -> str:
        """Render complete retrieval results as escaped XML."""
        root_attributes = {
            "query": str(payload.get("query", "")),
            "result_count": str(payload.get("result_count", 0)),
        }
        if payload.get("collection") is not None:
            root_attributes["collection"] = str(payload["collection"])
        root = ElementTree.Element("retrieval_results", root_attributes)

        if payload.get("error"):
            ElementTree.SubElement(root, "error").text = str(payload["error"])

        for result in payload.get("results", []):
            result_element = ElementTree.SubElement(
                root,
                "result",
                {
                    "rank": str(result["rank"]),
                    "chunk_id": str(result["chunk_id"]),
                },
            )
            source_attributes = {
                key: str(value)
                for key, value in result.get("source", {}).items()
                if value is not None
            }
            ElementTree.SubElement(result_element, "source", source_attributes)

            scores_element = ElementTree.SubElement(result_element, "scores")
            for name, value in result.get("scores", {}).items():
                ElementTree.SubElement(
                    scores_element,
                    "score",
                    {"name": str(name), "value": str(value)},
                )

            ElementTree.SubElement(result_element, "text").text = result.get(
                "text", ""
            )
            ElementTree.SubElement(result_element, "metadata").text = json.dumps(
                result.get("metadata", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            )

        return ElementTree.tostring(root, encoding="unicode")

    def _render_markdown(self, payload: Dict[str, Any]) -> str:
        """Render complete retrieval results as human-readable Markdown."""
        if payload.get("error"):
            return (
                "## 查询失败\n\n"
                f"查询: **{payload.get('query', '')}**\n\n"
                f"错误信息: {payload['error']}"
            )
        if not payload.get("results"):
            return (
                "## 未找到相关结果\n\n"
                f"查询: **{payload.get('query', '')}**"
            )

        lines = [
            "## 检索结果",
            "",
            f"针对查询 **\"{payload.get('query', '')}\"** "
            f"找到 {payload.get('result_count', 0)} 条相关结果:",
        ]
        for result in payload["results"]:
            source = result.get("source", {})
            lines.extend(
                [
                    "",
                    f"### [{result['rank']}] 结果 {result['rank']}",
                    f"**相关度分数:** `{result['scores']['final']}`",
                    f"**来源:** `{source.get('path', 'unknown')}`",
                ]
            )
            if source.get("page_start") is not None:
                page_label = str(source["page_start"])
                if source.get("page_end") != source.get("page_start"):
                    page_label += f"-{source.get('page_end')}"
                lines.append(f"**页码:** {page_label}")
            lines.extend(["", result.get("text", "")])
        return "\n".join(lines)

    def _validate_response_format(self, response_format: str) -> str:
        normalized = str(response_format).strip().lower()
        if normalized not in self.SUPPORTED_RESPONSE_FORMATS:
            supported = ", ".join(sorted(self.SUPPORTED_RESPONSE_FORMATS))
            raise ValueError(
                f"Unsupported response_format '{response_format}'. "
                f"Expected one of: {supported}"
            )
        return normalized

    @staticmethod
    def _first_present(metadata: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                return value
        return None

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        return str(value)
    
    def _build_metadata(
        self,
        query: str,
        collection: Optional[str],
        result_count: int,
    ) -> Dict[str, Any]:
        """Build response metadata.
        
        Args:
            query: Original query.
            collection: Collection name.
            result_count: Number of results.
            
        Returns:
            Metadata dictionary.
        """
        metadata = {
            "query": query,
            "result_count": result_count,
        }
        if collection:
            metadata["collection"] = collection
        return metadata
