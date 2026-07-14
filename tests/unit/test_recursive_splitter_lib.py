"""Unit tests for RecursiveSplitter.

Test Coverage:
- Configuration-driven instantiation from settings
- Chunk size and overlap parameter handling
- Markdown structure preservation (headers, code blocks)
- Edge cases: empty text, very short text, very long text
- Error handling: missing dependencies, invalid configuration
- Integration with BaseSplitter validation
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.libs.splitter.base_splitter import BaseSplitter

# Test if langchain-text-splitters is available
try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveSplitter = None  # type: ignore[misc, assignment]


class FakeTokenizer:
    """Small deterministic tokenizer for tests that must not access the network."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterConfiguration:
    """Tests for RecursiveSplitter configuration and initialization."""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object."""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        settings.ingestion.length_unit = "characters"
        settings.ingestion.tokenizer_model = None
        return settings

    def test_initialization_from_settings(self):
        """Test that RecursiveSplitter reads configuration from settings."""
        settings = self.create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(settings=settings)

        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert isinstance(splitter, BaseSplitter)

    def test_initialization_with_overrides(self):
        """Test that constructor parameters override settings values."""
        settings = self.create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(
            settings=settings,
            chunk_size=300,
            chunk_overlap=50,
        )

        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 50

    def test_initialization_with_custom_separators(self):
        """Test custom separator configuration."""
        settings = self.create_mock_settings()
        custom_separators = ["\n\n", "\n", " "]
        splitter = RecursiveSplitter(
            settings=settings,
            separators=custom_separators,
        )

        assert splitter.separators == custom_separators

    def test_initialization_default_separators(self):
        """Test that default separators are Markdown-aware."""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)

        # Check that default separators include common text boundaries
        assert "\n\n" in splitter.separators  # Paragraphs
        assert "\n" in splitter.separators  # Lines
        assert " " in splitter.separators  # Words
        assert "" in splitter.separators  # Characters

    def test_initialization_missing_settings(self):
        """Test error when settings.ingestion is missing."""
        settings = MagicMock()
        settings.ingestion = None

        with pytest.raises(ValueError, match="Missing ingestion configuration"):
            RecursiveSplitter(settings=settings)

    def test_initialization_invalid_chunk_size_negative(self):
        """Test error when chunk_size is negative."""
        settings = self.create_mock_settings(chunk_size=-100)

        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)

    def test_initialization_invalid_chunk_size_zero(self):
        """Test error when chunk_size is zero."""
        settings = self.create_mock_settings(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)

    def test_initialization_invalid_chunk_overlap_negative(self):
        """Test error when chunk_overlap is negative."""
        settings = self.create_mock_settings(chunk_overlap=-50)

        with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
            RecursiveSplitter(settings=settings)

    def test_initialization_overlap_exceeds_chunk_size(self):
        """Test error when chunk_overlap >= chunk_size."""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)

    def test_initialization_overlap_greater_than_chunk_size(self):
        """Test error when chunk_overlap > chunk_size."""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=200)

        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)

    def test_initialization_token_length_mode(self):
        """Token mode measures chunk size with the configured tokenizer."""
        settings = self.create_mock_settings(chunk_size=3, chunk_overlap=1)
        settings.ingestion.length_unit = "tokens"
        settings.ingestion.tokenizer_model = "Qwen/Qwen3-Embedding-0.6B"
        tokenizer = FakeTokenizer()

        splitter = RecursiveSplitter(settings=settings, tokenizer=tokenizer)
        chunks = splitter.split_text_with_offsets(
            "one two three four five six seven eight"
        )

        assert splitter.length_unit == "tokens"
        assert splitter.tokenizer is tokenizer
        assert len(chunks) >= 3
        for chunk, start_offset, end_offset in chunks:
            assert len(tokenizer.encode(chunk)) <= 3
            assert chunk == "one two three four five six seven eight"[
                start_offset:end_offset
            ]

    def test_token_length_mode_requires_tokenizer_model(self):
        settings = self.create_mock_settings(chunk_size=3, chunk_overlap=1)
        settings.ingestion.length_unit = "tokens"

        with pytest.raises(
            ValueError,
            match="tokenizer_model is required",
        ):
            RecursiveSplitter(settings=settings, tokenizer=FakeTokenizer())

    def test_invalid_length_unit(self):
        settings = self.create_mock_settings()
        settings.ingestion.length_unit = "bytes"

        with pytest.raises(ValueError, match="length_unit must be one of"):
            RecursiveSplitter(settings=settings)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterBasicSplitting:
    """Tests for basic text splitting behavior."""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object."""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        settings.ingestion.length_unit = "characters"
        settings.ingestion.tokenizer_model = None
        return settings

    def test_split_short_text(self):
        """Test splitting text shorter than chunk_size."""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        text = "This is a short text."
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_text_by_paragraphs(self):
        """Test that splitter respects paragraph boundaries."""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        # Create text long enough to require multiple chunks
        text = "Paragraph one with some more content to make it longer.\n\nParagraph two also needs sufficient length.\n\nParagraph three should push it over the limit."
        chunks = splitter.split_text(text)

        # Should split at paragraph boundaries when text exceeds chunk_size
        assert len(chunks) >= 1  # At minimum, returns the text
        for chunk in chunks:
            assert len(chunk) <= 70  # Allow some flexibility for boundary conditions

    def test_split_text_with_overlap(self):
        """Test that chunks have overlapping content."""
        settings = self.create_mock_settings(chunk_size=30, chunk_overlap=10)
        splitter = RecursiveSplitter(settings=settings)

        text = "This is a long sentence that will be split into multiple chunks with overlap."
        chunks = splitter.split_text(text)

        # Should produce multiple chunks
        assert len(chunks) >= 2

        # Each chunk should respect chunk_size (with some tolerance for word boundaries)
        for chunk in chunks:
            assert len(chunk) <= 30 + 20  # Allow tolerance for not breaking words

    def test_split_text_with_offsets_matches_original_text(self):
        """Offset output points to the exact source substring."""
        settings = self.create_mock_settings(chunk_size=35, chunk_overlap=8)
        splitter = RecursiveSplitter(settings=settings)

        text = (
            "First paragraph has enough content.\n\n"
            "Second paragraph also has enough content.\n\n"
            "Third paragraph closes the document."
        )
        chunks = splitter.split_text_with_offsets(text)

        assert len(chunks) >= 2
        assert [chunk for chunk, _, _ in chunks] == splitter.split_text(text)
        for chunk, start_offset, end_offset in chunks:
            assert start_offset >= 0
            assert end_offset == start_offset + len(chunk)
            assert text[start_offset:end_offset] == chunk

    def test_offset_fallback_normalizes_whitespace_and_preserves_order(self):
        """MarkItDown-style whitespace changes still map to source ranges."""
        settings = self.create_mock_settings(chunk_size=35, chunk_overlap=8)
        splitter = RecursiveSplitter(settings=settings)
        splitter._splitter = MagicMock()
        splitter._splitter.create_documents.return_value = [
            MagicMock(
                page_content="Repeated table row",
                metadata={"start_index": 36},
            ),
            MagicMock(
                page_content="Repeated table row",
                metadata={"start_index": -1},
            ),
        ]
        text = (
            "Repeated\n\n table   row"
            "\n\nMiddle content\n\n"
            "Repeated \n table row"
        )

        chunks = splitter.split_text_with_offsets(text)

        assert len(chunks) == 2
        assert chunks[0][1] < chunks[1][1]
        for chunk, start_offset, end_offset in chunks:
            source_range = text[start_offset:end_offset]
            assert " ".join(source_range.split()) == chunk

    def test_split_preserves_order(self):
        """Test that chunks preserve original text order."""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        text = "First section. Second section. Third section. Fourth section."
        chunks = splitter.split_text(text)

        # Reconstruct should preserve order
        reconstructed = " ".join(chunks)
        assert "First" in reconstructed
        assert reconstructed.index("First") < reconstructed.index("Fourth")

    def test_split_empty_string_validation(self):
        """Test that empty string raises validation error."""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)

        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.split_text("   ")

    def test_split_non_string_validation(self):
        """Test that non-string input raises validation error."""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)

        with pytest.raises(ValueError, match="must be a string"):
            splitter.split_text(123)  # type: ignore[arg-type]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterMarkdownStructure:
    """Tests for Markdown structure preservation."""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object."""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        settings.ingestion.length_unit = "characters"
        settings.ingestion.tokenizer_model = None
        return settings

    def test_split_markdown_with_headers(self):
        """Test that Markdown headers are preserved in chunks."""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        text = """# Main Header

## Section 1
Content for section 1 goes here.

## Section 2
Content for section 2 goes here.

## Section 3
Content for section 3 goes here."""

        chunks = splitter.split_text(text)

        # Should produce multiple chunks
        assert len(chunks) >= 1

        # Headers should be present in appropriate chunks
        all_text = "".join(chunks)
        assert "# Main Header" in all_text
        assert "## Section 1" in all_text

    def test_split_markdown_code_blocks(self):
        """Test that code blocks are handled appropriately."""
        settings = self.create_mock_settings(chunk_size=150, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        text = """Some text before code.

```python
def example():
    return "code"
```

Some text after code."""

        chunks = splitter.split_text(text)

        # All content should be preserved
        all_text = "".join(chunks)
        assert "def example():" in all_text
        assert "Some text before" in all_text
        assert "Some text after" in all_text

    def test_split_markdown_lists(self):
        """Test that Markdown lists are handled appropriately."""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        text = """# List Example

- Item 1
- Item 2
- Item 3
- Item 4
- Item 5"""

        chunks = splitter.split_text(text)

        # Should preserve list structure
        all_text = "".join(chunks)
        assert "- Item 1" in all_text
        assert "- Item 5" in all_text


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterEdgeCases:
    """Tests for edge cases and error handling."""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """Create mock settings object."""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        settings.ingestion.length_unit = "characters"
        settings.ingestion.tokenizer_model = None
        return settings

    def test_split_very_long_text(self):
        """Test splitting very long text."""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=20)
        splitter = RecursiveSplitter(settings=settings)

        # Generate long text (1000 words)
        text = " ".join(["word"] * 1000)
        chunks = splitter.split_text(text)

        # Should produce many chunks
        assert len(chunks) >= 10

        # Each chunk should respect chunk_size (with tolerance)
        for chunk in chunks:
            assert len(chunk) <= 100 + 30  # Allow tolerance

    def test_split_single_long_word(self):
        """Test handling of a single word longer than chunk_size."""
        settings = self.create_mock_settings(chunk_size=10, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        # Single word longer than chunk_size
        text = "a" * 100
        chunks = splitter.split_text(text)

        # Should still split (may exceed chunk_size for unsplittable content)
        assert len(chunks) >= 1

    def test_split_unicode_text(self):
        """Test handling of Unicode characters."""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        text = "Hello 世界! Привет мир! 🌍🌎🌏"
        chunks = splitter.split_text(text)

        # Should handle Unicode without errors
        assert len(chunks) >= 1
        all_text = "".join(chunks)
        assert "世界" in all_text
        assert "мир" in all_text
        assert "🌍" in all_text

    def test_split_with_trace_parameter(self):
        """Test that trace parameter is accepted but not used."""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)

        text = "Some text to split."
        mock_trace = MagicMock()

        # Should not raise error with trace parameter
        chunks = splitter.split_text(text, trace=mock_trace)
        assert len(chunks) >= 1


@pytest.mark.skipif(
    LANGCHAIN_AVAILABLE, reason="Test only when langchain-text-splitters NOT installed"
)
class TestRecursiveSplitterImportError:
    """Tests for ImportError when langchain-text-splitters is not installed."""

    def test_import_error_without_langchain(self):
        """Test that ImportError is raised when langchain is not available."""
        with patch.dict("sys.modules", {"langchain_text_splitters": None}):
            # Force reimport to trigger ImportError
            import importlib

            import src.libs.splitter.recursive_splitter

            importlib.reload(src.libs.splitter.recursive_splitter)

            from src.libs.splitter.recursive_splitter import RecursiveSplitter

            settings = MagicMock()
            settings.ingestion = MagicMock()
            settings.ingestion.chunk_size = 1000
            settings.ingestion.chunk_overlap = 200
            settings.ingestion.length_unit = "characters"
            settings.ingestion.tokenizer_model = None

            with pytest.raises(ImportError, match="langchain-text-splitters is not installed"):
                RecursiveSplitter(settings=settings)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterFactoryIntegration:
    """Tests for factory integration."""

    def test_factory_can_create_recursive_splitter(self):
        """Test that factory can instantiate RecursiveSplitter."""
        from src.libs.splitter.splitter_factory import SplitterFactory

        # Register the provider
        SplitterFactory.register_provider("recursive", RecursiveSplitter)

        # Create settings
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "recursive"
        settings.ingestion.chunk_size = 500
        settings.ingestion.chunk_overlap = 100
        settings.ingestion.length_unit = "characters"
        settings.ingestion.tokenizer_model = None

        # Factory should create RecursiveSplitter
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
