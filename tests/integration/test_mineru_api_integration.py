"""Opt-in real integration test for MinerU precise API v4."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.libs.loader.mineru_pdf_loader import MineruPdfLoader

_PDF = Path("tests/fixtures/sample_documents/simple.pdf")
_TABLE_PDF = Path("tests/fixtures/sample_documents/chinese_table_chart_doc.pdf")


@pytest.mark.integration
@pytest.mark.slow
def test_real_mineru_api_returns_normalized_loader_contract() -> None:
    settings = load_settings("config/settings.yaml")
    mineru_config = settings.ingestion.loader.mineru
    api_config = mineru_config.get("api", {})
    token_env = str(api_config.get("token_env", "")).strip()
    if not api_config.get("api_key") and not (token_env and os.getenv(token_env)):
        pytest.skip("MinerU api_key or token_env is not configured")
    loader = MineruPdfLoader(
        mineru_config=mineru_config,
        extract_images=False,
    )

    document = loader.load(_PDF)

    assert document.metadata["parser_provider"] == "mineru"
    assert document.metadata["page_count"] == 1
    assert document.metadata["page_spans"] == [
        {
            "page": 1,
            "page_index": 0,
            "start_offset": 0,
            "end_offset": len(document.text),
        }
    ]
    assert document.metadata["parsed_structure"]["blocks"]
    assert document.metadata["section_tree"]["section_count"] >= 1
    assert "Sample Document" in document.text


@pytest.mark.integration
@pytest.mark.slow
def test_real_mineru_api_preserves_table_html_and_page_offsets() -> None:
    settings = load_settings("config/settings.yaml")
    mineru_config = settings.ingestion.loader.mineru
    api_config = mineru_config.get("api", {})
    token_env = str(api_config.get("token_env", "")).strip()
    if not api_config.get("api_key") and not (token_env and os.getenv(token_env)):
        pytest.skip("MinerU api_key or token_env is not configured")

    document = MineruPdfLoader(
        mineru_config=mineru_config,
        extract_images=False,
    ).load(_TABLE_PDF)
    blocks = document.metadata["parsed_structure"]["blocks"]
    page_three_tables = [
        block
        for block in blocks
        if block["type"] == "table" and block["page_index"] == 2
    ]

    assert document.metadata["page_count"] == 6
    assert len(page_three_tables) == 1
    table = page_three_tables[0]
    assert "<table>" in table["content"]
    assert "text-embedding-3-large" in table["content"]
    assert "3072" in table["content"]
    assert table["caption"] == ["表 1：主流 Embedding 模型参数对比"]
    rendered = document.text[table["start_offset"] : table["end_offset"]]
    assert table["content"] in rendered
