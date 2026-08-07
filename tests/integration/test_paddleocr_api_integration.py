"""Opt-in real tests for the PaddleOCR Studio API."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.libs.loader.paddle_pdf_loader import PaddlePdfLoader

_PDF = Path("tests/fixtures/sample_documents/simple.pdf")
_AMCOR_PDF = Path(
    "data/benchmarks/financebench/pdfs/AMCOR_2023Q4_EARNINGS.pdf"
)
_SMALL_PDFS = [
    Path(
        "data/benchmarks/financebench/pdfs/"
        "JOHNSON_JOHNSON_2023_8K_dated-2023-08-23.pdf"
    ),
    Path(
        "data/benchmarks/financebench/pdfs/"
        "ULTABEAUTY_2023_8K_dated-2023-09-18.pdf"
    ),
    Path(
        "data/benchmarks/financebench/pdfs/"
        "COSTCO_2023_8K_dated-2023-08-16.pdf"
    ),
]


def _api_loader() -> PaddlePdfLoader:
    if not os.getenv("PADDLEOCR_API_TOKEN"):
        pytest.skip("PADDLEOCR_API_TOKEN is not configured")
    return PaddlePdfLoader(
        paddle_config={
            "backend": "api",
            "api": {
                "token_env": "PADDLEOCR_API_TOKEN",
                "model": "PaddleOCR-VL-1.6",
                "poll_interval_seconds": 2,
                "timeout_seconds": 600,
                "request_timeout_seconds": 120,
                "optional_payload": {
                    "useDocOrientationClassify": False,
                    "useDocUnwarping": False,
                    "useChartRecognition": False,
                    "restructurePages": True,
                    "mergeTables": False,
                    "relevelTitles": True,
                    "concatenatePages": False,
                    "returnMarkdownImages": False,
                },
            },
        },
        extract_images=False,
    )


def _assert_page_only_contract(document, expected_pages: int) -> None:
    assert document.metadata["page_count"] == expected_pages
    artifact_pages = document.metadata["parsed_artifact"]["restructured_pages"]
    assert [page["page_index"] for page in artifact_pages] == list(
        range(expected_pages)
    )

    spans = document.metadata["page_spans"]
    assert len(spans) == expected_pages
    assert [span["page"] for span in spans] == list(range(1, expected_pages + 1))
    assert all(
        "page_start" not in span and "page_end" not in span
        for span in spans
    )
    assert "\n\n".join(
        document.text[span["start_offset"] : span["end_offset"]]
        for span in spans
    ) == document.text


@pytest.mark.integration
@pytest.mark.slow
def test_real_paddleocr_api_returns_loader_contract() -> None:
    document = _api_loader().load(_PDF)

    _assert_page_only_contract(document, 1)
    assert document.metadata["page_spans"] == [
        {
            "page": 1,
            "start_offset": 0,
            "end_offset": len(document.text),
        }
    ]
    assert "Sample Document" in document.text
    assert "Section 1: Introduction" in document.text
    assert "parsed_artifact" in document.metadata


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_paddleocr_api_async_returns_same_contract() -> None:
    document = await _api_loader().aload(_PDF)

    _assert_page_only_contract(document, 1)
    assert "Sample Document" in document.text


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("pdf_path", _SMALL_PDFS, ids=lambda path: path.stem)
async def test_real_paddleocr_api_small_pdfs_preserve_physical_pages(
    pdf_path: Path,
) -> None:
    if os.getenv("PADDLEOCR_API_MULTI_TEST") != "1":
        pytest.skip("set PADDLEOCR_API_MULTI_TEST=1 to consume API jobs")
    if not pdf_path.is_file():
        pytest.skip(f"test PDF is unavailable: {pdf_path}")
    fitz = pytest.importorskip("fitz")
    with fitz.open(pdf_path) as pdf:
        expected_pages = pdf.page_count

    document = await _api_loader().aload(pdf_path)

    _assert_page_only_contract(document, expected_pages)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_paddleocr_api_amcor_14_page_contract() -> None:
    if os.getenv("PADDLEOCR_API_FULL_TEST") != "1":
        pytest.skip("set PADDLEOCR_API_FULL_TEST=1 to consume a full API job")
    if not _AMCOR_PDF.is_file():
        pytest.skip("FinanceBench AMCOR PDF is unavailable")

    document = await _api_loader().aload(_AMCOR_PDF)

    _assert_page_only_contract(document, 14)
    assert len(document.text) > 100_000
