"""Contract tests for the PaddleOCR-VL PDF loader.

The tests deliberately inject a runner and its artifact.  They must never
require PaddleOCR, Docker, a GPU, or a syntactically valid PDF.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.settings import Settings
from src.ingestion.chunking import DocumentChunker
from src.libs.loader.paddle_pdf_loader import PaddlePdfLoader
from src.libs.splitter.base_splitter import BaseSplitter


class FakeArtifact(dict[str, object]):
    """In-memory equivalent of the JSON artifact produced by the runner."""


class FakeRunner:
    def __init__(self, artifact: FakeArtifact):
        self.artifact = artifact
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> FakeArtifact:
        self.calls.append((args, kwargs))
        return self.artifact


class ParagraphSplitter(BaseSplitter):
    def split_text(self, text: str) -> list[str]:
        return [part for part in text.split("\n\n") if part]


class WholeDocumentSplitter(BaseSplitter):
    def split_text(self, text: str) -> list[str]:
        return [text]


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.pdf"
    path.write_bytes(b"not-a-real-pdf: the injected runner owns parsing")
    return path


def _artifact(tmp_path: Path, *pages: dict[str, object]) -> FakeArtifact:
    del tmp_path
    return FakeArtifact(
        pages=[{"page_index": page["page_index"]} for page in pages],
        restructured_pages=list(pages),
    )


def _page(page_index: int, markdown: str, images: dict[str, str] | None = None):
    return {
        "page_index": page_index,
        "markdown_text": markdown,
        "json": {"page_index": page_index},
        "images": images or {},
    }


def _loader(
    artifact: FakeArtifact,
    *,
    extract_images: bool = False,
    image_storage_dir: Path | None = None,
) -> tuple[PaddlePdfLoader, FakeRunner]:
    runner = FakeRunner(artifact)
    loader = PaddlePdfLoader(
        runner=runner,
        extract_images=extract_images,
        image_storage_dir=image_storage_dir or Path("unused-test-images"),
        paddle_config={
            "backend": "docker",
            "docker": {
                "engine": "transformers",
                "merge_tables": False,
                "relevel_titles": True,
                "concatenate_pages": False,
            },
        },
    )
    return loader, runner


def _chunker(monkeypatch: pytest.MonkeyPatch, splitter: BaseSplitter) -> DocumentChunker:
    from src.libs.splitter import splitter_factory

    monkeypatch.setattr(
        splitter_factory.SplitterFactory,
        "create",
        lambda settings: splitter,
    )
    settings = Mock(spec=Settings)
    settings.splitter = Mock()
    settings.splitter.provider = "fake"
    settings.splitter.chunk_size = 100
    settings.splitter.overlap = 0
    return DocumentChunker(settings)


def test_load_joins_each_page_markdown_and_records_exact_page_spans(
    tmp_path: Path,
    pdf_path: Path,
):
    artifact = _artifact(
        tmp_path,
        _page(0, "# Page one\n\nAlpha"),
        _page(1, "## Page two\n\nBeta"),
        _page(2, "Page three"),
    )
    loader, runner = _loader(artifact)

    document = loader.load(pdf_path)

    expected = "# Page one\n\nAlpha\n\n## Page two\n\nBeta\n\nPage three"
    assert document.text == expected
    assert document.metadata["page_count"] == 3
    assert document.metadata["page_spans"] == [
        {"page": 1, "start_offset": 0, "end_offset": 17},
        {"page": 2, "start_offset": 19, "end_offset": 36},
        {"page": 3, "start_offset": 38, "end_offset": 48},
    ]
    for span, page_text in zip(
        document.metadata["page_spans"],
        ["# Page one\n\nAlpha", "## Page two\n\nBeta", "Page three"],
        strict=True,
    ):
        assert document.text[span["start_offset"] : span["end_offset"]] == page_text
    assert len(runner.calls) == 1


def test_document_chunker_maps_loader_pages_to_1_based_page_nums_and_offsets(
    tmp_path: Path,
    pdf_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact = _artifact(tmp_path, _page(0, "First page"), _page(1, "Second page"))
    loader, _ = _loader(artifact)
    document = loader.load(pdf_path)

    chunks = _chunker(monkeypatch, ParagraphSplitter()).split_document(document)

    assert [chunk.text for chunk in chunks] == ["First page", "Second page"]
    assert [chunk.metadata["page_num"] for chunk in chunks] == [1, 2]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 10),
        (12, 23),
    ]
    assert [(chunk.metadata["start_offset"], chunk.metadata["end_offset"]) for chunk in chunks] == [
        (0, 10),
        (12, 23),
    ]


def test_document_chunker_marks_a_chunk_crossing_paddle_pages(
    tmp_path: Path,
    pdf_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact = _artifact(tmp_path, _page(0, "First page"), _page(1, "Second page"))
    loader, _ = _loader(artifact)
    document = loader.load(pdf_path)

    chunk = _chunker(monkeypatch, WholeDocumentSplitter()).split_document(document)[0]

    assert chunk.start_offset == 0
    assert chunk.end_offset == len(document.text)
    assert chunk.metadata["page_start"] == 1
    assert chunk.metadata["page_end"] == 2
    assert "page_num" not in chunk.metadata


def test_adjacent_page_tables_remain_independent_page_content(
    pdf_path: Path,
):
    first_table = "<table><tr><td>Q1</td><td>100</td></tr></table>"
    second_table = "<table><tr><td>Q2</td><td>200</td></tr></table>"
    artifact = FakeArtifact(
        pages=[
            {
                "res": {
                    "page_index": 0,
                    "prunedResult": {
                        "parsing_res_list": [
                            {
                                "block_label": "table",
                                "block_content": first_table,
                            }
                        ]
                    },
                }
            },
            {
                "res": {
                    "page_index": 1,
                    "prunedResult": {
                        "parsing_res_list": [
                            {"block_label": "table", "block_content": second_table},
                        ]
                    },
                }
            },
        ],
        restructured_pages=[
            _page(0, f"Page one\n{first_table}\nPage one note"),
            _page(1, f"Page two\n{second_table}\nPage two note"),
        ],
    )
    loader, _ = _loader(artifact)

    document = loader.load(pdf_path)
    spans = document.metadata["page_spans"]

    assert len(spans) == 2
    assert [span["page"] for span in spans] == [1, 2]
    assert all("page_start" not in span and "page_end" not in span for span in spans)
    assert document.text[spans[0]["start_offset"] : spans[0]["end_offset"]] == (
        f"Page one\n{first_table}\nPage one note"
    )
    assert document.text[spans[1]["start_offset"] : spans[1]["end_offset"]] == (
        f"Page two\n{second_table}\nPage two note"
    )


def test_vision_disabled_removes_markdown_images_and_emits_no_image_metadata(
    tmp_path: Path,
    pdf_path: Path,
):
    artifact = _artifact(
        tmp_path,
        _page(
            0,
            "Before ![revenue chart](assets/page_0000/chart.png) after",
            {"assets/page_0000/chart.png": "assets/page_0000/chart.png"},
        ),
    )
    loader, _ = _loader(artifact, extract_images=False)

    document = loader.load(pdf_path)

    assert "Before" in document.text and "after" in document.text
    assert "![" not in document.text
    assert "chart.png" not in document.text
    assert "[IMAGE:" not in document.text
    assert "images" not in document.metadata


def test_vision_enabled_replaces_markdown_image_with_placeholder_and_metadata(
    tmp_path: Path,
    pdf_path: Path,
):
    artifact_root = tmp_path / "runner-output"
    asset = artifact_root / "assets" / "page_0000" / "chart.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"fake-png")
    artifact = FakeArtifact(
        pages=[{"page_index": 0}],
        restructured_pages=[
            _page(
                0,
                "Before ![revenue chart](assets/page_0000/chart.png) after",
                {"assets/page_0000/chart.png": "assets/page_0000/chart.png"},
            )
        ],
    )
    loader, _ = _loader(artifact, extract_images=True)

    document = loader.load(pdf_path)

    images = document.metadata["images"]
    assert len(images) == 1
    image = images[0]
    assert image["id"]
    assert image["page"] == 1
    assert image["path"]
    assert f"[IMAGE: {image['id']}]" in document.text
    assert "![" not in document.text
    assert "chart.png" not in document.text


@pytest.mark.parametrize(
    ("extract_images", "markdown_images"),
    [(False, "remove"), (True, "placeholder")],
)
def test_cache_config_fingerprints_provider_engine_restructure_and_image_policy(
    tmp_path: Path,
    extract_images: bool,
    markdown_images: str,
):
    loader, _ = _loader(_artifact(tmp_path), extract_images=extract_images)

    config = loader.cache_config()

    assert config["parser"] == "paddleocr"
    assert config["backend"] == "docker"
    assert config["paddle_config"]["backend"] == "docker"
    docker = config["paddle_config"]["docker"]
    assert docker["engine"] == "transformers"
    assert {
        key: docker[key]
        for key in ("merge_tables", "relevel_titles", "concatenate_pages")
    } == {
        "merge_tables": False,
        "relevel_titles": True,
        "concatenate_pages": False,
    }
    assert config["extract_images"] is extract_images
    assert ("placeholder" if config["extract_images"] else "remove") == markdown_images


def test_api_cache_config_excludes_credentials() -> None:
    loader = PaddlePdfLoader(
        paddle_config={
            "backend": "api",
            "api": {
                "token": "must-not-be-cached",
                "token_env": "PADDLEOCR_API_TOKEN",
                "model": "PaddleOCR-VL-1.6",
                "poll_interval_seconds": 5,
            },
        },
        extract_images=False,
    )

    config = loader.cache_config()

    assert config["backend"] == "api"
    assert config["engine"] == "hosted_api"
    assert config["restructure"] is None
    assert config["paddle_config"] == {
        "backend": "api",
        "api": {
            "token_env": "PADDLEOCR_API_TOKEN",
            "model": "PaddleOCR-VL-1.6",
        },
    }
