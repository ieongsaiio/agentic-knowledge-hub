"""Tests for the MinerU PDF loader integration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.libs.loader.mineru_pdf_loader import MineruPdfLoader


class _FakeClient:
    def __init__(self, artifact: dict) -> None:
        self.artifact = artifact
        self.paths: list[Path] = []

    def run(self, path: str | Path) -> dict:
        self.paths.append(Path(path))
        return self.artifact


def _artifact() -> dict:
    return {
        "provider": "mineru",
        "version": "3.0.1",
        "full_markdown": "# Results\n\nRevenue table",
        "content_list_v2": None,
        "content_list": [
            {"type": "header", "text": "Company confidential", "page_idx": 0},
            {"type": "text", "text": "Results", "text_level": 1, "page_idx": 0},
            {
                "type": "table",
                "table_caption": ["Revenue by year"],
                "table_body": "<table><tr><td>2024</td><td>100</td></tr></table>",
                "table_footnote": ["Amounts in millions."],
                "bbox": [10, 20, 900, 800],
                "page_idx": 0,
            },
        ],
        "middle_json": {
            "_version_name": "3.0.1",
            "pdf_info": [{"page_idx": 0, "page_size": [612, 792]}],
        },
        "model_json": None,
        "source_files": {"archive_members": ["full.md"]},
    }


def test_loader_normalizes_assembles_and_attaches_section_tree(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    client = _FakeClient(_artifact())
    loader = MineruPdfLoader(
        mineru_config={
            "backend": "api",
            "api": {"model_version": "vlm"},
            "ignored_block_types": ["page_header", "page_footer", "page_number"],
        },
        extract_images=False,
        client=client,
    )

    document = loader.load(pdf)

    assert client.paths == [pdf.resolve()]
    assert "Company confidential" not in document.text
    assert document.text.startswith("# Results")
    assert "Revenue by year" in document.text
    assert "<table>" in document.text
    assert "Amounts in millions." in document.text
    assert document.metadata["parser_provider"] == "mineru"
    assert document.metadata["parser_version"] == "3.0.1"
    assert document.metadata["page_spans"] == [
        {
            "page": 1,
            "page_index": 0,
            "start_offset": 0,
            "end_offset": len(document.text),
        }
    ]
    table = next(
        block
        for block in document.metadata["parsed_structure"]["blocks"]
        if block["type"] == "table"
    )
    assert table["caption"] == ["Revenue by year"]
    assert table["footnotes"] == ["Amounts in millions."]
    assert document.text[table["start_offset"] : table["end_offset"]].startswith(
        "Revenue by year"
    )
    assert document.metadata["section_tree"]["section_count"] == 1


def test_loader_cache_config_excludes_secret_and_runtime_settings() -> None:
    loader = MineruPdfLoader(
        mineru_config={
            "backend": "api",
            "api": {
                "token_env": "SECRET_ENV",
                "model_version": "vlm",
                "language": "en",
                "poll_interval_seconds": 5,
                "timeout_seconds": 600,
            },
            "ignored_block_types": ["page_header"],
        },
        client=_FakeClient(_artifact()),
    )

    config = loader.cache_config()

    assert config["provider"] == "mineru"
    assert config["api"] == {"model_version": "vlm", "language": "en"}
    assert config["ignored_block_types"] == ["image", "page_header"]
    assert "SECRET_ENV" not in str(config)
    assert "timeout_seconds" not in str(config)


def test_loader_rejects_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="supports only the api backend"):
        MineruPdfLoader(mineru_config={"backend": "docker"})


def test_loader_groups_adjacent_compatible_tables_and_preserves_units(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "grouped.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    artifact = {
        "version": "3.0.1",
        "content_list_v2": [[
            {
                "type": "table",
                "content": {
                    "table_caption": ["Service fees"],
                    "html": "<table><tr><td>Revenue</td><td>100</td></tr></table>",
                },
                "bbox": [20, 100, 980, 300],
            },
            {
                "type": "table",
                "content": {
                    "table_caption": ["Other expenses"],
                    "html": "<table><tr><td>Expense</td><td>60</td></tr></table>",
                },
                "bbox": [20, 320, 980, 520],
            },
        ]],
        "middle_json": {
            "pdf_info": [{"page_idx": 0, "page_size": [1000, 1000]}]
        },
    }
    loader = MineruPdfLoader(
        mineru_config={"backend": "api", "table_grouping": {"enabled": True}},
        client=_FakeClient(artifact),
    )

    document = loader.load(pdf)

    tables = [
        block
        for block in document.metadata["parsed_structure"]["blocks"]
        if block["type"] == "table"
    ]
    assert len(tables) == 1
    table = tables[0]
    assert table["metadata"]["chunk_role"] == "table_group"
    assert table["metadata"]["unit_count"] == 2
    assert [unit["caption"] for unit in table["metadata"]["units"]] == [
        "Service fees",
        "Other expenses",
    ]
    assert document.text.count("table-section-caption") == 2
    assert "Revenue" in document.text
    assert "Expense" in document.text


def test_loader_can_disable_table_grouping(tmp_path: Path) -> None:
    pdf = tmp_path / "ungrouped.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    artifact = {
        "content_list_v2": [[
            {"type": "table", "content": {"html": "<table><tr><td>A</td></tr></table>"}, "bbox": [0, 0, 100, 40]},
            {"type": "table", "content": {"html": "<table><tr><td>B</td></tr></table>"}, "bbox": [0, 45, 100, 90]},
        ]],
        "middle_json": {"pdf_info": [{"page_idx": 0, "page_size": [100, 100]}]},
    }
    loader = MineruPdfLoader(
        mineru_config={"backend": "api", "table_grouping": {"enabled": False}},
        client=_FakeClient(artifact),
    )

    document = loader.load(pdf)

    assert sum(
        block["type"] == "table"
        for block in document.metadata["parsed_structure"]["blocks"]
    ) == 2
