"""Tests for normalizing MinerU artifacts into the shared parsed contract."""

from __future__ import annotations

from src.libs.loader.mineru_artifact_normalizer import MinerUArtifactNormalizer


def test_content_list_v2_is_preferred_and_preserves_structured_table_fields() -> None:
    artifact = {
        "provider": "mineru",
        "version": "3.0.1",
        "full_markdown": "# Report\n\n<table>...</table>",
        "content_list_v2": [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "Report"}],
                        "level": 1,
                    },
                    "bbox": [10, 10, 900, 50],
                },
                {
                    "type": "table",
                    "content": {
                        "html": "<table><tr><td>Revenue</td></tr></table>",
                        "table_caption": [
                            {"type": "text", "content": "Income statement"}
                        ],
                        "table_footnote": [
                            {"type": "text", "content": "USD millions"}
                        ],
                        "image_source": {"path": "images/table.jpg"},
                    },
                    "bbox": [10, 60, 900, 700],
                },
                {
                    "type": "page_footer",
                    "content": {
                        "page_footer_content": [
                            {"type": "text", "content": "Confidential"}
                        ]
                    },
                    "bbox": [10, 950, 900, 980],
                },
            ],
            [
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Second page."}
                        ]
                    },
                    "bbox": [20, 20, 800, 100],
                }
            ],
        ],
        "content_list": [
            {"type": "text", "text": "legacy must not be used", "page_idx": 9}
        ],
        "middle_json": {
            "pdf_info": [
                {"page_idx": 0, "page_size": [612, 792]},
                {"page_idx": 1, "page_size": [600, 800]},
            ]
        },
    }

    document = MinerUArtifactNormalizer().normalize(artifact)

    assert document.provider == "mineru"
    assert document.parser_version == "3.0.1"
    assert document.raw_markdown.startswith("# Report")
    assert [page.page_index for page in document.pages] == [0, 1]
    assert (document.pages[0].width, document.pages[0].height) == (612, 792)
    assert [block.type for block in document.pages[0].blocks] == [
        "title",
        "table",
        "page_footer",
    ]
    title, table, footer = document.pages[0].blocks
    assert title.content == "Report"
    assert title.level == 1
    assert table.content == "<table><tr><td>Revenue</td></tr></table>"
    assert table.caption == ["Income statement"]
    assert table.footnotes == ["USD millions"]
    assert table.images == [{"path": "images/table.jpg"}]
    assert table.page_index == 0
    assert table.order == 1
    assert footer.content == "Confidential"
    assert document.pages[1].blocks[0].content == "Second page."


def test_legacy_content_list_supports_vlm_and_pipeline_shapes_without_filtering() -> None:
    artifact = {
        "provider": "mineru",
        "version": "vlm",
        "full_markdown": "",
        "content_list_v2": None,
        "content_list": [
            {
                "type": "text",
                "text": "Overview",
                "text_level": 2,
                "page_idx": 0,
                "bbox": [1, 2, 3, 4],
            },
            {
                "type": "text",
                "text": "Body text",
                "page_idx": 0,
                "bbox": [1, 5, 3, 8],
            },
            {
                "type": "table",
                "table_body": "<table>legacy</table>",
                "table_caption": ["Table 1"],
                "table_footnote": ["Source: filing"],
                "img_path": "images/t1.jpg",
                "page_idx": 1,
                "bbox": [10, 20, 30, 40],
                "index": 7,
            },
            {
                "type": "list",
                "list_items": ["Alpha", "Beta"],
                "sub_type": "text",
                "page_idx": 1,
            },
            {"type": "header", "text": "Annual report", "page_idx": 1},
            {"type": "footer", "text": "2025", "page_idx": 1},
            {"type": "page_number", "text": "2", "page_idx": 1},
            {"type": "aside_text", "text": "Margin", "page_idx": 1},
            {"type": "page_footnote", "text": "Definition", "page_idx": 1},
            {
                "type": "code",
                "sub_type": "algorithm",
                "code_body": "return revenue",
                "code_caption": ["Algorithm 1"],
                "page_idx": 2,
            },
            {"type": "equation", "text": "$$x=1$$", "page_idx": 2},
            {
                "type": "chart",
                "content": "| Q1 | 10 |",
                "chart_caption": ["Revenue chart"],
                "chart_footnote": ["Unaudited"],
                "img_path": "images/chart.jpg",
                "page_idx": 2,
            },
            {"type": "image", "img_path": "images/logo.jpg", "page_idx": 2},
        ],
        "middle_json": {"pdf_info": []},
    }

    document = MinerUArtifactNormalizer().normalize(artifact)

    assert [page.page_index for page in document.pages] == [0, 1, 2]
    assert [block.type for block in document.pages[0].blocks] == ["title", "text"]
    page_one = document.pages[1].blocks
    assert [block.type for block in page_one] == [
        "table",
        "list",
        "page_header",
        "page_footer",
        "page_number",
        "page_aside_text",
        "page_footnote",
    ]
    assert page_one[0].order == 7
    assert page_one[1].content == "Alpha\nBeta"
    assert [block.type for block in document.pages[2].blocks] == [
        "code",
        "equation",
        "chart",
        "image",
    ]
    assert document.pages[2].blocks[0].metadata["sub_type"] == "algorithm"
    assert document.pages[2].blocks[2].caption == ["Revenue chart"]


def test_v2_page_objects_and_inline_spans_are_supported_with_stable_order() -> None:
    artifact = {
        "provider": "mineru",
        "version": "3.0",
        "full_markdown": "",
        "content_list_v2": [
            {
                "page_idx": 4,
                "page_size": [1000, 1400],
                "blocks": [
                    {
                        "type": "paragraph",
                        "order": 20,
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "Revenue "},
                                {
                                    "type": "hyperlink",
                                    "content": "rose",
                                    "url": "https://example.test",
                                },
                                {"type": "inline_equation", "content": "$10$"},
                            ]
                        },
                    },
                    {
                        "type": "page_header",
                        "order": 3,
                        "content": {
                            "page_header_content": [
                                {"type": "text", "content": "Header"}
                            ]
                        },
                    },
                ],
            }
        ],
        "content_list": [],
        "middle_json": {},
    }

    page = MinerUArtifactNormalizer().normalize(artifact).pages[0]

    assert page.page_index == 4
    assert (page.width, page.height) == (1000, 1400)
    assert [block.order for block in page.blocks] == [3, 20]
    assert [block.type for block in page.blocks] == ["page_header", "text"]
    assert page.blocks[1].content == "Revenue rose$10$"


def test_invalid_page_numbers_and_unknown_types_fail_clearly() -> None:
    normalizer = MinerUArtifactNormalizer()

    try:
        normalizer.normalize(
            {
                "provider": "mineru",
                "content_list": [{"type": "text", "text": "x", "page_idx": -1}],
            }
        )
    except ValueError as exc:
        assert "page_idx" in str(exc)
    else:
        raise AssertionError("negative page_idx should fail")

    document = normalizer.normalize(
        {
            "provider": "mineru",
            "content_list": [
                {"type": "custom_block", "content": "kept", "page_idx": 0}
            ],
        }
    )
    assert document.pages[0].blocks[0].type == "text"
    assert document.pages[0].blocks[0].metadata["source_type"] == "custom_block"
