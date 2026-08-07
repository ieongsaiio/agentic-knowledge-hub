"""Contract tests for loader-stage Markdown section trees."""

from src.libs.loader.markdown_section_tree import build_markdown_section_tree


def _flatten(node: dict) -> list[dict]:
    return [
        node,
        *[
            descendant
            for subsection in node["subsections"]
            for descendant in _flatten(subsection)
        ],
    ]


def test_builds_nested_sections_with_direct_content_only() -> None:
    text = (
        "# Recipe Book\n\nIntroduction.\n\n"
        "## Cookies\n\nCookie overview.\n\n"
        "### Ingredients\n\n- Flour\n- Sugar\n\n"
        "### Instructions\n\nBake the cookies.\n\n"
        "## Apple Pie\n\nPie overview."
    )

    tree = build_markdown_section_tree(text, document_id="doc_recipe")

    root = tree["root"]
    recipe = root["subsections"][0]
    cookies = recipe["subsections"][0]
    ingredients, instructions = cookies["subsections"]
    apple_pie = recipe["subsections"][1]

    assert tree["schema_version"] == 1
    assert recipe["heading_title"] == "Recipe Book"
    assert cookies["parent_section_id"] == recipe["id"]
    assert ingredients["path"] == [
        "Recipe Book",
        "Cookies",
        "Ingredients",
    ]
    assert ingredients["content"] == "### Ingredients\n\n- Flour\n- Sugar\n\n"
    assert "Ingredients" not in cookies["content"]
    assert "Instructions" not in cookies["content"]
    assert "Apple Pie" not in cookies["content"]
    assert apple_pie["sequence_number"] > instructions["sequence_number"]


def test_offsets_map_exactly_to_direct_and_subtree_content() -> None:
    text = "# Parent\n\nParent body.\n\n## Child\n\nChild body."

    tree = build_markdown_section_tree(text, document_id="doc_offsets")
    parent = tree["root"]["subsections"][0]
    child = parent["subsections"][0]

    assert text[
        parent["content_start_offset"]:parent["content_end_offset"]
    ] == parent["content"]
    assert text[parent["start_offset"]:parent["end_offset"]] == text
    assert text[child["start_offset"]:child["end_offset"]] == child["content"]


def test_skipped_heading_depth_uses_closest_shallower_parent() -> None:
    text = "# Report\n\nRoot.\n\n#### Detail\n\nDetail body."

    tree = build_markdown_section_tree(text, document_id="doc_depth")
    report = tree["root"]["subsections"][0]
    detail = report["subsections"][0]

    assert detail["heading_depth"] == 4
    assert detail["parent_section_id"] == report["id"]
    assert detail["path"] == ["Report", "Detail"]


def test_heading_markers_inside_fenced_code_are_not_sections() -> None:
    text = (
        "# API\n\n"
        "```markdown\n"
        "# This is code, not a section\n"
        "```\n\n"
        "## Usage\n\nRun it."
    )

    tree = build_markdown_section_tree(text, document_id="doc_code")
    sections = _flatten(tree["root"])

    assert [section["heading_title"] for section in sections] == [
        None,
        "API",
        "Usage",
    ]
    assert "# This is code, not a section" in sections[1]["content"]


def test_duplicate_headings_receive_unique_deterministic_ids() -> None:
    text = "# Notes\n\nFirst.\n\n# Notes\n\nSecond."

    first = build_markdown_section_tree(text, document_id="doc_duplicate")
    second = build_markdown_section_tree(text, document_id="doc_duplicate")
    first_sections = first["root"]["subsections"]
    second_sections = second["root"]["subsections"]

    assert first_sections[0]["id"] != first_sections[1]["id"]
    assert [section["id"] for section in first_sections] == [
        section["id"] for section in second_sections
    ]


def test_preamble_is_stored_as_root_direct_content() -> None:
    text = "Preamble text.\n\n# Report\n\nBody."

    tree = build_markdown_section_tree(text, document_id="doc_preamble")
    root = tree["root"]

    assert root["content"] == "Preamble text.\n\n"
    assert root["subsections"][0]["heading_title"] == "Report"


def test_page_metadata_uses_section_subtree_offsets() -> None:
    text = "# First\n\nPage one.\n\n## Child\n\nPage two."
    second_page_start = text.index("## Child")
    page_spans = [
        {"page": 1, "start_offset": 0, "end_offset": second_page_start},
        {"page": 2, "start_offset": second_page_start, "end_offset": len(text)},
    ]

    tree = build_markdown_section_tree(
        text,
        document_id="doc_pages",
        page_spans=page_spans,
    )
    parent = tree["root"]["subsections"][0]
    child = parent["subsections"][0]

    assert parent["page_start"] == 1
    assert parent["page_end"] == 2
    assert "page_num" not in parent
    assert child["page_start"] == 2
    assert child["page_end"] == 2
    assert child["page_num"] == 2
