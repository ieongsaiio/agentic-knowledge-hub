"""Tests for the provider-neutral parsed document contract."""

import json

import pytest

from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage


def test_parsed_document_round_trip_is_json_serializable() -> None:
    document = ParsedDocument(
        schema_version=1,
        provider="mineru",
        parser_version="2.1",
        pages=[
            ParsedPage(
                page_index=0,
                width=612.0,
                height=792.0,
                blocks=[
                    ParsedBlock(
                        block_id="p0_b0",
                        type="table",
                        content="<table><tr><td>Revenue</td></tr></table>",
                        page_index=0,
                        bbox=[10.0, 20.0, 300.0, 400.0],
                        order=2,
                        caption=["Revenue table"],
                        footnotes=["USD millions"],
                        images=[{"path": "images/table.jpg"}],
                        metadata={"confidence": 0.98},
                    )
                ],
            )
        ],
        raw_markdown="# Report",
        raw_artifact={"source": "fixture"},
    )

    payload = document.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert ParsedDocument.from_dict(payload) == document


def test_from_dict_accepts_omitted_optional_fields() -> None:
    document = ParsedDocument.from_dict(
        {
            "schema_version": 1,
            "provider": "paddle",
            "pages": [
                {
                    "page_index": 0,
                    "blocks": [
                        {
                            "block_id": "b0",
                            "type": "text",
                            "content": "Body",
                            "page_index": 0,
                        }
                    ],
                }
            ],
        }
    )

    block = document.pages[0].blocks[0]
    assert document.parser_version is None
    assert document.raw_artifact is None
    assert block.bbox is None
    assert block.caption == []
    assert block.metadata == {}


@pytest.mark.parametrize("page_index", [-1, True, "0"])
def test_page_index_must_be_a_non_negative_integer(page_index: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedPage.from_dict({"page_index": page_index, "blocks": []})


def test_unknown_contract_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        ParsedBlock.from_dict(
            {
                "block_id": "b0",
                "type": "text",
                "content": "Body",
                "page_index": 0,
                "provider_only_field": "unexpected",
            }
        )
