from scripts.export_benchmark_chunks import classify_chunk


def test_classify_table_group_by_role() -> None:
    assert classify_chunk({"chunk_role": "table_group", "unit_types": ["table"]}) == (
        "table_group"
    )


def test_classify_complete_special_and_mixed_units() -> None:
    assert classify_chunk({"unit_types": ["list"]}) == "list"
    assert classify_chunk({"unit_types": ["text", "list"]}) == "mixed:text+list"


def test_classify_defaults_to_text() -> None:
    assert classify_chunk({}) == "text"
