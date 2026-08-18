from scripts.evaluate_targeted_table_summaries import (
    deduplicate_dense_groups,
    merge_dense_results,
    select_target_chunk,
    summary_cache_key,
)


def test_select_target_chunk_prefers_atomic_fact_number_coverage() -> None:
    target = {
        "case_id": "case-1",
        "evidence_index": 1,
        "document_name": "report",
        "page_number": 4,
        "parent_table_id": "legacy",
    }
    case = {
        "answer": "$200",
        "evidence": [{"evidence_text": "Revenue increased from 100 to 200."}],
    }
    annotation = {
        "evidence_groups": [
            {"evidence_facts": [{"fact": "Revenue was 100 in 2022 and 200 in 2023."}]}
        ]
    }
    wrong = {"text": "Revenue was 100.", "dense_index_text": "", "metadata": {}}
    correct = {"text": "Revenue | 2022 | 2023\nRevenue | 100 | 200", "dense_index_text": "", "metadata": {}}

    selected, score = select_target_chunk(target, case, annotation, [wrong, correct])

    assert selected is correct
    assert score["number_coverage"] == 1.0


def test_summary_cache_key_changes_with_prompt_model_or_source() -> None:
    base = summary_cache_key(table_text="table", prompt="p1", model_config={"model": "m1"})
    assert base == summary_cache_key(table_text="table", prompt="p1", model_config={"model": "m1"})
    assert base != summary_cache_key(table_text="other", prompt="p1", model_config={"model": "m1"})
    assert base != summary_cache_key(table_text="table", prompt="p2", model_config={"model": "m1"})
    assert base != summary_cache_key(table_text="table", prompt="p1", model_config={"model": "m2"})


def test_merge_dense_results_orders_cosine_distance() -> None:
    merged = merge_dense_results(
        [("raw-a", 0.20), ("raw-b", 0.40)],
        [("alias-a", 0.10), ("alias-b", 0.30)],
    )
    assert merged == [
        ("alias-a", 0.10, "summary_alias"),
        ("raw-a", 0.20, "baseline"),
        ("alias-b", 0.30, "summary_alias"),
        ("raw-b", 0.40, "baseline"),
    ]


def test_deduplicate_dense_groups_keeps_best_raw_or_alias() -> None:
    merged = [
        ("raw-a", 0.10, "baseline"),
        ("alias-a", 0.20, "summary_alias"),
        ("alias-b", 0.30, "summary_alias"),
        ("raw-b", 0.40, "baseline"),
    ]
    assert deduplicate_dense_groups(
        merged,
        {
            "raw-a": "group-a",
            "alias-a": "group-a",
            "raw-b": "group-b",
            "alias-b": "group-b",
        },
    ) == [
        ("raw-a", 0.10, "baseline", "group-a"),
        ("alias-b", 0.30, "summary_alias", "group-b"),
    ]
