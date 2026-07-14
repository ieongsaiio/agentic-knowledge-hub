"""Unit tests for config-driven evaluation experiment execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from src.core.settings import EvaluationExperimentSettings, Settings
from src.observability.evaluation.experiment_runner import (
    ExperimentPlan,
    ExperimentRunner,
)


def _settings(
    experiments: list[dict[str, Any]] | None = None,
) -> Settings:
    return Settings.from_dict(
        {
            "llm": {
                "provider": "test",
                "model": "test-chat",
                "temperature": 0.0,
                "max_tokens": 64,
            },
            "embedding": {
                "provider": "test",
                "model": "test-embedding",
                "dimensions": 8,
            },
            "vector_store": {
                "provider": "test",
                "persist_directory": "./test-data",
                "collection_name": "test-collection",
            },
            "retrieval": {
                "dense_top_k": 4,
                "sparse_top_k": 4,
                "fusion_top_k": 2,
                "rrf_k": 60,
            },
            "rerank": {
                "enabled": False,
                "provider": "test",
                "model": "test-reranker",
                "top_k": 2,
            },
            "evaluation": {
                "enabled": True,
                "provider": "custom",
                "metrics": ["hit_rate"],
                "benchmark": {
                    "provider": "test-benchmark",
                    "source_url": "https://example.invalid/data.json",
                    "data_dir": "./test-benchmark",
                    "split": "unit-v1",
                    "auto_download": False,
                },
                "experiments": experiments or [],
            },
            "observability": {
                "log_level": "INFO",
                "trace_enabled": False,
                "trace_file": "./test-traces.jsonl",
                "structured_logging": False,
            },
            "ingestion": {
                "chunk_size": 256,
                "chunk_overlap": 32,
                "splitter": "recursive",
                "batch_size": 8,
            },
        }
    )


def _runner(settings: Settings) -> ExperimentRunner:
    return ExperimentRunner(settings, lambda plan: plan.name)


def test_plan_falls_back_to_legacy_baseline() -> None:
    plans = _runner(_settings()).plan()

    assert [plan.name for plan in plans] == ["baseline"]
    assert plans[0].settings == _settings()
    assert plans[0].collection_name.startswith("test-benchmark__")
    assert plans[0].reuses_index_of is None


def test_plan_keeps_configured_order_and_skips_disabled_experiments() -> None:
    settings = _settings(
        [
            {"name": "first", "overrides": {}},
            {"name": "disabled", "enabled": False, "overrides": {}},
            {"name": "third", "overrides": {}},
        ]
    )

    plans = _runner(settings).plan()

    assert [plan.name for plan in plans] == ["first", "third"]


def test_selection_filters_without_reordering_configured_experiments() -> None:
    settings = _settings(
        [
            {"name": "first", "overrides": {}},
            {"name": "second", "overrides": {}},
            {"name": "third", "overrides": {}},
        ]
    )

    plans = _runner(settings).plan(["third", "first"])

    assert [plan.name for plan in plans] == ["first", "third"]


def test_selection_rejects_unknown_experiment_names() -> None:
    settings = _settings([{"name": "known", "overrides": {}}])

    with pytest.raises(ValueError, match=r"Unknown experiment name\(s\): 'missing'"):
        _runner(settings).plan(["known", "missing"])


def test_plan_rejects_duplicate_experiment_names() -> None:
    settings = _settings(
        [
            {"name": "duplicate", "overrides": {}},
            {"name": "duplicate", "overrides": {}},
        ]
    )

    with pytest.raises(ValueError, match="Duplicate experiment name: 'duplicate'"):
        _runner(settings).plan()


def test_plan_rejects_blank_experiment_names() -> None:
    settings = _settings()
    invalid_evaluation = replace(
        settings.evaluation,
        experiments=[EvaluationExperimentSettings(name="  ")],
    )

    with pytest.raises(ValueError, match="Experiment names must be nonempty strings"):
        _runner(replace(settings, evaluation=invalid_evaluation)).plan()


def test_plan_rejects_an_all_disabled_suite() -> None:
    settings = _settings(
        [
            {"name": "one", "enabled": False, "overrides": {}},
            {"name": "two", "enabled": False, "overrides": {}},
        ]
    )

    with pytest.raises(ValueError, match="At least one experiment must be enabled"):
        _runner(settings).plan()


def test_rerank_only_experiment_reuses_fingerprint_and_collection() -> None:
    settings = _settings(
        [
            {"name": "baseline", "overrides": {}},
            {
                "name": "reranked",
                "overrides": {"rerank": {"enabled": True, "top_k": 1}},
            },
        ]
    )

    baseline, reranked = _runner(settings).plan()

    assert reranked.settings.rerank.enabled is True
    assert reranked.index_fingerprint == baseline.index_fingerprint
    assert reranked.collection_name == baseline.collection_name
    assert reranked.reuses_index_of == "baseline"


def test_ingestion_change_creates_a_distinct_fingerprint_and_collection() -> None:
    settings = _settings(
        [
            {"name": "baseline", "overrides": {}},
            {
                "name": "smaller-chunks",
                "overrides": {"ingestion": {"chunk_size": 128}},
            },
        ]
    )

    baseline, changed = _runner(settings).plan()

    assert changed.index_fingerprint != baseline.index_fingerprint
    assert changed.collection_name != baseline.collection_name
    assert changed.reuses_index_of is None


def test_run_returns_callback_results_in_plan_order() -> None:
    settings = _settings(
        [
            {"name": "first", "overrides": {}},
            {"name": "second", "overrides": {}},
            {"name": "third", "overrides": {}},
        ]
    )
    callback_order: list[str] = []

    def execute(plan: ExperimentPlan) -> str:
        callback_order.append(plan.name)
        return f"result-{plan.name}"

    report = ExperimentRunner(settings, execute).run()

    assert callback_order == ["first", "second", "third"]
    assert list(report.experiments) == ["first", "second", "third"]
    assert report.experiments == {
        "first": "result-first",
        "second": "result-second",
        "third": "result-third",
    }
    assert report.errors == {}


def test_run_isolates_callback_errors_and_continues() -> None:
    settings = _settings(
        [
            {"name": "first", "overrides": {}},
            {"name": "broken", "overrides": {}},
            {"name": "last", "overrides": {}},
        ]
    )
    callback_order: list[str] = []

    def execute(plan: ExperimentPlan) -> str:
        callback_order.append(plan.name)
        if plan.name == "broken":
            raise RuntimeError("callback failed")
        return plan.name.upper()

    report = ExperimentRunner(settings, execute).run()

    assert callback_order == ["first", "broken", "last"]
    assert report.experiments == {"first": "FIRST", "last": "LAST"}
    assert report.errors == {"broken": "callback failed"}
    assert [plan.name for plan in report.plans] == ["first", "broken", "last"]
