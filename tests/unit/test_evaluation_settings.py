"""Tests for nested evaluation settings parsing."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.settings import Settings, SettingsError, load_settings


def _base_settings() -> dict[str, Any]:
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        },
        "vector_store": {
            "provider": "chroma",
            "persist_directory": "./data/db/chroma",
            "collection_name": "knowledge_hub",
        },
        "retrieval": {
            "dense_top_k": 20,
            "sparse_top_k": 20,
            "fusion_top_k": 10,
            "rrf_k": 60,
        },
        "rerank": {
            "enabled": False,
            "provider": "none",
            "model": "",
            "top_k": 5,
        },
        "evaluation": {
            "enabled": False,
            "provider": "custom",
            "metrics": ["hit_rate"],
        },
        "observability": {
            "log_level": "INFO",
            "trace_enabled": True,
            "trace_file": "./logs/traces.jsonl",
            "structured_logging": True,
        },
    }


def test_parses_full_nested_evaluation_settings() -> None:
    data = _base_settings()
    data["evaluation"] = {
        "enabled": True,
        "provider": "composite",
        "backends": ["benchmark", "ragas"],
        "benchmark": {
            "provider": "financebench",
            "source_url": "https://example.invalid/financebench",
            "data_dir": "./data/benchmarks/financebench",
            "split": "test",
            "auto_download": False,
            "sample_size": 25,
            "seed": 7,
        },
        "metrics": ["document_hit_rate@5", "answer_token_f1"],
        "experiments": [
            {
                "name": "baseline",
                "enabled": False,
                "overrides": {
                    "retrieval": {"dense_top_k": 40},
                    "custom": {"arbitrary": ["nested", {"value": 3}]},
                },
            },
            {
                "name": "reranked",
                "overrides": {"rerank": {"enabled": True, "provider": "api"}},
            },
        ],
        "output": {
            "directory": "./reports/evaluation",
            "save_per_query": False,
            "formats": ["json", "csv"],
        },
    }

    evaluation = Settings.from_dict(data).evaluation

    assert evaluation.enabled is True
    assert evaluation.provider == "composite"
    assert evaluation.backends == ["benchmark", "ragas"]
    assert evaluation.metrics == ["document_hit_rate@5", "answer_token_f1"]
    assert evaluation.benchmark is not None
    assert evaluation.benchmark.provider == "financebench"
    assert evaluation.benchmark.source_url == "https://example.invalid/financebench"
    assert evaluation.benchmark.data_dir == "./data/benchmarks/financebench"
    assert evaluation.benchmark.split == "test"
    assert evaluation.benchmark.auto_download is False
    assert evaluation.benchmark.sample_size == 25
    assert evaluation.benchmark.seed == 7
    assert [experiment.name for experiment in evaluation.experiments] == [
        "baseline",
        "reranked",
    ]
    assert evaluation.experiments[0].enabled is False
    assert evaluation.experiments[0].overrides == {
        "retrieval": {"dense_top_k": 40},
        "custom": {"arbitrary": ["nested", {"value": 3}]},
    }
    assert evaluation.experiments[1].enabled is True
    assert evaluation.output.directory == "./reports/evaluation"
    assert evaluation.output.save_per_query is False
    assert evaluation.output.formats == ["json", "csv"]


def test_applies_defaults_to_optional_nested_fields() -> None:
    data = _base_settings()
    data["evaluation"]["benchmark"] = {
        "provider": "financebench",
        "source_url": "https://example.invalid/financebench",
        "data_dir": "./data/benchmarks/financebench",
    }
    data["evaluation"]["experiments"] = [{"name": "baseline"}]

    evaluation = Settings.from_dict(data).evaluation

    assert evaluation.benchmark is not None
    assert evaluation.benchmark.split == "open_source"
    assert evaluation.benchmark.auto_download is True
    assert evaluation.benchmark.sample_size is None
    assert evaluation.benchmark.seed == 42
    assert evaluation.experiments[0].enabled is True
    assert evaluation.experiments[0].overrides == {}
    assert evaluation.output.directory == "./data/evaluation"
    assert evaluation.output.save_per_query is True
    assert evaluation.output.formats == ["json", "jsonl", "csv"]


def test_legacy_minimal_evaluation_settings_remain_valid() -> None:
    evaluation = Settings.from_dict(_base_settings()).evaluation

    assert evaluation.backends == []
    assert evaluation.benchmark is None
    assert evaluation.experiments == []
    assert evaluation.output.formats == ["json", "jsonl", "csv"]


@pytest.mark.parametrize("invalid_benchmark", ["financebench", [], 3])
def test_rejects_non_mapping_benchmark(invalid_benchmark: object) -> None:
    data = _base_settings()
    data["evaluation"]["benchmark"] = invalid_benchmark

    with pytest.raises(SettingsError, match="evaluation.benchmark"):
        Settings.from_dict(data)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("provider", 1),
        ("source_url", []),
        ("data_dir", False),
        ("split", 4),
        ("auto_download", "yes"),
        ("sample_size", "twenty"),
        ("seed", 1.5),
    ],
)
def test_rejects_invalid_benchmark_field_types(field: str, invalid_value: object) -> None:
    data = _base_settings()
    benchmark = {
        "provider": "financebench",
        "source_url": "https://example.invalid/financebench",
        "data_dir": "./data/benchmarks/financebench",
    }
    benchmark[field] = invalid_value
    data["evaluation"]["benchmark"] = benchmark

    with pytest.raises(SettingsError, match=rf"evaluation\.benchmark\.{field}"):
        Settings.from_dict(data)


def test_rejects_non_mapping_experiment_entry() -> None:
    data = _base_settings()
    data["evaluation"]["experiments"] = ["baseline"]

    with pytest.raises(SettingsError, match=r"evaluation\.experiments\[0\]"):
        Settings.from_dict(data)


def test_rejects_non_mapping_experiment_overrides() -> None:
    data = _base_settings()
    data["evaluation"]["experiments"] = [{"name": "baseline", "overrides": ["retrieval"]}]

    with pytest.raises(SettingsError, match=r"evaluation\.experiments\[0\]\.overrides"):
        Settings.from_dict(data)


def test_rejects_non_list_output_formats() -> None:
    data = _base_settings()
    data["evaluation"]["output"] = {"formats": "json"}

    with pytest.raises(SettingsError, match=r"evaluation\.output\.formats"):
        Settings.from_dict(data)


def test_repository_settings_example_defines_public_configuration_shape() -> None:
    import yaml

    with open("config/settings.yaml.example", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    assert set(data) == {
        "llm",
        "embedding",
        "vision_llm",
        "vector_store",
        "retrieval",
        "rerank",
        "evaluation",
        "observability",
        "ingestion",
    }
    assert set(data["evaluation"]) == {
        "enabled",
        "provider",
        "backends",
        "benchmark",
        "metrics",
        "experiments",
        "output",
    }
