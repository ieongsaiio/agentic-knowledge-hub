"""Unit tests for evaluation experiment settings overrides."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.core.settings import (
    EvaluationExperimentSettings,
    Settings,
    SettingsError,
)
from src.observability.evaluation.settings_override import (
    apply_experiment_overrides,
    deep_merge,
)


@pytest.fixture
def base_settings() -> Settings:
    """Build the smallest complete settings object through public parsing."""

    return Settings.from_dict(
        {
            "llm": {
                "provider": "test",
                "model": "test-chat",
                "temperature": 0.0,
                "max_tokens": 256,
                "extra_chat_configs": {
                    "stop": ["BASE_STOP_1", "BASE_STOP_2"],
                    "seed": 7,
                },
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
                "dense_top_k": 10,
                "sparse_top_k": 12,
                "fusion_top_k": 6,
                "rrf_k": 60,
            },
            "rerank": {
                "enabled": False,
                "provider": "cross_encoder_api",
                "model": "test-reranker",
                "top_k": 3,
                "api_args": {
                    "base_url": "https://example.invalid/v1",
                    "timeout": 10,
                },
            },
            "evaluation": {
                "enabled": False,
                "provider": "custom",
                "metrics": ["hit_rate"],
            },
            "observability": {
                "log_level": "INFO",
                "trace_enabled": False,
                "trace_file": "./test-traces.jsonl",
                "structured_logging": False,
            },
        }
    )


def _experiment(
    overrides: dict[str, Any],
    *,
    name: str = "test-experiment",
) -> EvaluationExperimentSettings:
    return EvaluationExperimentSettings(name=name, overrides=overrides)


def test_empty_overrides_leave_baseline_unchanged(base_settings: Settings) -> None:
    result = apply_experiment_overrides(base_settings, _experiment({}))

    assert result == base_settings
    assert result is not base_settings


def test_nested_rerank_override_preserves_siblings_and_api_args(
    base_settings: Settings,
) -> None:
    result = apply_experiment_overrides(
        base_settings,
        _experiment({"rerank": {"enabled": True}}),
    )

    assert result.rerank.enabled is True
    assert result.rerank.provider == base_settings.rerank.provider
    assert result.rerank.model == base_settings.rerank.model
    assert result.rerank.top_k == base_settings.rerank.top_k
    assert result.rerank.api_args == base_settings.rerank.api_args


def test_retrieval_override_can_disable_sparse_only(
    base_settings: Settings,
) -> None:
    result = apply_experiment_overrides(
        base_settings,
        _experiment({"retrieval": {"enable_sparse": False}}),
    )

    assert result.retrieval.enable_dense is True
    assert result.retrieval.enable_sparse is False
    assert result.retrieval.dense_top_k == base_settings.retrieval.dense_top_k


def test_lists_replace_instead_of_merging(base_settings: Settings) -> None:
    result = apply_experiment_overrides(
        base_settings,
        _experiment(
            {
                "llm": {
                    "extra_chat_configs": {
                        "stop": ["EXPERIMENT_STOP"],
                    }
                }
            }
        ),
    )

    assert result.llm.extra_chat_configs == {
        "stop": ["EXPERIMENT_STOP"],
        "seed": 7,
    }


def test_apply_does_not_mutate_settings_or_overrides(
    base_settings: Settings,
) -> None:
    overrides = {
        "rerank": {
            "api_args": {
                "timeout": 30,
                "headers": {"X-Test": "experiment"},
            }
        }
    }
    original_settings = deepcopy(base_settings)
    original_overrides = deepcopy(overrides)

    result = apply_experiment_overrides(base_settings, _experiment(overrides))

    assert base_settings == original_settings
    assert overrides == original_overrides
    assert result.rerank.api_args == {
        "base_url": "https://example.invalid/v1",
        "timeout": 30,
        "headers": {"X-Test": "experiment"},
    }
    assert result.rerank.api_args is not base_settings.rerank.api_args


def test_explicit_none_replaces_base_value_without_mutating_inputs() -> None:
    base = {"section": {"enabled": True}, "items": ["base"]}
    overrides = {"section": None}
    original_base = deepcopy(base)
    original_overrides = deepcopy(overrides)

    result = deep_merge(base, overrides)

    assert result == {"section": None, "items": ["base"]}
    assert base == original_base
    assert overrides == original_overrides


@pytest.mark.parametrize("forbidden_key", ["evaluation", "observability", "unknown"])
def test_forbidden_top_level_override_keys_are_rejected(
    base_settings: Settings,
    forbidden_key: str,
) -> None:
    experiment_name = "forbidden-fields"

    with pytest.raises(SettingsError) as exc_info:
        apply_experiment_overrides(
            base_settings,
            _experiment({forbidden_key: {}}, name=experiment_name),
        )

    message = str(exc_info.value)
    assert experiment_name in message
    assert "forbidden override" in message
    assert f".overrides.{forbidden_key}" in message


def test_invalid_nested_value_reports_experiment_name(
    base_settings: Settings,
) -> None:
    experiment_name = "bad-rerank-enabled"

    with pytest.raises(SettingsError) as exc_info:
        apply_experiment_overrides(
            base_settings,
            _experiment(
                {"rerank": {"enabled": "not-a-boolean"}},
                name=experiment_name,
            ),
        )

    message = str(exc_info.value)
    assert experiment_name in message
    assert "invalid settings" in message
    assert "rerank.enabled" in message
