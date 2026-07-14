"""Deterministic settings overrides for evaluation experiments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from src.core.settings import (
    EvaluationExperimentSettings,
    Settings,
    SettingsError,
    validate_settings,
)

_ALLOWED_OVERRIDE_KEYS = frozenset(
    {
        "llm",
        "embedding",
        "vision_llm",
        "vector_store",
        "retrieval",
        "rerank",
        "ingestion",
    }
)


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    """Return a deep dictionary representation of typed settings."""

    return asdict(settings)


def deep_merge(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge mappings recursively without mutating either input.

    Lists, scalar values, and explicit ``None`` values replace the base value.
    """

    merged = deepcopy(dict(base))
    for key, override_value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = deepcopy(override_value)
    return merged


def _validate_override_keys(
    overrides: Mapping[str, Any],
    experiment_name: str,
) -> None:
    experiment_path = f"evaluation.experiments[{experiment_name!r}].overrides"
    invalid_keys = sorted(
        (key for key in overrides if key not in _ALLOWED_OVERRIDE_KEYS),
        key=str,
    )
    if invalid_keys:
        invalid_paths = ", ".join(f"{experiment_path}.{key}" for key in invalid_keys)
        raise SettingsError(
            f"Experiment {experiment_name!r} contains forbidden override field(s): {invalid_paths}"
        )


def apply_experiment_overrides(
    base_settings: Settings,
    experiment: EvaluationExperimentSettings,
) -> Settings:
    """Apply one experiment's overrides and return validated typed settings."""

    experiment_path = f"evaluation.experiments[{experiment.name!r}].overrides"
    overrides = experiment.overrides
    if not isinstance(overrides, Mapping):
        raise SettingsError(
            f"Experiment {experiment.name!r} has invalid overrides at "
            f"{experiment_path}: expected a mapping"
        )

    _validate_override_keys(overrides, experiment.name)
    merged = deep_merge(settings_to_dict(base_settings), overrides)
    for optional_section in ("ingestion", "vision_llm"):
        if merged.get(optional_section) is None:
            merged.pop(optional_section, None)

    try:
        settings = Settings.from_dict(merged)
        validate_settings(settings)
    except SettingsError as exc:
        raise SettingsError(
            f"Experiment {experiment.name!r} has invalid settings at {experiment_path}: {exc}"
        ) from exc

    return settings
