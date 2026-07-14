"""Config-driven planning and execution of evaluation experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.core.settings import EvaluationExperimentSettings, Settings
from src.observability.evaluation.index_fingerprint import (
    build_index_fingerprint,
)
from src.observability.evaluation.index_fingerprint import (
    collection_name as build_collection_name,
)
from src.observability.evaluation.settings_override import (
    apply_experiment_overrides,
)


@dataclass(frozen=True)
class ExperimentPlan:
    """The effective settings and index assignment for one experiment."""

    name: str
    settings: Settings
    index_fingerprint: str
    collection_name: str
    reuses_index_of: str | None = None


@dataclass(frozen=True)
class EvaluationSuiteReport:
    """Results and failures from an ordered experiment suite."""

    experiments: Mapping[str, Any]
    plans: Sequence[ExperimentPlan]
    elapsed_ms: float
    errors: Mapping[str, str]


class ExperimentRunner:
    """Plan and execute configured ablation experiments."""

    def __init__(
        self,
        base_settings: Settings,
        execute_experiment: Callable[[ExperimentPlan], Any],
        benchmark_provider: str | None = None,
        dataset_version: str | None = None,
    ) -> None:
        if not callable(execute_experiment):
            raise TypeError("execute_experiment must be callable")

        self._base_settings = base_settings
        self._execute_experiment = execute_experiment
        self._benchmark_provider = benchmark_provider
        self._dataset_version = dataset_version

    def plan(
        self,
        experiment_names: Sequence[str] | None = None,
    ) -> list[ExperimentPlan]:
        """Build ordered plans for enabled, selected experiments."""

        experiments = self._configured_experiments()
        self._validate_experiments(experiments)
        selected_names = self._validate_selection(experiments, experiment_names)

        plans: list[ExperimentPlan] = []
        first_plan_by_fingerprint: dict[str, str] = {}
        provider, version = self._benchmark_identity()

        for experiment in experiments:
            if not experiment.enabled or experiment.name not in selected_names:
                continue

            effective_settings = apply_experiment_overrides(
                self._base_settings,
                experiment,
            )
            fingerprint = build_index_fingerprint(
                effective_settings,
                benchmark_provider=provider,
                dataset_version=version,
            )
            first_plan_name = first_plan_by_fingerprint.get(fingerprint)
            plans.append(
                ExperimentPlan(
                    name=experiment.name,
                    settings=effective_settings,
                    index_fingerprint=fingerprint,
                    collection_name=build_collection_name(provider, fingerprint),
                    reuses_index_of=first_plan_name,
                )
            )
            first_plan_by_fingerprint.setdefault(fingerprint, experiment.name)

        if not plans:
            raise ValueError("No enabled experiments were selected")
        return plans

    def run(
        self,
        experiment_names: Sequence[str] | None = None,
    ) -> EvaluationSuiteReport:
        """Execute planned experiments in order, isolating callback failures."""

        started = time.perf_counter()
        plans = self.plan(experiment_names)
        reports: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for plan in plans:
            try:
                reports[plan.name] = self._execute_experiment(plan)
            except Exception as exc:
                errors[plan.name] = str(exc) or type(exc).__name__

        return EvaluationSuiteReport(
            experiments=reports,
            plans=plans,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            errors=errors,
        )

    def _configured_experiments(self) -> list[EvaluationExperimentSettings]:
        configured = list(self._base_settings.evaluation.experiments)
        if configured:
            return configured
        return [EvaluationExperimentSettings(name="baseline")]

    @staticmethod
    def _validate_experiments(
        experiments: Sequence[EvaluationExperimentSettings],
    ) -> None:
        names: set[str] = set()
        for experiment in experiments:
            name = experiment.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Experiment names must be nonempty strings")
            if name in names:
                raise ValueError(f"Duplicate experiment name: {name!r}")
            names.add(name)

        if not any(experiment.enabled for experiment in experiments):
            raise ValueError("At least one experiment must be enabled")

    @staticmethod
    def _validate_selection(
        experiments: Sequence[EvaluationExperimentSettings],
        experiment_names: Sequence[str] | None,
    ) -> set[str]:
        configured_names = {experiment.name for experiment in experiments}
        if experiment_names is None:
            return configured_names
        if isinstance(experiment_names, (str, bytes)):
            raise TypeError("experiment_names must be a sequence of names")

        requested = list(experiment_names)
        if any(not isinstance(name, str) or not name.strip() for name in requested):
            raise ValueError("Requested experiment names must be nonempty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("Requested experiment names must be unique")

        unknown = [name for name in requested if name not in configured_names]
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"Unknown experiment name(s): {rendered}")
        return set(requested)

    def _benchmark_identity(self) -> tuple[str, str]:
        benchmark = self._base_settings.evaluation.benchmark
        provider = self._benchmark_provider
        version = self._dataset_version

        if provider is None:
            provider = benchmark.provider if benchmark is not None else "financebench"
        if version is None:
            version = benchmark.split if benchmark is not None else "open_source"
        return provider, version


__all__ = [
    "EvaluationSuiteReport",
    "ExperimentPlan",
    "ExperimentRunner",
]
