"""Factory for creating benchmark provider instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.core.settings import BenchmarkSettings, Settings
    from src.libs.benchmark.base_benchmark import BaseBenchmark


def _get_base_benchmark() -> type[BaseBenchmark]:
    """Import the benchmark base class only when validation is needed."""
    from src.libs.benchmark.base_benchmark import BaseBenchmark

    return BaseBenchmark


def _get_financebench_benchmark() -> type[BaseBenchmark]:
    """Lazy import FinanceBenchBenchmark to avoid import cycles."""
    from src.libs.benchmark.financebench_benchmark import FinanceBenchBenchmark

    return FinanceBenchBenchmark


class BenchmarkFactory:
    """Create benchmark implementations selected by configuration."""

    _PROVIDERS: dict[str, type[BaseBenchmark]] = {}
    _LAZY_PROVIDERS: dict[str, Callable[[], type[BaseBenchmark]]] = {
        "financebench": _get_financebench_benchmark,
    }

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[BaseBenchmark],
    ) -> None:
        """Register a benchmark provider implementation."""
        base_benchmark = _get_base_benchmark()
        if not issubclass(provider_class, base_benchmark):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseBenchmark"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        settings: Settings | BenchmarkSettings,
    ) -> BaseBenchmark:
        """Create the configured benchmark provider."""
        available = cls._format_available_providers()

        try:
            if hasattr(settings, "evaluation"):
                evaluation_settings = settings.evaluation
                if evaluation_settings is None:
                    raise AttributeError("settings.evaluation is None")
                benchmark_settings = evaluation_settings.benchmark
            else:
                benchmark_settings = settings

            if benchmark_settings is None:
                raise AttributeError("benchmark settings are None")

            provider = benchmark_settings.provider
            if not isinstance(provider, str) or not provider.strip():
                raise AttributeError("benchmark provider is missing")
            provider_name = provider.lower()
        except AttributeError as exc:
            raise ValueError(
                "Missing required configuration: "
                "settings.evaluation.benchmark.provider. "
                f"Available providers: {available}."
            ) from exc

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            loader = cls._LAZY_PROVIDERS.get(provider_name)
            if loader is not None:
                provider_class = loader()
                cls._PROVIDERS[provider_name] = provider_class

        if provider_class is None:
            raise ValueError(
                f"Unsupported Benchmark provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=benchmark_settings)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate Benchmark provider '{provider_name}': {exc}"
            ) from exc

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return all registered and built-in provider names."""
        return sorted(set(cls._PROVIDERS) | set(cls._LAZY_PROVIDERS))

    @classmethod
    def _format_available_providers(cls) -> str:
        providers = cls.list_providers()
        return ", ".join(providers) if providers else "none"
