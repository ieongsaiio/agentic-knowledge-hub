"""Factory for pluggable query-rewriting providers."""

from __future__ import annotations

from typing import Any

from src.libs.query_rewriter.base_query_rewriter import BaseQueryRewriter


class QueryRewriterFactory:
    """Create a query rewriter selected by retrieval settings."""

    _PROVIDERS: dict[str, type[BaseQueryRewriter]] = {}

    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: type[BaseQueryRewriter],
    ) -> None:
        if not issubclass(provider_class, BaseQueryRewriter):
            raise ValueError("Query rewriter provider must inherit BaseQueryRewriter")
        cls._PROVIDERS[name.strip().lower()] = provider_class

    @classmethod
    def create(cls, settings: Any, **kwargs: Any) -> BaseQueryRewriter:
        config = settings.retrieval.query_rewriter
        provider_name = config.provider.strip().lower()
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS)) or "none"
            raise ValueError(
                f"Unsupported query rewriter provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        return provider_class(settings=settings, **kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        return sorted(cls._PROVIDERS)

