"""Contracts for pluggable query-rewriting providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuerySlot:
    """A retrieval constraint or entity extracted from the question."""

    slot_type: str
    value: str
    role: str = ""
    must_preserve: bool = True


@dataclass(frozen=True)
class SourceHint:
    """A soft hint about where supporting content may appear."""

    document_type: str = ""
    section: str = ""
    content_form: str = "unknown"
    confidence: str = "low"


@dataclass(frozen=True)
class RewrittenQuery:
    """One standalone retrieval query and its intended purpose."""

    query_id: str
    purpose: str
    query: str


@dataclass(frozen=True)
class QueryRewritePlan:
    """Structured output shared by every query-rewriting provider."""

    original_query: str
    question_types: list[str] = field(default_factory=list)
    slots: list[QuerySlot] = field(default_factory=list)
    source_hints: list[SourceHint] = field(default_factory=list)
    rewrite_needed: bool = False
    strategy: str = ""
    reason: str = ""
    rewritten_queries: list[RewrittenQuery] = field(default_factory=list)

    @property
    def queries(self) -> list[str]:
        """Return only non-empty rewritten query strings."""
        return [item.query for item in self.rewritten_queries if item.query.strip()]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseQueryRewriter(ABC):
    """Provider interface for optional query understanding and rewriting."""

    @abstractmethod
    def rewrite(
        self,
        query: str,
        *,
        context: str | None = None,
        trace: Any | None = None,
    ) -> QueryRewritePlan:
        """Analyze a question and return zero or more retrieval queries."""
