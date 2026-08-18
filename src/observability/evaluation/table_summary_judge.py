"""LLM-as-Judge rubric for retrieval-oriented table summaries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.libs.llm import BaseLLM, LLMFactory, Message

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_SCORE_FIELDS = (
    "faithfulness",
    "key_information_coverage",
    "retrieval_utility",
    "trend_relationship_quality",
    "conciseness_clarity",
)
_WEIGHTS = {
    "faithfulness": 0.40,
    "key_information_coverage": 0.20,
    "retrieval_utility": 0.20,
    "trend_relationship_quality": 0.10,
    "conciseness_clarity": 0.10,
}
_FAITHFULNESS_CAPS = {0: 19.0, 1: 39.0, 2: 59.0, 3: 84.0, 4: 100.0}


@dataclass(frozen=True)
class TableSummaryJudgement:
    """Validated rubric scores and deterministic aggregate score."""

    faithfulness: int
    key_information_coverage: int
    retrieval_utility: int
    trend_relationship_quality: int
    conciseness_clarity: int
    unsupported_claims: tuple[str, ...] = ()
    missing_key_information: tuple[str, ...] = ()
    reason: str = ""

    @property
    def raw_score(self) -> float:
        """Return the weighted score before factual-accuracy capping."""
        weighted = sum(
            getattr(self, field) * weight for field, weight in _WEIGHTS.items()
        )
        return round(weighted / 4 * 100, 2)

    @property
    def overall_score(self) -> float:
        """Return a 0-100 score capped by the faithfulness band."""
        return min(self.raw_score, _FAITHFULNESS_CAPS[self.faithfulness])

    @property
    def quality_level(self) -> str:
        """Return a stable human-readable quality band."""
        score = self.overall_score
        if score >= 90:
            return "excellent"
        if score >= 75:
            return "good"
        if score >= 60:
            return "usable"
        if score >= 40:
            return "poor"
        return "reject"


class LLMTableSummaryJudge:
    """Evaluate a table summary with the global configured LLM."""

    def __init__(
        self,
        settings: Any,
        *,
        llm: BaseLLM | None = None,
        prompt_path: str | Path = "config/prompts/table_summary_judge.txt",
        max_output_tokens: int = 1200,
    ) -> None:
        if settings is None and llm is None:
            raise ValueError("settings or an injected llm is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.llm = llm if llm is not None else LLMFactory.create(settings)
        self.max_output_tokens = max_output_tokens
        path = resolve_path(prompt_path)
        self.prompt = path.read_text(encoding="utf-8").strip()
        if not self.prompt:
            raise ValueError("table summary judge prompt cannot be empty")

    def judge(
        self,
        table_source: str,
        summary: str,
        *,
        document_name: str | None = None,
        page_range: str | None = None,
        section_path: str | None = None,
        table_title: str | None = None,
        previous_context: str | None = None,
        footnotes: list[str] | None = None,
        next_context: str | None = None,
        table_unit_count: int = 1,
        trace: Any = None,
    ) -> TableSummaryJudgement:
        """Judge one summary against exactly the source supplied to its generator."""
        judgement, _response = self.judge_with_response(
            table_source,
            summary,
            document_name=document_name,
            page_range=page_range,
            section_path=section_path,
            table_title=table_title,
            previous_context=previous_context,
            footnotes=footnotes,
            next_context=next_context,
            table_unit_count=table_unit_count,
            trace=trace,
        )
        return judgement

    def judge_with_response(
        self,
        table_source: str,
        summary: str,
        *,
        document_name: str | None = None,
        page_range: str | None = None,
        section_path: str | None = None,
        table_title: str | None = None,
        previous_context: str | None = None,
        footnotes: list[str] | None = None,
        next_context: str | None = None,
        table_unit_count: int = 1,
        trace: Any = None,
    ) -> tuple[TableSummaryJudgement, Any]:
        """Return the judgement and provider response for observability."""
        if not table_source.strip():
            raise ValueError("table_source cannot be empty")
        if not summary.strip():
            raise ValueError("summary cannot be empty")
        if table_unit_count <= 0:
            raise ValueError("table_unit_count must be positive")
        user_prompt = "\n".join(
            part
            for part in (
                self._element("document_name", document_name),
                self._element("page_range", page_range),
                self._element("section_path", section_path),
                self._element("table_title", table_title),
                f"<table_unit_count>{table_unit_count}</table_unit_count>",
                self._element("previous_context", previous_context),
                self._element("table_source", table_source),
                *[
                    self._element("footnote", footnote)
                    for footnote in (footnotes or [])
                    if footnote.strip()
                ],
                self._element("next_context", next_context),
                self._element("candidate_summary", summary),
            )
            if part
        )
        response = self.llm.chat(
            [
                Message(role="system", content=self.prompt),
                Message(role="user", content=user_prompt),
            ],
            trace=trace,
            temperature=0.0,
            max_tokens=self.max_output_tokens,
        )
        return self._parse_response(response.content), response

    @classmethod
    def _element(cls, name: str, value: str | None) -> str:
        if not value:
            return ""
        return f"<{name}><![CDATA[{cls._cdata(value)}]]></{name}>"

    @staticmethod
    def _cdata(value: str) -> str:
        return value.replace("]]>", "]]]]><![CDATA[>")

    @staticmethod
    def _parse_response(content: str) -> TableSummaryJudgement:
        match = _JSON_OBJECT_RE.search(content)
        if match is None:
            raise ValueError("table summary judge did not return a JSON object")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("table summary judge returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("table summary judgement must be a JSON object")

        scores: dict[str, int] = {}
        for field in _SCORE_FIELDS:
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError(f"{field} must be an integer from 0 to 4")
            scores[field] = value

        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")

        return TableSummaryJudgement(
            **scores,
            unsupported_claims=LLMTableSummaryJudge._string_list(
                payload.get("unsupported_claims"),
                "unsupported_claims",
            ),
            missing_key_information=LLMTableSummaryJudge._string_list(
                payload.get("missing_key_information"),
                "missing_key_information",
            ),
            reason=reason.strip(),
        )

    @staticmethod
    def _string_list(value: Any, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} must be a list of strings")
        return tuple(item.strip() for item in value if item.strip())
