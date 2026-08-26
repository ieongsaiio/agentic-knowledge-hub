"""LLM-backed query-understanding and rewriting provider."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.core.settings import resolve_path
from src.libs.llm import LLMFactory, Message
from src.libs.query_rewriter.base_query_rewriter import (
    BaseQueryRewriter,
    QueryRewritePlan,
    QuerySlot,
    RewrittenQuery,
    SourceHint,
)

logger = logging.getLogger(__name__)


class LLMQueryRewriter(BaseQueryRewriter):
    """Generate a validated rewrite plan with the configured text LLM."""

    def __init__(self, settings: Any, llm: Any | None = None) -> None:
        config = settings.retrieval.query_rewriter
        llm_settings = replace(settings.llm, **dict(config.llm or {}))
        self._llm = llm or LLMFactory.create(replace(settings, llm=llm_settings))
        self._max_queries = config.max_queries
        self._fail_on_error = config.fail_on_error
        self._system_prompt = self._load_prompt(config.prompt_path)

    @staticmethod
    def _load_prompt(prompt_path: str) -> str:
        path: Path = resolve_path(prompt_path)
        prompt = path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Query rewriter prompt is empty: {path}")
        return prompt

    def rewrite(
        self,
        query: str,
        *,
        context: str | None = None,
        trace: Any | None = None,
    ) -> QueryRewritePlan:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        context_text = (context or "").strip() or "None"
        user_prompt = (
            "<context>\n"
            f"{context_text}\n"
            "</context>\n\n"
            "<question>\n"
            f"{query}\n"
            "</question>"
        )
        try:
            response = self._llm.chat(
                [
                    Message(role="system", content=self._system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                trace=trace,
            )
            return self._parse_plan(query, response.content)
        except Exception as exc:
            if self._fail_on_error:
                raise RuntimeError(f"Query rewriting failed: {exc}") from exc
            logger.warning("Query rewriting failed; using original query: %s", exc)
            return QueryRewritePlan(
                original_query=query,
                rewrite_needed=False,
                reason=f"rewriter_fallback: {exc}",
            )

    def _parse_plan(self, original_query: str, content: str) -> QueryRewritePlan:
        payload = self._extract_json(content)
        decision = payload.get("rewrite_decision") or {}
        if not isinstance(decision, dict):
            raise ValueError("rewrite_decision must be an object")

        slots = [
            QuerySlot(
                slot_type=str(item.get("slot_type", "other")).strip() or "other",
                value=str(item.get("value", "")).strip(),
                role=str(item.get("role", "")).strip(),
                must_preserve=bool(item.get("must_preserve", True)),
            )
            for item in self._object_list(payload.get("slots"), "slots")
            if str(item.get("value", "")).strip()
        ]
        source_hints = [
            SourceHint(
                document_type=str(item.get("document_type", "")).strip(),
                section=str(item.get("section", "")).strip(),
                content_form=str(item.get("content_form", "unknown")).strip()
                or "unknown",
                confidence=str(item.get("confidence", "low")).strip() or "low",
            )
            for item in self._object_list(payload.get("source_hints"), "source_hints")
        ]

        seen = {original_query.casefold()}
        rewrites = []
        for index, item in enumerate(
            self._object_list(payload.get("rewritten_queries"), "rewritten_queries"),
            start=1,
        ):
            rewritten = str(item.get("query", "")).strip()
            identity = rewritten.casefold()
            if not rewritten or identity in seen:
                continue
            seen.add(identity)
            rewrites.append(
                RewrittenQuery(
                    query_id=str(item.get("query_id", f"q{index}")).strip()
                    or f"q{index}",
                    purpose=str(item.get("purpose", "")).strip(),
                    query=rewritten,
                )
            )
            if len(rewrites) >= self._max_queries:
                break

        requested = bool(decision.get("rewrite_needed", False))
        rewrite_needed = requested and bool(rewrites)
        return QueryRewritePlan(
            original_query=original_query,
            question_types=self._string_list(payload.get("question_types")),
            slots=slots,
            source_hints=source_hints,
            rewrite_needed=rewrite_needed,
            strategy=str(decision.get("strategy", "")).strip(),
            reason=str(decision.get("reason", "")).strip(),
            rewritten_queries=rewrites if rewrite_needed else [],
        )

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        text = content.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("LLM response does not contain a JSON object")
            text = text[start : end + 1]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("LLM response JSON must be an object")
        return payload

    @staticmethod
    def _object_list(value: Any, field_name: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError(f"{field_name} must be a list of objects")
        return value

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("question_types must be a list")
        return [str(item).strip() for item in value if str(item).strip()]
