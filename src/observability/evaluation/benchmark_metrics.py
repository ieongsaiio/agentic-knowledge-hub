"""Deterministic, offline metrics for document-grounded benchmarks."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from src.libs.benchmark.base_benchmark import BenchmarkCase

__all__ = [
    "BenchmarkMetrics",
    "aggregate",
    "normalize_document_name",
    "normalize_evidence_text",
    "parse_metric_name",
]


_RANKED_METRICS = {
    "document_hit_rate",
    "document_mrr",
    "page_hit_rate",
    "evidence_hit_rate",
    "evidence_mrr",
}
_ANSWER_METRICS = {
    "answer_exact_match",
    "answer_token_f1",
    "numeric_accuracy",
}
_DOCUMENT_FIELDS = (
    "document_name",
    "doc_name",
    "source_path",
    "file_path",
    "filename",
    "file_name",
    "source",
    "document",
    "path",
    "title",
    "source_ref",
    "document_id",
    "doc_id",
)
_TEXT_FIELDS = ("text", "content", "page_content", "chunk_text")
_ID_FIELDS = ("chunk_id", "id", "document_id", "doc_id")
_SINGLE_PAGE_FIELDS = ("page_num", "page", "page_number")
_NUMERIC_REL_TOLERANCE = 0.01
_NUMERIC_ABS_TOLERANCE = 1e-6

_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<open>\()?
    \s*
    (?P<sign1>[+-])?
    \s*
    (?P<currency>
        US\$|USD|EUR|GBP|CNY|RMB|JPY|CAD|AUD|HKD|[$€£¥]
    )?
    \s*
    (?P<sign2>[+-])?
    \s*
    (?P<number>
        (?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?
        |
        \.\d+
    )
    \s*
    (?P<suffix>
        %|percent(?:age)?
        |thousand|million|billion|trillion
        |k|m|bn|b
    )?
    \s*
    (?P<close>\))?
    (?![\w])
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALE_FACTORS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}


def normalize_document_name(value: Any) -> str:
    """Return a case-insensitive basename stem for a document reference."""
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = normalized.strip("\"'")
    normalized = normalized.split("?", 1)[0].split("#", 1)[0]
    normalized = normalized.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""

    name = PurePosixPath(normalized).name
    return PurePosixPath(name).stem.casefold().strip()


def normalize_evidence_text(value: Any) -> str:
    """Normalize Unicode, case, punctuation, and whitespace for comparison."""
    if value is None:
        return ""

    value = unicodedata.normalize("NFKC", str(value)).casefold()
    characters: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if character.isspace() or category[0] in {"P", "S", "Z"}:
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def parse_metric_name(name: str) -> tuple[str, int | None]:
    """Parse a metric name and optional positive ``@k`` suffix."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("metric name must be a non-empty string")

    normalized = name.strip().casefold()
    if normalized.count("@") > 1:
        raise ValueError(f"invalid metric name: {name!r}")

    if "@" in normalized:
        base, raw_k = normalized.rsplit("@", 1)
        if not base or not raw_k.isdigit() or int(raw_k) < 1:
            raise ValueError(f"invalid metric cutoff: {name!r}")
        k: int | None = int(raw_k)
    else:
        base = normalized
        k = None

    if base == "mrr":
        base = "document_mrr"
    return base, k


@dataclass(frozen=True)
class _Metric:
    output_name: str
    base: str
    k: int | None


@dataclass(frozen=True)
class _Evidence:
    document: str
    page: int | None
    text: str


@dataclass(frozen=True)
class _Retrieved:
    text: str
    metadata: Mapping[str, Any]
    identifier: str
    documents: frozenset[str]
    pages: tuple[tuple[int, int], ...]


def _read_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)
    try:
        return getattr(value, field, default)
    except Exception:
        return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    try:
        return vars(value)
    except TypeError:
        return {}


def _first_text(item: Any, metadata: Mapping[str, Any]) -> str:
    if isinstance(item, str):
        return item
    for container in (item, metadata):
        for field in _TEXT_FIELDS:
            value = _read_field(container, field)
            if value is not None:
                return str(value)
    return ""


def _first_identifier(item: Any, metadata: Mapping[str, Any]) -> str:
    for container in (item, metadata):
        for field in _ID_FIELDS:
            value = _read_field(container, field)
            if value is not None and str(value).strip():
                return str(value)
    return ""


def _document_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [nested for field in _DOCUMENT_FIELDS if (nested := value.get(field)) is not None]
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def _extract_documents(item: Any, metadata: Mapping[str, Any], identifier: str) -> frozenset[str]:
    documents: set[str] = set()
    for container in (item, metadata):
        for field in _DOCUMENT_FIELDS:
            raw_value = _read_field(container, field)
            if raw_value is None:
                continue
            for value in _document_values(raw_value):
                document = normalize_document_name(value)
                if document:
                    documents.add(document)

    if not documents and identifier:
        normalized_id = normalize_document_name(identifier)
        if normalized_id:
            documents.add(normalized_id)
    return frozenset(documents)


def _coerce_page(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _extract_pages(item: Any, metadata: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    pages: list[tuple[int, int]] = []
    for container in (item, metadata):
        start = _coerce_page(_read_field(container, "page_start"))
        end = _coerce_page(_read_field(container, "page_end"))
        if start is not None or end is not None:
            start = end if start is None else start
            end = start if end is None else end
            assert start is not None and end is not None
            pages.append((min(start, end), max(start, end)))

        for field in _SINGLE_PAGE_FIELDS:
            page = _coerce_page(_read_field(container, field))
            if page is not None:
                pages.append((page, page))

    return tuple(dict.fromkeys(pages))


def _extract_retrieved(item: Any) -> _Retrieved:
    metadata = _as_mapping(_read_field(item, "metadata", {}))
    identifier = _first_identifier(item, metadata)
    return _Retrieved(
        text=_first_text(item, metadata),
        metadata=metadata,
        identifier=identifier,
        documents=_extract_documents(item, metadata, identifier),
        pages=_extract_pages(item, metadata),
    )


def _extract_evidences(case: BenchmarkCase) -> list[_Evidence]:
    evidences: list[_Evidence] = []
    for evidence in _read_field(case, "evidences", []) or []:
        document = normalize_document_name(
            _read_field(
                evidence,
                "document_name",
                _read_field(evidence, "evidence_doc_name", ""),
            )
        )
        page = _coerce_page(
            _read_field(
                evidence,
                "page_number",
                _read_field(evidence, "page_num"),
            )
        )
        text = str(
            _read_field(
                evidence,
                "text",
                _read_field(evidence, "evidence_text", ""),
            )
            or ""
        )
        evidences.append(_Evidence(document=document, page=page, text=text))
    return evidences


def _token_counter(value: Any) -> Counter[str]:
    return Counter(normalize_evidence_text(value).split())


def _coverage(reference: Any, context: Any) -> float:
    reference_tokens = _token_counter(reference)
    if not reference_tokens:
        return 0.0
    context_tokens = _token_counter(context)
    covered = sum(min(count, context_tokens[token]) for token, count in reference_tokens.items())
    return covered / sum(reference_tokens.values())


def _evidence_text_matches(reference: str, retrieved_text: str) -> bool:
    expected = normalize_evidence_text(reference)
    actual = normalize_evidence_text(retrieved_text)
    if not expected or not actual:
        return False
    if expected in actual:
        return True
    return _coverage(expected, actual) >= 0.5


def _document_hit(item: _Retrieved, expected_documents: set[str]) -> bool:
    return bool(item.documents.intersection(expected_documents))


def _document_match_ranks(
    retrieved: Sequence[_Retrieved],
    evidences: Sequence[_Evidence],
) -> tuple[int | None, ...]:
    ranks: list[int | None] = []
    for evidence in evidences:
        if not evidence.document:
            continue
        first_rank = None
        for rank, item in enumerate(retrieved, start=1):
            if evidence.document in item.documents:
                first_rank = rank
                break
        ranks.append(first_rank)
    return tuple(ranks)


def _document_hit_rate(retrieved: Sequence[_Retrieved], evidences: Sequence[_Evidence]) -> float:
    match_ranks = _document_match_ranks(retrieved, evidences)
    if not match_ranks:
        return 0.0
    hits = sum(1 for rank in match_ranks if rank is not None)
    return hits / len(match_ranks)


def _document_mrr(retrieved: Sequence[_Retrieved], evidences: Sequence[_Evidence]) -> float:
    match_ranks = _document_match_ranks(retrieved, evidences)
    ranks = [rank for rank in match_ranks if rank is not None]
    if not ranks:
        return 0.0
    return 1.0 / min(ranks)


def _page_hit_rate(retrieved: Sequence[_Retrieved], evidences: Sequence[_Evidence]) -> float:
    evidence_count = 0
    hits = 0
    for evidence in evidences:
        if not evidence.document or evidence.page is None:
            continue
        evidence_count += 1
        for item in retrieved:
            if evidence.document not in item.documents:
                continue
            if item.pages:
                if any(start <= evidence.page <= end for start, end in item.pages):
                    hits += 1
                    break
            elif _evidence_text_matches(evidence.text, item.text):
                hits += 1
                break
    if evidence_count <= 0:
        return 0.0
    return hits / evidence_count


def _evidence_hit_rate(
    match_ranks: Sequence[int | None],
    evidence_count: int,
    cutoff: int,
) -> float:
    if evidence_count <= 0:
        return 0.0
    hits = sum(
        1
        for rank in match_ranks[:evidence_count]
        if rank is not None and rank <= cutoff
    )
    return hits / evidence_count


def _evidence_mrr(match_ranks: Sequence[int | None], cutoff: int) -> float:
    ranks_within_cutoff = [
        rank for rank in match_ranks if rank is not None and rank <= cutoff
    ]
    if not ranks_within_cutoff:
        return 0.0
    return 1.0 / min(ranks_within_cutoff)


def _answer_exact_match(reference: Any, answer: Any) -> float:
    expected = normalize_evidence_text(reference)
    actual = normalize_evidence_text(answer)
    return 1.0 if expected and expected == actual else 0.0


def _answer_token_f1(reference: Any, answer: Any) -> float:
    expected = _token_counter(reference)
    actual = _token_counter(answer)
    expected_count = sum(expected.values())
    actual_count = sum(actual.values())
    if expected_count == 0 or actual_count == 0:
        return 0.0

    common = sum((expected & actual).values())
    if common == 0:
        return 0.0
    precision = common / actual_count
    recall = common / expected_count
    return 2.0 * precision * recall / (precision + recall)


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []

    normalized = unicodedata.normalize("NFKC", str(value))
    candidates: list[tuple[bool, int, float]] = []
    for match in _NUMBER_RE.finditer(normalized):
        try:
            number = float(match.group("number").replace(",", ""))
        except ValueError:
            continue

        suffix = (match.group("suffix") or "").casefold()
        explicit = bool(match.group("currency") or suffix)
        if suffix in {"%", "percent", "percentage"}:
            number /= 100.0
        else:
            number *= _SCALE_FACTORS.get(suffix, 1.0)

        if (
            match.group("sign1") == "-"
            or match.group("sign2") == "-"
            or (match.group("open") and match.group("close"))
        ):
            number = -abs(number)
        candidates.append((explicit, match.start(), number))

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    return [candidate[2] for candidate in candidates]


def _numeric_accuracy(reference: Any, answer: Any) -> float | None:
    expected_values = _numeric_values(reference)
    if not expected_values:
        return None
    actual_values = _numeric_values(answer)
    if not actual_values:
        return 0.0

    expected = expected_values[0]
    return (
        1.0
        if any(
            math.isclose(
                expected,
                actual,
                rel_tol=_NUMERIC_REL_TOLERANCE,
                abs_tol=_NUMERIC_ABS_TOLERANCE,
            )
            for actual in actual_values
        )
        else 0.0
    )


class BenchmarkMetrics:
    """Compute configured benchmark metrics without network dependencies."""

    def __init__(self, metrics: Sequence[str]) -> None:
        if isinstance(metrics, str):
            metrics = [metrics]

        configured: list[_Metric] = []
        for raw_name in metrics:
            try:
                base, k = parse_metric_name(raw_name)
            except (TypeError, ValueError):
                continue

            if base in _RANKED_METRICS and k is not None:
                configured.append(_Metric(str(raw_name).strip().casefold(), base, k))
            elif base in _ANSWER_METRICS and k is None:
                configured.append(_Metric(str(raw_name).strip().casefold(), base, None))
        self.metrics = tuple(configured)

    def evaluate_case(
        self,
        case: BenchmarkCase,
        retrieved: list[Any],
        answer: str | None,
        evidence_ranks: Sequence[int | None] | None = None,
    ) -> dict[str, float]:
        """Evaluate one benchmark case using the configured metrics."""
        extracted = [_extract_retrieved(item) for item in (retrieved or [])]
        evidences = _extract_evidences(case)
        matched_evidence_ranks = tuple(evidence_ranks or ())
        reference_answer = _read_field(case, "reference_answer", "")
        results: dict[str, float] = {}

        for metric in self.metrics:
            top_k = extracted[: metric.k] if metric.k is not None else extracted
            if metric.base == "document_hit_rate":
                score = _document_hit_rate(top_k, evidences)
            elif metric.base == "document_mrr":
                score = _document_mrr(top_k, evidences)
            elif metric.base == "page_hit_rate":
                score = _page_hit_rate(top_k, evidences)
            elif metric.base == "evidence_hit_rate":
                assert metric.k is not None
                score = _evidence_hit_rate(
                    matched_evidence_ranks,
                    len(evidences),
                    metric.k,
                )
            elif metric.base == "evidence_mrr":
                assert metric.k is not None
                score = _evidence_mrr(matched_evidence_ranks, metric.k)
            elif metric.base == "answer_exact_match":
                score = _answer_exact_match(reference_answer, answer)
            elif metric.base == "answer_token_f1":
                score = _answer_token_f1(reference_answer, answer)
            elif metric.base == "numeric_accuracy":
                numeric_score = _numeric_accuracy(reference_answer, answer)
                if numeric_score is None:
                    continue
                score = numeric_score
            else:
                continue
            results[metric.output_name] = float(score)

        return results

    @staticmethod
    def aggregate(case_results: list[dict[str, float]]) -> dict[str, float]:
        """Average each metric over cases where that metric is applicable."""
        return aggregate(case_results)


def aggregate(case_results: list[dict[str, float]]) -> dict[str, float]:
    """Average each present metric, excluding omitted/non-applicable metrics."""
    metric_names = sorted(
        {metric_name for case_result in case_results for metric_name in case_result}
    )
    averages: dict[str, float] = {}
    for metric_name in metric_names:
        values = [
            float(case_result[metric_name])
            for case_result in case_results
            if metric_name in case_result
        ]
        if values:
            averages[metric_name] = sum(values) / len(values)
    return averages
