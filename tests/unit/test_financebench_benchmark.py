"""Offline unit tests for the FinanceBench benchmark provider."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import pytest

from src.core.settings import BenchmarkSettings
from src.libs.benchmark import BenchmarkFactory, FinanceBenchBenchmark
from src.libs.benchmark.financebench_benchmark import normalize_financebench_case


def _raw_case(index: int, **overrides: Any) -> dict[str, Any]:
    case = {
        "financebench_id": f"fb-{index}",
        "question": f"Question {index}?",
        "answer": f"Answer {index}.",
        "doc_name": f"report-{index}.pdf",
        "question_type": "numeric",
        "question_reasoning": f"Reasoning {index}",
        "company": f"Company {index}",
        "justification": f"Justification {index}",
        "evidence": [
            {
                "evidence_page_num": index,
                "evidence_text": f"Evidence {index}",
            }
        ],
    }
    case.update(overrides)
    return case


def _write_cached_artifacts(data_dir: Path, count: int = 5) -> list[dict[str, Any]]:
    rows = [_raw_case(index) for index in range(count)]
    dataset_path = data_dir / "data" / "financebench_open_source.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    pdf_dir = data_dir / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "cached-report.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    return rows


def _settings(
    data_dir: Path,
    *,
    auto_download: bool,
    sample_size: int | None = None,
    seed: int = 42,
) -> BenchmarkSettings:
    return BenchmarkSettings(
        provider="FinanceBench",
        source_url="https://example.invalid/financebench.zip",
        data_dir=str(data_dir),
        auto_download=auto_download,
        sample_size=sample_size,
        seed=seed,
    )


def test_normalize_financebench_case_converts_page_and_preserves_metadata() -> None:
    raw = _raw_case(
        0,
        evidence=[
            {
                "evidence_doc_name": "evidence-report.pdf",
                "evidence_page_num": 0,
                "evidence_text": "Revenue increased.",
            }
        ],
    )

    case = normalize_financebench_case(raw, row_number=7)

    assert case.case_id == "fb-0"
    assert case.query == "Question 0?"
    assert case.reference_answer == "Answer 0."
    assert case.expected_documents == ["evidence-report.pdf"]
    assert case.expected_pages == [1]
    assert case.expected_evidence == ["Revenue increased."]
    assert case.metadata == {
        "doc_name": "report-0.pdf",
        "question_type": "numeric",
        "question_reasoning": "Reasoning 0",
        "reasoning": "Reasoning 0",
        "company": "Company 0",
        "justification": "Justification 0",
    }


@pytest.mark.parametrize("entrypoint", ["prepare", "load_cases"])
def test_factory_uses_configured_deterministic_sample(tmp_path: Path, entrypoint: str) -> None:
    rows = _write_cached_artifacts(tmp_path)
    settings = _settings(tmp_path, auto_download=False, sample_size=3, seed=17)
    expected_ids = [
        rows[index]["financebench_id"]
        for index in random.Random(settings.seed).sample(range(len(rows)), settings.sample_size)
    ]

    benchmark = BenchmarkFactory.create(settings)
    load = getattr(benchmark, entrypoint)
    first_ids = [case.case_id for case in load()]
    second_ids = [case.case_id for case in load()]

    assert isinstance(benchmark, FinanceBenchBenchmark)
    assert first_ids == second_ids == expected_ids


def test_existing_cache_is_idempotent_and_never_requests_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cached_artifacts(tmp_path)
    benchmark = FinanceBenchBenchmark(_settings(tmp_path, auto_download=True))

    def fail_if_requested(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("valid cached FinanceBench artifacts must not use network")

    monkeypatch.setattr(
        "src.libs.benchmark.financebench_benchmark.requests.get",
        fail_if_requested,
    )

    assert benchmark.download() == benchmark.dataset_path
    assert benchmark.download() == benchmark.dataset_path
    assert len(benchmark.prepare()) == 5


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ([], "'evidence' must be a non-empty list"),
        ([None], "evidence[0] must be an object"),
        (
            [{"evidence_page_num": -1, "evidence_text": "text"}],
            "evidence[0].evidence_page_num cannot be negative",
        ),
        (
            [{"evidence_page_num": True, "evidence_text": "text"}],
            "evidence[0].evidence_page_num must be a zero-based integer",
        ),
        (
            [{"evidence_page_num": 0, "evidence_text": ""}],
            "evidence[0].evidence_text must be a non-empty string",
        ),
    ],
)
def test_malformed_evidence_reports_row_context(evidence: list[Any], message: str) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"Malformed FinanceBench case at row 4: {message}"),
    ):
        normalize_financebench_case(_raw_case(1, evidence=evidence), row_number=4)


def test_auto_download_false_rejects_missing_artifacts(tmp_path: Path) -> None:
    benchmark = FinanceBenchBenchmark(_settings(tmp_path, auto_download=False))

    with pytest.raises(
        FileNotFoundError,
        match="artifacts are missing and auto_download is disabled",
    ):
        benchmark.prepare()
