"""Unit tests for EvalRunner and golden test set loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.eval_runner import (
    EvalReport,
    EvalRunner,
    GoldenTestCase,
    QueryResult,
    load_test_set,
)

# ── Fixtures / Helpers ────────────────────────────────────────────


class StubEvaluator(BaseEvaluator):
    """Evaluator that returns fixed metrics for testing."""

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"hit_rate": 1.0, "mrr": 0.5}


def _write_golden_json(path: Path, test_cases: List[Dict]) -> None:
    data = {"test_cases": test_cases}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── Tests: load_test_set ──────────────────────────────────────────


class TestLoadTestSet:
    def test_load_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "golden.json"
        _write_golden_json(f, [
            {"query": "What is RAG?", "expected_chunk_ids": ["c1"]},
            {"query": "How does BM25 work?"},
        ])

        cases = load_test_set(f)

        assert len(cases) == 2
        assert cases[0].query == "What is RAG?"
        assert cases[0].expected_chunk_ids == ["c1"]
        assert cases[1].expected_chunk_ids == []

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_test_set("nonexistent.json")

    def test_load_invalid_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text('{"wrong_key": []}', encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'test_cases'"):
            load_test_set(f)


# ── Tests: TestCase ───────────────────────────────────────────────


class TestGoldenTestCase:
    def test_from_dict_full(self) -> None:
        tc = GoldenTestCase.from_dict({
            "query": "Q",
            "expected_chunk_ids": ["a", "b"],
            "expected_sources": ["doc.pdf"],
            "reference_answer": "Answer",
        })
        assert tc.query == "Q"
        assert tc.expected_chunk_ids == ["a", "b"]
        assert tc.expected_sources == ["doc.pdf"]
        assert tc.reference_answer == "Answer"

    def test_from_dict_minimal(self) -> None:
        tc = GoldenTestCase.from_dict({"query": "Q"})
        assert tc.expected_chunk_ids == []
        assert tc.reference_answer is None


# ── Tests: EvalRunner ─────────────────────────────────────────────


class TestEvalRunner:
    def test_run_without_evaluator_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        runner = EvalRunner(evaluator=None)
        with pytest.raises(ValueError, match="requires an evaluator"):
            runner.run(f)

    def test_run_empty_test_set_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [])

        runner = EvalRunner(evaluator=StubEvaluator())
        with pytest.raises(ValueError, match="empty"):
            runner.run(f)

    def test_run_with_stub_evaluator(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [
            {"query": "What is RAG?"},
            {"query": "How does hybrid search work?"},
        ])

        runner = EvalRunner(evaluator=StubEvaluator())
        report = runner.run(f)

        assert isinstance(report, EvalReport)
        assert len(report.query_results) == 2
        assert report.aggregate_metrics["hit_rate"] == 1.0
        assert report.aggregate_metrics["mrr"] == 0.5
        assert report.total_elapsed_ms > 0

    def test_run_with_hybrid_search(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [
            {"query": "RAG", "expected_chunk_ids": ["c1"]},
        ])

        mock_search = MagicMock()
        mock_search.search.return_value = [
            MagicMock(chunk_id="c1", text="RAG is...", score=0.9),
        ]

        runner = EvalRunner(
            hybrid_search=mock_search,
            evaluator=StubEvaluator(),
        )
        report = runner.run(f)

        assert len(report.query_results) == 1
        assert report.query_results[0].retrieved_chunk_ids == ["c1"]

    def test_run_with_answer_generator(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        def gen(query, chunks):
            return f"Generated answer for: {query}"

        runner = EvalRunner(
            evaluator=StubEvaluator(),
            answer_generator=gen,
        )
        report = runner.run(f)

        assert "Generated answer for: Q" == report.query_results[0].generated_answer

    def test_retrieval_only_metrics_skip_answer_generation(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])
        generator = MagicMock(return_value="unused")
        settings = SimpleNamespace(
            evaluation=SimpleNamespace(
                provider="composite",
                backends=["benchmark"],
                metrics=[
                    "document_hit_rate@5",
                    "page_hit_rate@5",
                    "evidence_hit_rate@5",
                    "evidence_mrr@5",
                ],
            )
        )
        runner = EvalRunner(
            settings=settings,
            evaluator=StubEvaluator(),
            answer_generator=generator,
        )

        report = runner.run(f)

        generator.assert_not_called()
        assert report.query_results[0].generated_answer is None

    def test_answer_metric_enables_answer_generation(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])
        calls = []

        def generator(query, chunks):
            calls.append((query, chunks))
            return "Generated answer"

        settings = SimpleNamespace(
            evaluation=SimpleNamespace(
                provider="benchmark",
                backends=[],
                metrics=["answer_token_f1"],
            )
        )
        runner = EvalRunner(
            settings=settings,
            evaluator=StubEvaluator(),
            answer_generator=generator,
        )

        report = runner.run(f)

        assert calls == [("Q", [])]
        assert report.query_results[0].generated_answer == "Generated answer"

    def test_report_to_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "g.json"
        _write_golden_json(f, [{"query": "Q"}])

        runner = EvalRunner(evaluator=StubEvaluator())
        report = runner.run(f)

        d = report.to_dict()
        assert "aggregate_metrics" in d
        assert "query_results" in d
        assert d["query_count"] == 1
        assert d["evaluator_name"] == "StubEvaluator"

    def test_run_cases_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint.jsonl"
        cached = QueryResult(
            query="Q1",
            case_id="case-1",
            metrics={"hit_rate": 0.25},
            elapsed_ms=12.0,
        )
        checkpoint.write_text(
            json.dumps({"case_key": "id:case-1", "result": cached.to_dict()}) + "\n",
            encoding="utf-8",
        )
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator.evaluate.return_value = {"hit_rate": 1.0}
        runner = EvalRunner(evaluator=evaluator)
        cases = [
            GoldenTestCase(query="Q1", case_id="case-1"),
            GoldenTestCase(query="Q2", case_id="case-2"),
        ]

        report = runner.run_cases(cases, checkpoint_path=checkpoint)

        assert [result.case_id for result in report.query_results] == ["case-1", "case-2"]
        assert report.query_results[0].metrics == {"hit_rate": 0.25}
        evaluator.evaluate.assert_called_once()
        assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 2


class TestEvalRunnerAggregation:
    """Test metric aggregation logic."""

    def test_aggregate_averages_correctly(self) -> None:
        results = [
            QueryResult(query="q1", metrics={"hit_rate": 1.0, "mrr": 1.0}),
            QueryResult(query="q2", metrics={"hit_rate": 0.0, "mrr": 0.5}),
        ]

        avg = EvalRunner._aggregate_metrics(results)

        assert avg["hit_rate"] == pytest.approx(0.5)
        assert avg["mrr"] == pytest.approx(0.75)

    def test_aggregate_empty_returns_empty(self) -> None:
        assert EvalRunner._aggregate_metrics([]) == {}

    def test_aggregate_partial_metrics(self) -> None:
        """When some queries have metrics that others don't."""
        results = [
            QueryResult(query="q1", metrics={"hit_rate": 1.0}),
            QueryResult(query="q2", metrics={"faithfulness": 0.9}),
        ]

        avg = EvalRunner._aggregate_metrics(results)

        # Each metric averaged over only the queries that produced it
        assert avg["hit_rate"] == 1.0
        assert avg["faithfulness"] == 0.9


# ── Tests: Golden test set fixture ────────────────────────────────


class TestGoldenTestSetFixture:
    """Validate the actual golden test set file exists and is valid."""

    def test_golden_set_loads(self) -> None:
        golden_path = Path("tests/fixtures/golden_test_set.json")
        if not golden_path.exists():
            pytest.skip("Golden test set not present")

        cases = load_test_set(golden_path)
        assert len(cases) >= 1
        for tc in cases:
            assert tc.query.strip(), "Query must be non-empty"
