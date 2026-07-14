"""Unit tests for the Evaluation Panel dashboard page."""

from __future__ import annotations

import json
from pathlib import Path


class TestEvaluationPanelHelpers:
    """Test helper functions in evaluation_panel module."""

    def test_save_and_load_history(self, tmp_path: Path) -> None:
        """History round-trip: save then load."""
        from src.observability.dashboard.pages import evaluation_panel as ep

        # Temporarily override history path
        original = ep.EVAL_HISTORY_PATH
        ep.EVAL_HISTORY_PATH = tmp_path / "eval_history.jsonl"

        try:
            report = {
                "evaluator_name": "custom",
                "query_count": 2,
                "total_elapsed_ms": 123.4,
                "aggregate_metrics": {"hit_rate": 0.8},
            }

            ep._save_to_history(report)
            history = ep._load_history()

            assert len(history) == 1
            assert history[0]["evaluator_name"] == "custom"
            assert history[0]["aggregate_metrics"]["hit_rate"] == 0.8
            assert "timestamp" in history[0]
        finally:
            ep.EVAL_HISTORY_PATH = original

    def test_load_history_empty(self, tmp_path: Path) -> None:
        """Load returns empty list when no history file exists."""
        from src.observability.dashboard.pages import evaluation_panel as ep

        original = ep.EVAL_HISTORY_PATH
        ep.EVAL_HISTORY_PATH = tmp_path / "nonexistent.jsonl"

        try:
            assert ep._load_history() == []
        finally:
            ep.EVAL_HISTORY_PATH = original

    def test_load_history_tolerates_bad_lines(self, tmp_path: Path) -> None:
        """Malformed lines are skipped."""
        from src.observability.dashboard.pages import evaluation_panel as ep

        original = ep.EVAL_HISTORY_PATH
        hist_file = tmp_path / "eval_history.jsonl"
        hist_file.write_text(
            '{"ok": true}\nBAD LINE\n{"ok": false}\n',
            encoding="utf-8",
        )
        ep.EVAL_HISTORY_PATH = hist_file

        try:
            history = ep._load_history()
            assert len(history) == 2
            assert history[0]["ok"] is True
            assert history[1]["ok"] is False
        finally:
            ep.EVAL_HISTORY_PATH = original

    def test_save_history_creates_parent_dir(self, tmp_path: Path) -> None:
        """_save_to_history creates missing parent directories."""
        from src.observability.dashboard.pages import evaluation_panel as ep

        original = ep.EVAL_HISTORY_PATH
        ep.EVAL_HISTORY_PATH = tmp_path / "subdir" / "eval.jsonl"

        try:
            ep._save_to_history({"test": True})
            assert ep.EVAL_HISTORY_PATH.exists()
        finally:
            ep.EVAL_HISTORY_PATH = original

    def test_benchmark_results_helpers_load_metrics_and_latency(
        self,
        tmp_path: Path,
    ) -> None:
        """Benchmark report helpers expose metrics and latency columns."""
        from src.observability.dashboard.pages import evaluation_panel as ep

        run_dir = tmp_path / "20260707T000000_000000Z"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "created_at": "2026-07-07T00:00:00+00:00",
                    "case_count": 3,
                    "elapsed_ms": 600.0,
                    "plans": [
                        {
                            "name": "baseline",
                            "collection_name": "financebench__abc",
                            "index_fingerprint": "abc",
                            "reuses_index_of": None,
                            "effective_settings": {
                                "evaluation": {
                                    "metrics": [
                                        "document_hit_rate@5",
                                        "evidence_mrr@5",
                                    ]
                                }
                            },
                        }
                    ],
                    "successful_experiments": ["baseline"],
                    "errors": {},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "comparison.csv").write_text(
            "\n".join(
                [
                    "experiment,status,collection,index_fingerprint,"
                    "reuses_index_of,query_count,total_elapsed_ms,"
                    "document_hit_rate@5,evidence_mrr@5,error",
                    "baseline,success,financebench__abc,abc,,3,600,0.9,0.5,",
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "baseline.summary.json").write_text(
            json.dumps(
                {
                    "experiment": "baseline",
                    "collection": "financebench__abc",
                    "query_count": 3,
                    "total_elapsed_ms": 600,
                    "aggregate_metrics": {
                        "document_hit_rate@5": 0.9,
                        "evidence_mrr@5": 0.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        cases = [
            {
                "case_id": "c1",
                "query": "q1",
                "retrieved_chunk_ids": ["a"],
                "metrics": {"document_hit_rate@5": 1.0},
                "elapsed_ms": 100.0,
            },
            {
                "case_id": "c2",
                "query": "q2",
                "retrieved_chunk_ids": ["b", "c"],
                "metrics": {"document_hit_rate@5": 1.0},
                "elapsed_ms": 200.0,
            },
            {
                "case_id": "c3",
                "query": "q3",
                "retrieved_chunk_ids": [],
                "metrics": {"document_hit_rate@5": 0.0},
                "elapsed_ms": 300.0,
            },
        ]
        (run_dir / "baseline.cases.jsonl").write_text(
            "\n".join(json.dumps(case) for case in cases),
            encoding="utf-8",
        )

        runs = ep._discover_benchmark_runs(tmp_path)
        assert len(runs) == 1
        rows = ep._build_benchmark_experiment_rows(run_dir, runs[0]["run"])

        assert rows[0]["Experiment"] == "baseline"
        assert rows[0]["Configured Metrics"] == (
            "document_hit_rate@5, evidence_mrr@5"
        )
        assert rows[0]["document_hit_rate@5"] == 0.9
        assert rows[0]["evidence_mrr@5"] == 0.5
        assert rows[0]["Avg Query Latency (ms)"] == 200.0
        assert rows[0]["P50 Query Latency (ms)"] == 200.0
        assert rows[0]["P95 Query Latency (ms)"] == 290.0
        assert rows[0]["Max Query Latency (ms)"] == 300.0

        case_rows = ep._build_benchmark_case_rows(run_dir, "baseline")
        assert len(case_rows) == 3
        assert case_rows[1]["Retrieved Chunks"] == 2
        assert case_rows[2]["document_hit_rate@5"] == 0.0


class TestEvaluationPanelImport:
    """Verify the module can be imported without side effects."""

    def test_module_imports(self) -> None:
        from src.observability.dashboard.pages import evaluation_panel

        assert hasattr(evaluation_panel, "render")
        assert callable(evaluation_panel.render)

    def test_default_golden_path(self) -> None:
        from src.observability.dashboard.pages.evaluation_panel import (
            DEFAULT_GOLDEN_SET,
        )

        assert DEFAULT_GOLDEN_SET == Path("tests/fixtures/golden_test_set.json")
