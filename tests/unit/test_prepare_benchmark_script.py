"""Offline tests for the benchmark preparation command."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.core import settings as settings_module
from src.core.settings import BenchmarkSettings
from src.libs.benchmark.benchmark_factory import BenchmarkFactory


@dataclass(frozen=True)
class _EvaluationSettings:
    benchmark: BenchmarkSettings | None


@dataclass(frozen=True)
class _Settings:
    evaluation: _EvaluationSettings


class _FakeBenchmark:
    def __init__(
        self,
        data_dir: Path,
        *,
        answer: str = "TOP-SECRET-REFERENCE-ANSWER",
        prepare_error: Exception | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.pdf_dir = data_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True)
        self._cases = [
            SimpleNamespace(case_id="case-001", reference_answer=answer),
            SimpleNamespace(case_id="case-002", reference_answer="another answer"),
        ]
        self._prepare_error = prepare_error

    def prepare(self) -> list[SimpleNamespace]:
        if self._prepare_error is not None:
            raise self._prepare_error
        return self._cases


@pytest.fixture
def prepare_benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Avoid the script's Windows console wrapper interfering with pytest capture.
    with monkeypatch.context() as import_patch:
        import_patch.setattr(sys, "platform", "test")
        return importlib.import_module("scripts.prepare_benchmark")


def _settings(data_dir: Path, sample_size: int | None = None) -> _Settings:
    benchmark = BenchmarkSettings(
        provider="FakeBench",
        source_url="https://example.invalid/never-requested",
        data_dir=str(data_dir),
        sample_size=sample_size,
    )
    return _Settings(evaluation=_EvaluationSettings(benchmark=benchmark))


def _patch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    settings: _Settings,
    benchmark: _FakeBenchmark,
    captured: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(settings_module, "load_settings", lambda _path: settings)

    def create(effective_settings: _Settings) -> _FakeBenchmark:
        if captured is not None:
            captured["settings"] = effective_settings
        return benchmark

    monkeypatch.setattr(BenchmarkFactory, "create", staticmethod(create))


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
def test_success_summary_has_counts_without_answer_leakage(
    prepare_benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    as_json: bool,
) -> None:
    benchmark = _FakeBenchmark(tmp_path)
    nested = benchmark.pdf_dir / "nested"
    nested.mkdir()
    (benchmark.pdf_dir / "report.pdf").write_bytes(b"%PDF")
    (nested / "appendix.PDF").write_bytes(b"%PDF")
    (benchmark.pdf_dir / "notes.txt").write_text("not a PDF", encoding="utf-8")
    _patch_dependencies(monkeypatch, _settings(tmp_path), benchmark)
    argv = ["prepare_benchmark.py", "--config", "fake.yaml"]
    if as_json:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)

    exit_code = prepare_benchmark.main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "TOP-SECRET-REFERENCE-ANSWER" not in captured.out
    if as_json:
        assert json.loads(captured.out) == {
            "provider": "FakeBench",
            "data_dir": str(tmp_path),
            "pdf_count": 2,
            "case_count": 2,
            "first_case_id": "case-001",
        }
    else:
        assert captured.out.splitlines() == [
            "Provider: FakeBench",
            f"Data dir: {tmp_path}",
            "PDF count: 2",
            "Case count: 2",
            "First case ID: case-001",
        ]


def test_configuration_failure_returns_exit_2(
    prepare_benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_load(_path: str) -> None:
        raise ValueError("sensitive configuration detail")

    monkeypatch.setattr(settings_module, "load_settings", fail_load)
    monkeypatch.setattr(sys, "argv", ["prepare_benchmark.py", "--config", "bad.yaml"])

    exit_code = prepare_benchmark.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == ("Configuration error: could not load or validate bad.yaml\n")
    assert "sensitive configuration detail" not in captured.err


def test_prepare_failure_returns_exit_1(
    prepare_benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    benchmark = _FakeBenchmark(
        tmp_path,
        prepare_error=RuntimeError("sensitive preparation detail"),
    )
    _patch_dependencies(monkeypatch, _settings(tmp_path), benchmark)
    monkeypatch.setattr(sys, "argv", ["prepare_benchmark.py"])

    exit_code = prepare_benchmark.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "Benchmark preparation failed.\n"
    assert "sensitive preparation detail" not in captured.err


def test_sample_size_override_is_passed_only_to_factory(
    prepare_benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    original_settings = _settings(tmp_path, sample_size=99)
    benchmark = _FakeBenchmark(tmp_path)
    captured_factory: dict[str, Any] = {}
    _patch_dependencies(
        monkeypatch,
        original_settings,
        benchmark,
        captured_factory,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_benchmark.py", "--sample-size", "3"],
    )

    exit_code = prepare_benchmark.main()

    capsys.readouterr()
    effective_settings = captured_factory["settings"]
    assert exit_code == 0
    assert effective_settings.evaluation.benchmark.sample_size == 3
    assert original_settings.evaluation.benchmark.sample_size == 99
    assert effective_settings is not original_settings
