"""Tests for the generated-data cleanup script."""

from pathlib import Path

import pytest

from scripts.clear_data import build_targets, clear_targets


def _config() -> dict:
    return {
        "vector_store": {"persist_directory": "./data/db/chroma"},
        "observability": {"trace_file": "./logs/traces.jsonl"},
        "evaluation": {"output": {"directory": "./data/evaluation"}},
    }


def test_storage_cleanup_preserves_benchmark_data(tmp_path: Path) -> None:
    chroma = tmp_path / "data" / "db" / "chroma"
    bm25 = tmp_path / "data" / "db" / "bm25"
    images = tmp_path / "data" / "images"
    benchmark = tmp_path / "data" / "benchmarks" / "financebench"
    for directory in (chroma, bm25, images, benchmark):
        directory.mkdir(parents=True)
        (directory / "content.bin").write_bytes(b"data")
    (tmp_path / "data" / "db" / "ingestion_history.db").write_bytes(b"db")
    (tmp_path / "data" / "db" / "image_index.db").write_bytes(b"db")

    targets = build_targets(
        tmp_path,
        _config(),
        storage=True,
        logs=False,
        evaluation=False,
    )
    removed, _ = clear_targets(targets, dry_run=False)

    assert removed == 5
    assert not chroma.exists()
    assert not bm25.exists()
    assert not images.exists()
    assert benchmark.exists()
    assert (benchmark / "content.bin").exists()


def test_logs_scope_only_removes_jsonl_files(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    trace = logs / "traces.jsonl"
    history = logs / "eval_history.jsonl"
    text_log = logs / "application.log"
    trace.write_text("{}\n", encoding="utf-8")
    history.write_text("{}\n", encoding="utf-8")
    text_log.write_text("keep", encoding="utf-8")

    targets = build_targets(
        tmp_path,
        _config(),
        storage=False,
        logs=True,
        evaluation=False,
    )
    clear_targets(targets, dry_run=False)

    assert not trace.exists()
    assert not history.exists()
    assert text_log.exists()


def test_dry_run_does_not_remove_data(tmp_path: Path) -> None:
    chroma = tmp_path / "data" / "db" / "chroma"
    chroma.mkdir(parents=True)

    targets = build_targets(
        tmp_path,
        _config(),
        storage=True,
        logs=False,
        evaluation=False,
    )
    removed, _ = clear_targets(targets, dry_run=True)

    assert removed == 0
    assert chroma.exists()


def test_rejects_configured_path_outside_project(tmp_path: Path) -> None:
    config = _config()
    config["vector_store"]["persist_directory"] = str(tmp_path.parent)

    with pytest.raises(ValueError, match="outside project root"):
        build_targets(
            tmp_path,
            config,
            storage=True,
            logs=False,
            evaluation=False,
        )


def test_rejects_target_that_contains_benchmarks(tmp_path: Path) -> None:
    config = _config()
    config["vector_store"]["persist_directory"] = "./data"

    with pytest.raises(ValueError, match="benchmark source data"):
        build_targets(
            tmp_path,
            config,
            storage=True,
            logs=False,
            evaluation=False,
        )


def test_rejects_target_inside_benchmarks(tmp_path: Path) -> None:
    config = _config()
    config["vector_store"]["persist_directory"] = (
        "./data/benchmarks/financebench"
    )

    with pytest.raises(ValueError, match="benchmark source data"):
        build_targets(
            tmp_path,
            config,
            storage=True,
            logs=False,
            evaluation=False,
        )
