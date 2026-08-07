"""Regression tests for benchmark evaluation checkpoint identity."""

from types import SimpleNamespace

from scripts import evaluate


def _settings(output_directory: str) -> SimpleNamespace:
    evaluation = SimpleNamespace(
        metrics=["macro_evidence_hit_rate@5"],
        output=SimpleNamespace(directory=output_directory),
    )
    return SimpleNamespace(
        retrieval=SimpleNamespace(dense_top_k=20, sparse_top_k=20),
        rerank=SimpleNamespace(
            provider="cross_encoder_api",
            api_args={"model": "test-reranker"},
            model="",
            top_k=5,
        ),
        evaluation=evaluation,
    )


def test_checkpoint_changes_when_evidence_judge_prompt_changes(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    prompt_path = project_root / "config" / "prompts" / "evidence_judge.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("prompt version one", encoding="utf-8")
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", project_root)

    settings = _settings(str(tmp_path / "output"))
    plan = SimpleNamespace(
        name="baseline",
        index_fingerprint="index-123",
        settings=settings,
    )
    cases = [SimpleNamespace(case_id="case-1")]

    first = evaluate._benchmark_checkpoint_path(plan, cases, settings, None)
    prompt_path.write_text("prompt version two", encoding="utf-8")
    second = evaluate._benchmark_checkpoint_path(plan, cases, settings, None)

    assert first != second


def test_benchmark_rejects_silent_reranker_fallback() -> None:
    settings = SimpleNamespace(rerank=SimpleNamespace(enabled=True))
    reranker = SimpleNamespace(is_enabled=False)

    try:
        evaluate._require_benchmark_reranker(settings, reranker)
    except RuntimeError as exc:
        assert "reranker is enabled but unavailable" in str(exc)
    else:
        raise AssertionError("Expected benchmark reranker validation to fail")


def test_benchmark_allows_explicitly_disabled_reranker() -> None:
    settings = SimpleNamespace(rerank=SimpleNamespace(enabled=False))
    reranker = SimpleNamespace(is_enabled=False)

    assert evaluate._require_benchmark_reranker(settings, reranker) is reranker
