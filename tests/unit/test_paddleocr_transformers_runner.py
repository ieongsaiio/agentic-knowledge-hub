"""Unit tests for the PaddleOCR Transformers runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts import paddleocr_transformers_runner as runner


class _FakeResult:
    def __init__(self, page_index: int, text: str) -> None:
        self.json = {"res": {"page_index": page_index}, "text": text}
        self.markdown = {"markdown_texts": f"# {text}", "markdown_images": {}}


def _install_fake_paddleocr(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[Any], dict[str, Any]]:
    calls: dict[str, Any] = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs: Any) -> None:
            calls["init"] = kwargs

        def predict_iter(self, **kwargs: Any) -> list[_FakeResult]:
            calls["predict_iter"] = kwargs
            return [_FakeResult(3, "source page")]

        def restructure_pages(self, pages: list[_FakeResult], **kwargs: Any) -> list[_FakeResult]:
            calls["restructure_pages"] = {"pages": pages, **kwargs}
            return [_FakeResult(7, "restructured page")]

    fake_paddleocr = ModuleType("paddleocr")
    fake_paddleocr.PaddleOCRVL = FakePaddleOCRVL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
    return FakePaddleOCRVL, calls


@pytest.mark.parametrize("vision_enabled", [False, True])
def test_main_passes_config_and_writes_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vision_enabled: bool,
) -> None:
    _, calls = _install_fake_paddleocr(monkeypatch)
    input_path = tmp_path / "input.pdf"
    output_path = tmp_path / "result.json"
    input_path.write_bytes(b"fake pdf")
    config = {
        "engine": "transformers",
        "pipeline_version": "v9.9",
        "use_queues": False,
        "merge_tables": False,
        "relevel_titles": False,
        "concatenate_pages": True,
        "extra_setting": "preserved",
    }
    argv = [
        "paddleocr_transformers_runner.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--config-json",
        json.dumps(config),
    ]
    if vision_enabled:
        argv.append("--vision-enabled")
    monkeypatch.setattr(sys, "argv", argv)

    assert runner.main() == 0

    init = calls["init"]
    assert init["engine"] == "transformers"
    assert init["pipeline_version"] == "v9.9"
    assert init["use_queues"] is False
    assert ("image" in init["markdown_ignore_labels"]) is not vision_enabled
    assert calls["predict_iter"] == {"input": str(input_path)}

    restructure = calls["restructure_pages"]
    assert restructure["merge_tables"] is False
    assert restructure["relevel_titles"] is False
    assert restructure["concatenate_pages"] is True
    assert len(restructure["pages"]) == 1

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["config"] == config
    assert payload["pages"] == [{"res": {"page_index": 3}, "text": "source page"}]
    assert payload["restructured_pages"] == [
        {
            "page_index": 7,
            "markdown_text": "# restructured page",
            "json": {
                "res": {"page_index": 7},
                "text": "restructured page",
            },
            "images": {},
        }
    ]
    assert isinstance(payload["elapsed_seconds"], float)
    assert payload["elapsed_seconds"] >= 0


def test_main_returns_two_for_invalid_config_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, calls = _install_fake_paddleocr(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paddleocr_transformers_runner.py",
            "--input",
            str(tmp_path / "unused.pdf"),
            "--output",
            str(tmp_path / "unused.json"),
            "--config-json",
            "{bad json",
        ],
    )

    assert runner.main() == 2
    assert "--config-json is not valid JSON" in capsys.readouterr().err
    assert calls == {}
