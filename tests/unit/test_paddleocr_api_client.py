"""Tests for the PaddleOCR Studio job API client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.libs.loader.paddleocr_api_client import PaddleOcrApiClient


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
        text: str | None = None,
        content: bytes = b"",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload or {})
        self.content = content

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("response is not JSON")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")


class FakeSession:
    def __init__(self, *, posts: list[FakeResponse], gets: list[FakeResponse]) -> None:
        self.posts = list(posts)
        self.gets = list(gets)
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.get_calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append((url, kwargs))
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append((url, kwargs))
        return self.gets.pop(0)


def _jsonl_result() -> str:
    return "\n".join(
        [
            json.dumps(
                {
                    "result": {
                        "layoutParsingResults": [
                            {
                                "pageIndex": 0,
                                "markdown": {
                                    "text": "# First page\n\nRevenue was $10 million.",
                                    "images": {},
                                },
                                "parsing_res_list": [
                                    {"block_label": "text", "block_content": "Revenue"}
                                ],
                            }
                        ]
                    }
                }
            ),
            json.dumps(
                {
                    "result": {
                        "layoutParsingResults": [
                            {
                                "pageIndex": 1,
                                "markdown": {
                                    "text": "## Second page\n\nOperating income increased.",
                                    "images": {},
                                },
                            }
                        ]
                    }
                }
            ),
        ]
    )


def _session_for_success() -> FakeSession:
    return FakeSession(
        posts=[
            FakeResponse({"data": {"jobId": "job-123"}}),
        ],
        gets=[
            FakeResponse({"data": {"state": "pending"}}),
            FakeResponse(
                {
                    "data": {
                        "state": "done",
                        "extractProgress": {"extractedPages": 2},
                        "resultUrl": {"jsonUrl": "https://download.test/result.jsonl"},
                    }
                }
            ),
            FakeResponse(text=_jsonl_result()),
        ],
    )


def test_sync_client_uploads_pdf_polls_and_returns_loader_artifact(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    session = _session_for_success()
    client = PaddleOcrApiClient(
        {
            "token": "secret-token",
            "job_url": "https://example.test/api/v2/ocr/jobs",
            "model": "PaddleOCR-VL-1.6",
            "poll_interval_seconds": 0,
            "timeout_seconds": 30,
            "optional_payload": {
                "useDocOrientationClassify": False,
                "useDocUnwarping": False,
                "useChartRecognition": False,
                "restructurePages": True,
                "mergeTables": False,
                "relevelTitles": True,
                "concatenatePages": False,
                "returnMarkdownImages": False,
            },
        },
        session=session,
    )

    artifact = client.run(pdf)

    assert [page["page_index"] for page in artifact["restructured_pages"]] == [0, 1]
    assert artifact["restructured_pages"][0]["markdown_text"].startswith("# First page")
    assert artifact["pages"][0]["res"]["page_index"] == 0
    post_url, post_kwargs = session.post_calls[0]
    assert post_url == "https://example.test/api/v2/ocr/jobs"
    assert post_kwargs["headers"]["Authorization"] == "bearer secret-token"
    assert post_kwargs["data"]["model"] == "PaddleOCR-VL-1.6"
    assert json.loads(post_kwargs["data"]["optionalPayload"])[
        "useDocOrientationClassify"
    ] is False
    submitted_options = json.loads(post_kwargs["data"]["optionalPayload"])
    assert submitted_options["restructurePages"] is True
    assert submitted_options["mergeTables"] is False
    assert submitted_options["relevelTitles"] is True
    assert submitted_options["concatenatePages"] is False
    assert submitted_options["returnMarkdownImages"] is False
    assert "file" in post_kwargs["files"]


@pytest.mark.asyncio
async def test_async_client_uses_same_contract_without_blocking_poll_sleep(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    client = PaddleOcrApiClient(
        {
            "token": "secret-token",
            "poll_interval_seconds": 0,
            "timeout_seconds": 30,
        },
        session=_session_for_success(),
    )

    artifact = await client.run_async(pdf)

    assert artifact["restructured_pages"][1]["page_index"] == 1
    assert "Operating income" in artifact["restructured_pages"][1]["markdown_text"]


def test_client_reads_token_from_environment_and_never_exposes_it_in_cache_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLEOCR_API_TOKEN", "environment-secret")

    client = PaddleOcrApiClient({"token_env": "PADDLEOCR_API_TOKEN"})

    assert client.authorization_header == "bearer environment-secret"
    assert "environment-secret" not in json.dumps(client.cache_config())
    assert "token" not in client.cache_config()


def test_failed_job_raises_with_remote_error(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    session = FakeSession(
        posts=[FakeResponse({"data": {"jobId": "job-failed"}})],
        gets=[
            FakeResponse(
                {"data": {"state": "failed", "errorMsg": "unsupported document"}}
            )
        ],
    )
    client = PaddleOcrApiClient(
        {"token": "secret-token", "poll_interval_seconds": 0},
        session=session,
    )

    with pytest.raises(RuntimeError, match="unsupported document"):
        client.run(pdf)


def test_client_rejects_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PADDLEOCR_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="PADDLEOCR_API_TOKEN"):
        PaddleOcrApiClient({})
