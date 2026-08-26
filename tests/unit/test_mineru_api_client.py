"""Unit tests for the MinerU precise API v4 client."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.libs.loader.mineru_api_client import MinerUApiClient


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
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    def __init__(
        self,
        *,
        posts: list[FakeResponse],
        puts: list[FakeResponse],
        gets: list[FakeResponse],
    ) -> None:
        self.posts = list(posts)
        self.puts = list(puts)
        self.gets = list(gets)
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.put_calls: list[tuple[str, dict[str, Any]]] = []
        self.get_calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append((url, kwargs))
        return self.posts.pop(0)

    def put(self, url: str, **kwargs: Any) -> FakeResponse:
        data = kwargs.get("data")
        if hasattr(data, "read"):
            kwargs["data"] = data.read()
        self.put_calls.append((url, kwargs))
        return self.puts.pop(0)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append((url, kwargs))
        return self.gets.pop(0)


def _result_zip() -> bytes:
    buffer = io.BytesIO()
    middle = {
        "_backend": "vlm",
        "_version_name": "3.0.1",
        "pdf_info": [{"page_idx": 0, "page_size": [612, 792]}],
    }
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sample/full.md", "# Revenue\n\nRevenue was 10.")
        archive.writestr(
            "sample/sample_content_list.json",
            json.dumps([{"type": "text", "text": "legacy", "page_idx": 0}]),
        )
        archive.writestr(
            "sample/sample_content_list_v2.json",
            json.dumps([[{"type": "paragraph", "content": {"paragraph_content": []}}]]),
        )
        archive.writestr("sample/layout.json", json.dumps(middle))
        archive.writestr("sample/sample_model.json", json.dumps([[{"type": "text"}]]))
        archive.writestr("sample/images/table.jpg", b"image")
    return buffer.getvalue()


def _success_session() -> FakeSession:
    return FakeSession(
        posts=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-123",
                        "file_urls": ["https://upload.test/signed"],
                    },
                    "msg": "ok",
                }
            )
        ],
        puts=[FakeResponse()],
        gets=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-123",
                        "extract_result": [
                            {"file_name": "sample.pdf", "state": "waiting-file"}
                        ],
                    },
                }
            ),
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-123",
                        "extract_result": [
                            {
                                "file_name": "sample.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.test/result.zip",
                            }
                        ],
                    },
                }
            ),
            FakeResponse(content=_result_zip()),
        ],
    )


def _batch_result_zip(markdown: str, page_count: int) -> bytes:
    buffer = io.BytesIO()
    legacy = [
        {"type": "text", "text": f"page-{index}", "page_idx": index}
        for index in range(page_count)
    ]
    v2 = [
        [
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": f"page-{index}"}
                    ]
                },
            }
        ]
        for index in range(page_count)
    ]
    middle = {
        "_version_name": "3.0.1",
        "pdf_info": [
            {"page_idx": index, "page_size": [612, 792]}
            for index in range(page_count)
        ],
    }
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("full.md", markdown)
        archive.writestr("batch_content_list.json", json.dumps(legacy))
        archive.writestr("batch_content_list_v2.json", json.dumps(v2))
        archive.writestr("layout.json", json.dumps(middle))
    return buffer.getvalue()


def test_signed_upload_poll_and_zip_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_TEST_TOKEN", "environment-secret")
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    session = _success_session()
    sleeps: list[float] = []
    client = MinerUApiClient(
        {
            "base_url": "https://mineru.test/api/v4",
            "token_env": "MINERU_TEST_TOKEN",
            "model_version": "vlm",
            "language": "en",
            "enable_table": True,
            "enable_formula": False,
            "is_ocr": False,
            "page_ranges": "1-2",
            "poll_interval_seconds": 0.25,
        },
        session=session,
        sleeper=sleeps.append,
    )

    artifact = client.run(pdf)

    assert artifact["provider"] == "mineru"
    assert artifact["version"] == "3.0.1"
    assert artifact["full_markdown"].startswith("# Revenue")
    assert artifact["content_list"][0]["text"] == "legacy"
    assert artifact["content_list_v2"][0][0]["type"] == "paragraph"
    assert artifact["middle_json"]["_backend"] == "vlm"
    assert artifact["model_json"][0][0]["type"] == "text"
    assert "sample/images/table.jpg" in artifact["source_files"]["archive_members"]
    assert artifact["source_files"]["input_file"] == "sample.pdf"
    assert sleeps == [0.25]

    post_url, post_kwargs = session.post_calls[0]
    assert post_url == "https://mineru.test/api/v4/file-urls/batch"
    assert post_kwargs["headers"]["Authorization"] == "Bearer environment-secret"
    assert post_kwargs["json"] == {
        "files": [
            {
                "name": "sample.pdf",
                "is_ocr": False,
                "page_ranges": "1-2",
            }
        ],
        "model_version": "vlm",
        "language": "en",
        "enable_table": True,
        "enable_formula": False,
    }
    put_url, put_kwargs = session.put_calls[0]
    assert put_url == "https://upload.test/signed"
    assert "headers" not in put_kwargs
    assert put_kwargs["data"] == b"%PDF-1.4 test"
    assert "environment-secret" not in json.dumps(artifact)


def test_token_prefers_direct_config_and_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="MINERU_API_TOKEN"):
        MinerUApiClient({})

    direct_client = MinerUApiClient(
        {"api_key": "direct-secret", "token_env": "MISSING_TOKEN"}
    )
    assert direct_client.authorization_header == "Bearer direct-secret"
    assert "direct-secret" not in json.dumps(direct_client.cache_config())

    monkeypatch.setenv("CUSTOM_MINERU_TOKEN", "secret")
    client = MinerUApiClient({"token_env": "CUSTOM_MINERU_TOKEN"})
    assert client.authorization_header == "Bearer secret"
    assert "secret" not in json.dumps(client.cache_config())


def test_large_pdf_is_split_and_page_indexes_are_merged(
    tmp_path: Path,
) -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "large.pdf"
    document = fitz.open()
    for index in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1}")
    document.save(pdf_path)
    document.close()

    session = FakeSession(
        posts=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-a",
                        "file_urls": ["https://upload.test/a"],
                    },
                }
            ),
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "batch_id": "batch-b",
                        "file_urls": ["https://upload.test/b"],
                    },
                }
            ),
        ],
        puts=[FakeResponse(), FakeResponse()],
        gets=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "large_pages_0001_0002.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.test/a.zip",
                            }
                        ]
                    },
                }
            ),
            FakeResponse(content=_batch_result_zip("# Batch A", 2)),
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "large_pages_0003_0003.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.test/b.zip",
                            }
                        ]
                    },
                }
            ),
            FakeResponse(content=_batch_result_zip("# Batch B", 1)),
        ],
    )
    client = MinerUApiClient(
        {
            "api_key": "secret",
            "max_pages_per_request": 2,
            "poll_interval_seconds": 0,
        },
        session=session,
        sleeper=lambda _: None,
    )

    artifact = client.run(pdf_path)

    assert artifact["full_markdown"] == "# Batch A\n\n# Batch B"
    assert len(artifact["content_list_v2"]) == 3
    assert [item["page_idx"] for item in artifact["content_list"]] == [0, 1, 2]
    assert [
        item["page_idx"] for item in artifact["middle_json"]["pdf_info"]
    ] == [0, 1, 2]
    assert artifact["batch_ids"] == ["batch-a", "batch-b"]
    assert [batch["page_offset"] for batch in artifact["source_files"]["batches"]] == [
        0,
        2,
    ]
    assert len(session.post_calls) == 2
    assert len(session.put_calls) == 2


def test_remote_failed_state_has_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_TOKEN", "secret")
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"bad")
    session = FakeSession(
        posts=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {"batch_id": "batch-bad", "file_urls": ["signed"]},
                }
            )
        ],
        puts=[FakeResponse()],
        gets=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "file_name": "bad.pdf",
                                "state": "failed",
                                "err_msg": "document is damaged",
                            }
                        ]
                    },
                }
            )
        ],
    )
    client = MinerUApiClient(
        {"poll_interval_seconds": 0},
        session=session,
        sleeper=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="document is damaged"):
        client.run(pdf)


def test_api_code_and_invalid_zip_errors_are_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_TOKEN", "secret")
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"pdf")
    api_error = FakeSession(
        posts=[FakeResponse({"code": -60004, "msg": "empty file"})],
        puts=[],
        gets=[],
    )
    with pytest.raises(RuntimeError, match="-60004.*empty file"):
        MinerUApiClient({}, session=api_error).run(pdf)

    invalid_zip = _success_session()
    invalid_zip.gets[-1] = FakeResponse(content=b"not a zip")
    with pytest.raises(RuntimeError, match="valid ZIP"):
        MinerUApiClient(
            {"poll_interval_seconds": 0},
            session=invalid_zip,
            sleeper=lambda _: None,
        ).run(pdf)


def test_timeout_uses_injected_clock_and_sleeper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINERU_API_TOKEN", "secret")
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"pdf")
    session = FakeSession(
        posts=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {"batch_id": "batch-slow", "file_urls": ["signed"]},
                }
            )
        ],
        puts=[FakeResponse()],
        gets=[
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {"file_name": "sample.pdf", "state": "running"}
                        ]
                    },
                }
            )
        ],
    )
    ticks = iter([0.0, 2.0])
    client = MinerUApiClient(
        {"timeout_seconds": 1, "poll_interval_seconds": 0},
        session=session,
        sleeper=lambda _: None,
        clock=lambda: next(ticks),
    )

    with pytest.raises(TimeoutError, match="batch-slow.*1"):
        client.run(pdf)
