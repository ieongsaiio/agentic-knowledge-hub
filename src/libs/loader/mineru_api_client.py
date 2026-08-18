"""Synchronous client for MinerU's precise API v4 signed-upload flow."""

from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import requests

_DEFAULT_BASE_URL = "https://mineru.net/api/v4"
_PENDING_STATES = {"waiting-file", "pending", "running", "converting"}


class MinerUApiClient:
    """Upload one local document, wait for extraction, and read its result ZIP."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        session: Any = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config or {}))
        self.session = session or requests.Session()
        self._sleep = sleeper or time.sleep
        self._clock = clock or time.monotonic
        self.base_url = str(self.config.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self.token_env = str(
            self.config.get("token_env", "MINERU_API_TOKEN")
        ).strip()
        configured_token = str(self.config.get("api_key", "")).strip()
        token = configured_token or os.getenv(self.token_env, "").strip()
        if not token:
            raise ValueError(
                "MinerU API token is missing; set mineru.api.api_key or "
                f"environment variable {self.token_env}"
            )
        self._token = token
        self.model_version = str(self.config.get("model_version", "vlm")).strip()
        if self.model_version not in {"pipeline", "vlm", "MinerU-HTML"}:
            raise ValueError(
                "MinerU api.model_version must be pipeline, vlm, or MinerU-HTML"
            )
        self.poll_interval = self._number(
            "poll_interval_seconds", default=5, allow_zero=True
        )
        self.timeout = self._number(
            "timeout_seconds", default=1800, allow_zero=False
        )
        self.request_timeout = self._number(
            "request_timeout_seconds", default=120, allow_zero=False
        )
        max_pages = self.config.get("max_pages_per_request", 200)
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
            raise ValueError("MinerU api.max_pages_per_request must be a positive integer")
        self.max_pages_per_request = max_pages

    @property
    def authorization_header(self) -> str:
        return f"Bearer {self._token}"

    def cache_config(self) -> dict[str, Any]:
        """Return output-affecting settings, excluding credentials and timings."""
        keys = (
            "model_version",
            "language",
            "enable_table",
            "enable_formula",
            "is_ocr",
            "page_ranges",
            "extra_formats",
            "max_pages_per_request",
        )
        return {
            key: copy.deepcopy(self.config[key])
            for key in keys
            if key in self.config
        } | {"model_version": self.model_version}

    def run(self, file_path: str | Path) -> dict[str, Any]:
        """Run the complete signed-upload workflow for one local file."""
        path = self._validate_path(file_path)
        if path.suffix.lower() == ".pdf" and "page_ranges" not in self.config:
            page_count = self._pdf_page_count(path)
            if page_count is not None and page_count > self.max_pages_per_request:
                return self._run_pdf_batches(path, page_count)
        return self._run_single(path)

    def _run_single(self, path: Path) -> dict[str, Any]:
        """Run one API request for a document within the remote page limit."""
        started_at = self._clock()
        batch_id, upload_url = self._request_upload(path)
        self._upload(path, upload_url)
        result = self._poll(batch_id, path.name, started_at)
        archive_bytes = self._download_archive(result, batch_id)
        artifact = self._read_archive(archive_bytes, path.name)
        artifact["batch_id"] = batch_id
        artifact["elapsed_seconds"] = max(0.0, self._clock() - started_at)
        artifact["config"] = self.cache_config()
        return artifact

    def _run_pdf_batches(self, path: Path, page_count: int) -> dict[str, Any]:
        """Parse a large PDF in bounded requests and merge page-indexed artifacts."""
        started_at = self._clock()
        artifacts: list[tuple[int, dict[str, Any]]] = []
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - project PDF stack includes it
            raise RuntimeError(
                "PyMuPDF is required to split PDFs exceeding MinerU's page limit"
            ) from exc

        with tempfile.TemporaryDirectory(prefix="mineru_pdf_batches_") as temp_dir:
            with fitz.open(path) as source:
                for start in range(0, page_count, self.max_pages_per_request):
                    end = min(start + self.max_pages_per_request, page_count)
                    batch_path = Path(temp_dir) / (
                        f"{path.stem}_pages_{start + 1:04d}_{end:04d}.pdf"
                    )
                    batch = fitz.open()
                    try:
                        batch.insert_pdf(source, from_page=start, to_page=end - 1)
                        batch.save(batch_path)
                    finally:
                        batch.close()
                    artifacts.append((start, self._run_single(batch_path)))

        merged = self._merge_batch_artifacts(path.name, artifacts)
        merged["elapsed_seconds"] = max(0.0, self._clock() - started_at)
        merged["config"] = self.cache_config()
        return merged

    @staticmethod
    def _pdf_page_count(path: Path) -> int | None:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - project PDF stack includes it
            raise RuntimeError("PyMuPDF is required to inspect PDF page counts") from exc
        try:
            with fitz.open(path) as document:
                return len(document)
        except (fitz.FileDataError, RuntimeError, ValueError):
            return None

    @classmethod
    def _merge_batch_artifacts(
        cls,
        input_file: str,
        artifacts: list[tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        if not artifacts:
            raise RuntimeError("MinerU PDF batching produced no artifacts")

        first = artifacts[0][1]
        merged_v2: list[Any] = []
        merged_legacy: list[Any] = []
        merged_pdf_info: list[Any] = []
        merged_model: list[Any] = []
        markdown_parts: list[str] = []
        batches: list[dict[str, Any]] = []

        for page_offset, artifact in artifacts:
            markdown = artifact.get("full_markdown")
            if isinstance(markdown, str) and markdown.strip():
                markdown_parts.append(markdown.strip())
            merged_v2.extend(cls._shift_v2_pages(artifact.get("content_list_v2"), page_offset))
            merged_legacy.extend(
                cls._shift_flat_blocks(artifact.get("content_list"), page_offset)
            )
            middle = artifact.get("middle_json")
            if isinstance(middle, Mapping):
                merged_pdf_info.extend(
                    cls._shift_flat_blocks(middle.get("pdf_info"), page_offset)
                )
            model = artifact.get("model_json")
            if isinstance(model, list):
                merged_model.extend(copy.deepcopy(model))
            batches.append(
                {
                    "page_offset": page_offset,
                    "batch_id": artifact.get("batch_id"),
                    "source_files": copy.deepcopy(artifact.get("source_files")),
                }
            )

        middle_json = copy.deepcopy(first.get("middle_json"))
        if not isinstance(middle_json, dict):
            middle_json = {}
        middle_json["pdf_info"] = merged_pdf_info
        return {
            "provider": "mineru",
            "version": first.get("version"),
            "full_markdown": "\n\n".join(markdown_parts),
            "content_list_v2": merged_v2 or None,
            "content_list": merged_legacy or None,
            "middle_json": middle_json,
            "model_json": merged_model or None,
            "batch_ids": [artifact.get("batch_id") for _, artifact in artifacts],
            "source_files": {
                "input_file": input_file,
                "batches": batches,
            },
        }

    @classmethod
    def _shift_v2_pages(cls, value: Any, page_offset: int) -> list[Any]:
        if not isinstance(value, list):
            return []
        if all(isinstance(page, list) for page in value):
            return copy.deepcopy(value)
        return cls._shift_flat_blocks(value, page_offset)

    @staticmethod
    def _shift_flat_blocks(value: Any, page_offset: int) -> list[Any]:
        if not isinstance(value, list):
            return []
        shifted = copy.deepcopy(value)
        for item in shifted:
            if not isinstance(item, dict):
                continue
            for key in ("page_idx", "page_index"):
                page_index = item.get(key)
                if isinstance(page_index, int) and not isinstance(page_index, bool):
                    item[key] = page_index + page_offset
        return shifted

    def _request_upload(self, path: Path) -> tuple[str, str]:
        file_options: dict[str, Any] = {"name": path.name}
        for key in ("is_ocr", "data_id", "page_ranges"):
            if key in self.config:
                file_options[key] = copy.deepcopy(self.config[key])

        payload: dict[str, Any] = {
            "files": [file_options],
            "model_version": self.model_version,
        }
        for key in (
            "language",
            "enable_table",
            "enable_formula",
            "callback",
            "seed",
            "extra_formats",
        ):
            if key in self.config:
                payload[key] = copy.deepcopy(self.config[key])

        response = self._request(
            "request signed upload URL",
            self.session.post,
            f"{self.base_url}/file-urls/batch",
            headers=self._api_headers(),
            json=payload,
            timeout=self.request_timeout,
        )
        data = self._api_payload(response, "request signed upload URL").get("data")
        if not isinstance(data, Mapping):
            raise RuntimeError("MinerU upload URL response is missing data")
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise RuntimeError("MinerU upload URL response is missing data.batch_id")
        if (
            not isinstance(file_urls, list)
            or len(file_urls) != 1
            or not isinstance(file_urls[0], str)
            or not file_urls[0]
        ):
            raise RuntimeError(
                "MinerU upload URL response must contain exactly one data.file_urls entry"
            )
        return batch_id, file_urls[0]

    def _upload(self, path: Path, upload_url: str) -> None:
        # MinerU explicitly requires no Content-Type header for signed PUT uploads.
        with path.open("rb") as source:
            response = self._request(
                "upload file to MinerU signed URL",
                self.session.put,
                upload_url,
                data=source,
                timeout=self.request_timeout,
            )
        self._raise_for_status(response, "upload file to MinerU signed URL")

    def _poll(
        self,
        batch_id: str,
        file_name: str,
        started_at: float,
    ) -> dict[str, Any]:
        while True:
            response = self._request(
                f"poll MinerU batch {batch_id}",
                self.session.get,
                f"{self.base_url}/extract-results/batch/{batch_id}",
                headers=self._api_headers(),
                timeout=self.request_timeout,
            )
            data = self._api_payload(
                response, f"poll MinerU batch {batch_id}"
            ).get("data")
            result = self._select_result(data, file_name)
            state = str(result.get("state", "")).strip().lower()
            if state == "done":
                return result
            if state == "failed":
                detail = result.get("err_msg") or "unknown remote error"
                raise RuntimeError(
                    f"MinerU batch {batch_id} failed for {file_name}: {detail}"
                )
            if state not in _PENDING_STATES:
                raise RuntimeError(
                    f"MinerU batch {batch_id} returned unknown state {state!r} "
                    f"for {file_name}"
                )
            if self._clock() - started_at >= self.timeout:
                raise TimeoutError(
                    f"MinerU batch {batch_id} timed out after {self.timeout:g} seconds"
                )
            self._sleep(self.poll_interval)

    def _download_archive(
        self,
        result: Mapping[str, Any],
        batch_id: str,
    ) -> bytes:
        url = result.get("full_zip_url")
        if not isinstance(url, str) or not url:
            raise RuntimeError(
                f"MinerU batch {batch_id} completed without full_zip_url"
            )
        response = self._request(
            f"download MinerU batch {batch_id} ZIP",
            self.session.get,
            url,
            timeout=self.request_timeout,
        )
        self._raise_for_status(response, f"download MinerU batch {batch_id} ZIP")
        content = getattr(response, "content", None)
        if not isinstance(content, bytes):
            raise RuntimeError(f"MinerU batch {batch_id} ZIP response has no bytes")
        return content

    def _read_archive(self, archive_bytes: bytes, input_file: str) -> dict[str, Any]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError("MinerU result is not a valid ZIP archive") from exc

        with archive:
            members = sorted(
                name for name in archive.namelist() if not name.endswith("/")
            )
            selected = {
                "full_markdown": self._select_member(
                    members, lambda name: PurePosixPath(name).name == "full.md"
                ),
                "content_list_v2": self._select_member(
                    members,
                    lambda name: PurePosixPath(name).name.endswith(
                        "_content_list_v2.json"
                    ),
                ),
                "content_list": self._select_member(
                    members,
                    lambda name: PurePosixPath(name).name.endswith(
                        "_content_list.json"
                    )
                    and not PurePosixPath(name).name.endswith(
                        "_content_list_v2.json"
                    ),
                ),
                "middle_json": self._select_member(
                    members,
                    lambda name: PurePosixPath(name).name.endswith("_middle.json")
                    or PurePosixPath(name).name == "layout.json",
                ),
                "model_json": self._select_member(
                    members,
                    lambda name: PurePosixPath(name).name.endswith("_model.json")
                    or PurePosixPath(name).name == "model.json",
                ),
            }
            if selected["full_markdown"] is None:
                raise RuntimeError("MinerU result ZIP is missing full.md")
            if (
                selected["content_list_v2"] is None
                and selected["content_list"] is None
            ):
                raise RuntimeError(
                    "MinerU result ZIP is missing content_list_v2.json and content_list.json"
                )

            full_markdown = self._read_text(
                archive, selected["full_markdown"], "full.md"
            )
            artifact: dict[str, Any] = {
                "provider": "mineru",
                "version": self.model_version,
                "full_markdown": full_markdown,
                "content_list": self._read_json_optional(
                    archive, selected["content_list"], "content_list.json"
                ),
                "content_list_v2": self._read_json_optional(
                    archive, selected["content_list_v2"], "content_list_v2.json"
                ),
                "middle_json": self._read_json_optional(
                    archive, selected["middle_json"], "middle.json"
                ),
                "model_json": self._read_json_optional(
                    archive, selected["model_json"], "model.json"
                ),
                "source_files": {
                    "input_file": input_file,
                    "archive_members": members,
                    "selected": selected,
                },
            }
            middle = artifact["middle_json"]
            if isinstance(middle, Mapping):
                version = middle.get("_version_name")
                if isinstance(version, str) and version.strip():
                    artifact["version"] = version
            return artifact

    @staticmethod
    def _select_member(
        members: list[str],
        predicate: Callable[[str], bool],
    ) -> str | None:
        matches = [member for member in members if predicate(member)]
        if not matches:
            return None
        return sorted(matches, key=lambda value: (value.count("/"), value))[0]

    @classmethod
    def _read_json_optional(
        cls,
        archive: zipfile.ZipFile,
        member: str | None,
        label: str,
    ) -> Any:
        if member is None:
            return None
        text = cls._read_text(archive, member, label)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MinerU result ZIP contains invalid {label}: {exc}"
            ) from exc

    @staticmethod
    def _read_text(
        archive: zipfile.ZipFile,
        member: str,
        label: str,
    ) -> str:
        try:
            return archive.read(member).decode("utf-8-sig")
        except (KeyError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Unable to read MinerU {label} from result ZIP") from exc

    @staticmethod
    def _select_result(data: Any, file_name: str) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise RuntimeError("MinerU batch response is missing data")
        results = data.get("extract_result")
        if isinstance(results, Mapping):
            results = [results]
        if not isinstance(results, list) or not results:
            raise RuntimeError("MinerU batch response is missing data.extract_result")
        mappings = [dict(item) for item in results if isinstance(item, Mapping)]
        for item in mappings:
            if item.get("file_name") == file_name:
                return item
        if len(mappings) == 1:
            return mappings[0]
        raise RuntimeError(
            f"MinerU batch response has no extraction result for {file_name}"
        )

    def _api_headers(self) -> dict[str, str]:
        return {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json",
        }

    @classmethod
    def _api_payload(cls, response: Any, action: str) -> dict[str, Any]:
        cls._raise_for_status(response, action)
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to {action}: response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unable to {action}: response root is not an object")
        code = payload.get("code")
        if code != 0:
            detail = payload.get("msg") or "unknown API error"
            raise RuntimeError(f"Unable to {action}: MinerU code {code}: {detail}")
        return payload

    @staticmethod
    def _request(action: str, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(f"Unable to {action}: {exc}") from exc

    @staticmethod
    def _raise_for_status(response: Any, action: str) -> None:
        status = int(getattr(response, "status_code", 0))
        if 200 <= status < 300:
            return
        detail = str(getattr(response, "text", "")).strip()
        raise RuntimeError(
            f"Unable to {action}: HTTP {status}"
            + (f": {detail[:500]}" if detail else "")
        )

    def _number(self, key: str, *, default: float, allow_zero: bool) -> float:
        value = self.config.get(key, default)
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and value >= 0
            and (allow_zero or value > 0)
        )
        if not valid:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"MinerU api.{key} must be a {qualifier} number")
        return float(value)

    @staticmethod
    def _validate_path(file_path: str | Path) -> Path:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path


__all__ = ["MinerUApiClient"]
