"""Client for the asynchronous PaddleOCR Studio document parsing API."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests

_DEFAULT_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
_DEFAULT_MODEL = "PaddleOCR-VL-1.6"
_DEFAULT_OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}
_SECRET_KEYS = {"token", "api_key", "access_token"}


class PaddleOcrApiClient:
    """Submit PDF parsing jobs and normalize JSONL results for ``PaddlePdfLoader``."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        session: Any = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config or {}))
        self.session = session or requests.Session()
        self.job_url = str(self.config.get("job_url", _DEFAULT_JOB_URL)).rstrip("/")
        self.model = str(self.config.get("model", _DEFAULT_MODEL))
        self.token_env = str(
            self.config.get("token_env", "PADDLEOCR_API_TOKEN")
        ).strip()
        token = str(self.config.get("token") or os.getenv(self.token_env, "")).strip()
        if not token:
            raise ValueError(
                f"PaddleOCR API token is missing; set environment variable {self.token_env}"
            )
        self._token = token
        self.poll_interval = self._positive_number(
            "poll_interval_seconds",
            allow_zero=True,
            default=5,
        )
        self.timeout = self._positive_number(
            "timeout_seconds",
            allow_zero=False,
            default=1800,
        )
        self.request_timeout = self._positive_number(
            "request_timeout_seconds",
            allow_zero=False,
            default=120,
        )
        optional_payload = self.config.get(
            "optional_payload",
            _DEFAULT_OPTIONAL_PAYLOAD,
        )
        if not isinstance(optional_payload, Mapping):
            raise ValueError("PaddleOCR api.optional_payload must be a mapping")
        self.optional_payload = copy.deepcopy(dict(optional_payload))

    @property
    def authorization_header(self) -> str:
        return f"bearer {self._token}"

    def cache_config(self) -> dict[str, Any]:
        """Return output-affecting API settings without credentials."""
        return {
            key: copy.deepcopy(value)
            for key, value in self.config.items()
            if key not in _SECRET_KEYS
            and key
            not in {
                "poll_interval_seconds",
                "timeout_seconds",
                "request_timeout_seconds",
            }
        }

    def run(self, file_path: str | Path) -> dict[str, Any]:
        """Submit, poll, download, and normalize one PDF synchronously."""
        path = self._validate_path(file_path)
        started_at = time.perf_counter()
        job_id = self._submit(path)
        job_data = self._poll(job_id, started_at)
        result_text = self._download_result(job_data)
        return self._build_artifact(result_text, job_id, time.perf_counter() - started_at)

    async def run_async(self, file_path: str | Path) -> dict[str, Any]:
        """Asynchronous facade using worker threads for ``requests`` I/O."""
        path = self._validate_path(file_path)
        started_at = time.perf_counter()
        job_id = await asyncio.to_thread(self._submit, path)
        job_data = await self._poll_async(job_id, started_at)
        result_text = await asyncio.to_thread(self._download_result, job_data)
        return self._build_artifact(
            result_text,
            job_id,
            time.perf_counter() - started_at,
        )

    def _submit(self, path: Path) -> str:
        headers = {"Authorization": self.authorization_header}
        data = {
            "model": self.model,
            "optionalPayload": json.dumps(self.optional_payload),
        }
        with path.open("rb") as source:
            response = self.session.post(
                self.job_url,
                headers=headers,
                data=data,
                files={"file": (path.name, source, "application/pdf")},
                timeout=self.request_timeout,
            )
        payload = self._response_json(response, "submit PaddleOCR job")
        job_id = self._nested(payload, "data", "jobId")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("PaddleOCR submit response is missing data.jobId")
        return job_id

    def _poll(self, job_id: str, started_at: float) -> dict[str, Any]:
        while True:
            data = self._get_job(job_id)
            state = str(data.get("state", "")).lower()
            if state == "done":
                return data
            if state == "failed":
                detail = data.get("errorMsg") or "unknown remote error"
                raise RuntimeError(f"PaddleOCR API job {job_id} failed: {detail}")
            if state not in {"pending", "running"}:
                raise RuntimeError(
                    f"PaddleOCR API job {job_id} returned unknown state: {state!r}"
                )
            self._check_deadline(job_id, started_at)
            time.sleep(self.poll_interval)

    async def _poll_async(self, job_id: str, started_at: float) -> dict[str, Any]:
        while True:
            data = await asyncio.to_thread(self._get_job, job_id)
            state = str(data.get("state", "")).lower()
            if state == "done":
                return data
            if state == "failed":
                detail = data.get("errorMsg") or "unknown remote error"
                raise RuntimeError(f"PaddleOCR API job {job_id} failed: {detail}")
            if state not in {"pending", "running"}:
                raise RuntimeError(
                    f"PaddleOCR API job {job_id} returned unknown state: {state!r}"
                )
            self._check_deadline(job_id, started_at)
            await asyncio.sleep(self.poll_interval)

    def _get_job(self, job_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.job_url}/{job_id}",
            headers={"Authorization": self.authorization_header},
            timeout=self.request_timeout,
        )
        payload = self._response_json(response, f"poll PaddleOCR job {job_id}")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise RuntimeError("PaddleOCR job response is missing data")
        return dict(data)

    def _download_result(self, job_data: Mapping[str, Any]) -> str:
        result_url = job_data.get("resultUrl")
        if not isinstance(result_url, Mapping):
            raise RuntimeError("PaddleOCR completed job is missing data.resultUrl")
        json_url = result_url.get("jsonUrl")
        if not isinstance(json_url, str) or not json_url:
            raise RuntimeError("PaddleOCR completed job is missing resultUrl.jsonUrl")
        response = self.session.get(json_url, timeout=self.request_timeout)
        self._raise_for_status(response, "download PaddleOCR JSONL")
        return str(response.text)

    def _build_artifact(
        self,
        jsonl_text: str,
        job_id: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        raw_pages: list[dict[str, Any]] = []
        restructured_pages: list[dict[str, Any]] = []
        fallback_index = 0
        for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"PaddleOCR result JSONL line {line_number} is invalid: {exc}"
                ) from exc
            result = payload.get("result") if isinstance(payload, Mapping) else None
            layouts = (
                result.get("layoutParsingResults")
                if isinstance(result, Mapping)
                else None
            )
            if not isinstance(layouts, list):
                raise RuntimeError(
                    f"PaddleOCR result JSONL line {line_number} is missing "
                    "result.layoutParsingResults"
                )
            for layout in layouts:
                if not isinstance(layout, Mapping):
                    raise RuntimeError("PaddleOCR layout parsing result must be an object")
                page_index = self._page_index(layout, fallback_index)
                normalized_layout = copy.deepcopy(dict(layout))
                normalized_layout["page_index"] = page_index
                markdown = layout.get("markdown")
                if not isinstance(markdown, Mapping):
                    markdown = {}
                markdown_text = markdown.get(
                    "text",
                    markdown.get("markdown_texts", markdown.get("markdown_text", "")),
                )
                if not isinstance(markdown_text, str):
                    raise RuntimeError(
                        f"PaddleOCR page {page_index} markdown text must be a string"
                    )
                images = markdown.get(
                    "images",
                    markdown.get("markdown_images", {}),
                )
                if not isinstance(images, (Mapping, list)):
                    images = {}
                raw_pages.append({"res": normalized_layout})
                restructured_pages.append(
                    {
                        "page_index": page_index,
                        "markdown_text": markdown_text,
                        "json": normalized_layout,
                        "images": copy.deepcopy(images),
                    }
                )
                fallback_index += 1
        if not restructured_pages:
            raise RuntimeError("PaddleOCR result JSONL contains no parsed pages")
        raw_pages.sort(key=lambda page: page["res"]["page_index"])
        restructured_pages.sort(key=lambda page: page["page_index"])
        return {
            "elapsed_seconds": elapsed_seconds,
            "config": self.cache_config(),
            "api_job_id": job_id,
            "pages": raw_pages,
            "restructured_pages": restructured_pages,
        }

    @staticmethod
    def _page_index(layout: Mapping[str, Any], fallback: int) -> int:
        for key in ("pageIndex", "page_index", "pageNum", "page_num"):
            value = layout.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return fallback

    def _check_deadline(self, job_id: str, started_at: float) -> None:
        if time.perf_counter() - started_at >= self.timeout:
            raise RuntimeError(
                f"PaddleOCR API job {job_id} timed out after {self.timeout:g} seconds"
            )

    def _positive_number(
        self,
        key: str,
        *,
        allow_zero: bool,
        default: float,
    ) -> float:
        value = self.config.get(key, default)
        minimum_ok = value >= 0 if isinstance(value, (int, float)) else False
        if isinstance(value, bool) or not minimum_ok or (not allow_zero and value == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"PaddleOCR api.{key} must be a {qualifier} number")
        return float(value)

    @staticmethod
    def _validate_path(file_path: str | Path) -> Path:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    @classmethod
    def _response_json(cls, response: Any, action: str) -> dict[str, Any]:
        cls._raise_for_status(response, action)
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to {action}: response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unable to {action}: response root is not an object")
        return payload

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

    @staticmethod
    def _nested(value: Mapping[str, Any], *keys: str) -> Any:
        current: Any = value
        for key in keys:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current


__all__ = ["PaddleOcrApiClient"]
