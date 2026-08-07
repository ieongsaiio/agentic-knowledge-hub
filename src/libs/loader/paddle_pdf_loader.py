"""PDF loader backed by PaddleOCR-VL through Docker or the hosted API."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_DEFAULT_DOCKER_IMAGE = "agentic-knowledge-hub/paddleocr-vl-transformers:latest"
_DEFAULT_RESTRUCTURE = {
    "merge_tables": False,
    "relevel_titles": True,
    "concatenate_pages": False,
}
_SECRET_KEYS = {"token", "api_key", "access_token"}


class PaddlePdfLoader(BaseLoader):
    """Load a PDF from the artifact emitted by PaddleOCR-VL."""

    def __init__(
        self,
        paddle_config: dict[str, Any] | None = None,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
        runner: Callable[..., Any] | None = None,
    ) -> None:
        config = copy.deepcopy(paddle_config or {})
        self.paddle_config = config
        self.backend = str(
            config.get("backend", config.get("execution", "docker"))
        ).strip().lower()
        if self.backend not in {"docker", "api"}:
            raise ValueError("PaddleOCR backend must be one of: docker, api")
        self.docker_config = self._backend_config(config, "docker")
        self.api_config = self._backend_config(config, "api")
        self.extract_images = bool(extract_images)
        self.image_storage_dir = Path(image_storage_dir)
        self.runner = runner

    def cache_config(self) -> dict[str, Any]:
        """Return every parsing policy that can alter cached output."""
        selected_config = (
            self.docker_config if self.backend == "docker" else self.api_config
        )
        restructure = {
            key: self.docker_config.get(key, default)
            for key, default in _DEFAULT_RESTRUCTURE.items()
        }
        sanitized_selected = {
            key: copy.deepcopy(value)
            for key, value in selected_config.items()
            if key not in _SECRET_KEYS
            and key
            not in {
                "poll_interval_seconds",
                "timeout_seconds",
                "request_timeout_seconds",
            }
        }
        return {
            "loader": "PaddlePdfLoader",
            "loader_schema_version": 6,
            "provider": "paddle",
            "parser": "paddleocr",
            "backend": self.backend,
            "extract_images": self.extract_images,
            "engine": (
                self.docker_config.get("engine", "transformers")
                if self.backend == "docker"
                else "hosted_api"
            ),
            "restructure": restructure if self.backend == "docker" else None,
            "image_policy": {
                "extract_images": self.extract_images,
                "markdown_images": "placeholder" if self.extract_images else "remove",
            },
            "paddle_config": {
                "backend": self.backend,
                self.backend: sanitized_selected,
            },
        }

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        doc_hash = self._compute_file_hash(path)
        artifact, artifact_root = self._run(path, doc_hash)
        try:
            return self._document_from_artifact(
                path,
                doc_hash,
                artifact,
                artifact_root,
            )
        finally:
            if self.runner is None and artifact_root is not None:
                shutil.rmtree(artifact_root, ignore_errors=True)

    async def aload(self, file_path: str | Path) -> Document:
        """Load without blocking the event loop while the remote API is polled."""
        if self.backend != "api" or self.runner is not None:
            return await asyncio.to_thread(self.load, file_path)

        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")
        doc_hash = await asyncio.to_thread(self._compute_file_hash, path)
        from src.libs.loader.paddleocr_api_client import PaddleOcrApiClient

        artifact = await PaddleOcrApiClient(self.api_config).run_async(path)
        return self._document_from_artifact(path, doc_hash, artifact, None)

    def _run(self, path: Path, doc_hash: str) -> tuple[dict[str, Any], Path | None]:
        try:
            if self.runner is None:
                if self.backend == "api":
                    return self._run_api(path)
                return self._run_docker(path)

            target = getattr(self.runner, "run", self.runner)
            result = target(
                path,
                copy.deepcopy(self.paddle_config),
                self.image_storage_dir / doc_hash,
            )
            return self._read_artifact(result)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR runner failed for {path}: {exc}") from exc

    def _run_api(self, path: Path) -> tuple[dict[str, Any], None]:
        from src.libs.loader.paddleocr_api_client import PaddleOcrApiClient

        return PaddleOcrApiClient(self.api_config).run(path), None

    def _run_docker(self, path: Path) -> tuple[dict[str, Any], Path]:
        docker_image = str(
            self.docker_config.get("docker_image", _DEFAULT_DOCKER_IMAGE)
        ).strip()
        if not docker_image:
            raise RuntimeError("PaddleOCR docker_image cannot be empty")

        timeout = self.docker_config.get("timeout_seconds", 7200)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise RuntimeError("PaddleOCR timeout_seconds must be a positive number")

        with tempfile.TemporaryDirectory(prefix="paddle-pdf-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_path = temp_dir / "artifact.json"
            command = ["docker", "run", "--rm"]
            command.extend(
                self._docker_device_args(self.docker_config.get("device", "gpu:0"))
            )
            cache_volume = str(
                self.docker_config.get(
                    "docker_cache_volume",
                    "paddleocr-transformers-cache",
                )
            ).strip()
            if cache_volume:
                command.extend(["-v", f"{cache_volume}:/workspace/.cache"])
            paddlex_cache_volume = str(
                self.docker_config.get(
                    "paddlex_cache_volume", "paddleocr-paddlex-cache"
                )
            ).strip()
            if paddlex_cache_volume:
                command.extend(["-v", f"{paddlex_cache_volume}:/root/.paddlex"])
            shm_size = str(self.docker_config.get("shm_size", "4g")).strip()
            if shm_size:
                command.extend(["--shm-size", shm_size])
            command.extend(
                [
                    "-v",
                    f"{path}:/workspace/input/document.pdf:ro",
                    "-v",
                    f"{temp_dir}:/workspace/output",
                    docker_image,
                    "--input",
                    "/workspace/input/document.pdf",
                    "--output",
                    "/workspace/output/artifact.json",
                    "--config-json",
                    json.dumps(self.docker_config, ensure_ascii=True),
                ]
            )
            if self.extract_images:
                command.append("--vision-enabled")

            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=float(timeout),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"PaddleOCR Docker runner timed out after {timeout} seconds"
                ) from exc
            except OSError as exc:
                raise RuntimeError(f"Unable to start Docker: {exc}") from exc

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "no output").strip()
                raise RuntimeError(
                    "PaddleOCR Docker runner failed with exit code "
                    f"{completed.returncode}: {detail}"
                )
            artifact, _ = self._read_artifact(artifact_path)

            # Docker writes assets beside artifact.json. Preserve them before
            # the temporary bind mount is removed.
            persistent_root = Path(tempfile.mkdtemp(prefix="paddle-artifact-"))
            try:
                assets = temp_dir / "assets"
                if assets.is_dir():
                    shutil.copytree(assets, persistent_root / "assets")
                return artifact, persistent_root
            except Exception:
                shutil.rmtree(persistent_root, ignore_errors=True)
                raise

    def _document_from_artifact(
        self,
        path: Path,
        doc_hash: str,
        artifact: dict[str, Any],
        artifact_root: Path | None,
    ) -> Document:
        page_texts: list[str] = []
        page_images: list[list[dict[str, Any]]] = []
        page_numbers: list[int] = []
        pages = self._artifact_pages(artifact)
        for page in pages:
            page_number = page["page_index"] + 1
            text, images = self._normalize_images(
                page["markdown_text"],
                page.get("images"),
                artifact_root,
                doc_hash,
                page_number,
            )
            page_texts.append(text)
            page_images.append(images)
            page_numbers.append(page_number)

        text, page_spans, images = self._combine_pages(
            page_texts,
            page_images,
            page_numbers,
        )
        raw_pages = artifact.get("pages")
        page_count = len(raw_pages) if isinstance(raw_pages, list) else len(pages)
        metadata: dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "page_count": page_count,
            "page_spans": page_spans,
        }
        title = self._extract_title(text)
        if title:
            metadata["title"] = title
        if images:
            metadata["images"] = images
        metadata["parsed_artifact"] = artifact
        document = Document(
            id=f"doc_{doc_hash[:16]}",
            text=text,
            metadata=metadata,
        )
        return self._attach_section_tree(document)

    @staticmethod
    def _backend_config(
        config: Mapping[str, Any],
        backend: str,
    ) -> dict[str, Any]:
        nested = config.get(backend)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise ValueError(f"PaddleOCR paddle.{backend} must be a mapping")
            return copy.deepcopy(dict(nested))

        # Compatibility with the previous flat ``paddle.execution`` schema.
        excluded = {"backend", "execution", "docker", "api"}
        return {
            key: copy.deepcopy(value)
            for key, value in config.items()
            if key not in excluded
        }

    @staticmethod
    def _docker_device_args(device: Any) -> list[str]:
        normalized = str(device or "cpu").strip().lower()
        if normalized in {"", "cpu", "none"}:
            return []
        if normalized in {"gpu", "all"}:
            return ["--gpus", "all"]
        if normalized.startswith("gpu:"):
            normalized = normalized.split(":", 1)[1]
        return ["--gpus", f"device={normalized}"]

    @staticmethod
    def _read_artifact(result: Any) -> tuple[dict[str, Any], Path | None]:
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result)), None
        if isinstance(result, (str, Path)):
            artifact_path = Path(result)
            if not artifact_path.is_file():
                raise RuntimeError(f"PaddleOCR artifact JSON was not found: {artifact_path}")
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Unable to read PaddleOCR artifact JSON {artifact_path}: {exc}"
                ) from exc
            if not isinstance(artifact, dict):
                raise RuntimeError("PaddleOCR artifact JSON root must be an object")
            return artifact, artifact_path.parent

        # Lightweight artifact objects are useful for tests and custom runners.
        restructured = getattr(result, "restructured_pages", None)
        if isinstance(restructured, list):
            artifact = {
                "pages": copy.deepcopy(getattr(result, "pages", restructured)),
                "restructured_pages": copy.deepcopy(restructured),
            }
            root = getattr(result, "root", None)
            return artifact, Path(root) if root is not None else None
        raise RuntimeError("PaddleOCR runner must return an artifact mapping, object, or JSON path")

    @staticmethod
    def _artifact_pages(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw_pages = artifact.get("pages")
        restructured = artifact.get("restructured_pages")
        if not isinstance(raw_pages, list):
            raise RuntimeError("Invalid PaddleOCR artifact: 'pages' must be a list")
        if not isinstance(restructured, list):
            raise RuntimeError("Invalid PaddleOCR artifact: 'restructured_pages' must be a list")

        pages: list[dict[str, Any]] = []
        for index, raw_page in enumerate(restructured):
            if not isinstance(raw_page, Mapping):
                raise RuntimeError(
                    f"Invalid PaddleOCR artifact: restructured_pages[{index}] must be an object"
                )
            page = dict(raw_page)
            page_index = page.get("page_index")
            markdown = page.get("markdown_text")
            if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
                raise RuntimeError(
                    f"Invalid PaddleOCR artifact: restructured_pages[{index}].page_index "
                    "must be a non-negative integer"
                )
            if not isinstance(markdown, str):
                raise RuntimeError(
                    f"Invalid PaddleOCR artifact: restructured_pages[{index}].markdown_text "
                    "must be a string"
                )
            if "json" not in page:
                raise RuntimeError(
                    f"Invalid PaddleOCR artifact: restructured_pages[{index}] is missing 'json'"
                )
            pages.append(page)

        pages.sort(key=lambda page: page["page_index"])
        return pages

    def _normalize_images(
        self,
        markdown: str,
        raw_images: Any,
        artifact_root: Path | None,
        doc_hash: str,
        page_number: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not self.extract_images:
            return _MARKDOWN_IMAGE_RE.sub("", markdown), []

        descriptors = self._image_descriptors(raw_images)
        by_reference: dict[str, dict[str, Any]] = {}
        for descriptor in descriptors:
            for key in ("reference", "path", "image_path", "src", "url"):
                value = descriptor.get(key)
                if isinstance(value, str):
                    by_reference[value] = descriptor

        images: list[dict[str, Any]] = []
        offset_delta = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal offset_delta
            target = self._markdown_target(match.group(1))
            sequence = len(images) + 1
            descriptor = by_reference.get(target)
            if descriptor is None and sequence <= len(descriptors):
                descriptor = descriptors[sequence - 1]
            descriptor = descriptor or {"path": target}
            image_id = str(descriptor.get("id") or f"{doc_hash[:8]}_{page_number}_{sequence}")
            placeholder = f"[IMAGE: {image_id}]"
            stored_path = self._store_image(descriptor, target, artifact_root, doc_hash, image_id)
            position = descriptor.get("position")
            if not isinstance(position, Mapping):
                position = {
                    key: descriptor[key]
                    for key in ("bbox", "width", "height", "index")
                    if key in descriptor
                }
            else:
                position = dict(position)
            position.setdefault("page", page_number)
            images.append(
                {
                    "id": image_id,
                    "path": stored_path,
                    "page": page_number,
                    "text_offset": match.start() + offset_delta,
                    "text_length": len(placeholder),
                    "position": position,
                }
            )
            offset_delta += len(placeholder) - len(match.group(0))
            return placeholder

        return _MARKDOWN_IMAGE_RE.sub(replace, markdown), images

    def _store_image(
        self,
        descriptor: Mapping[str, Any],
        markdown_target: str,
        artifact_root: Path | None,
        doc_hash: str,
        image_id: str,
    ) -> str:
        source_value = next(
            (
                descriptor.get(key)
                for key in ("path", "image_path", "src", "url")
                if descriptor.get(key)
            ),
            markdown_target,
        )
        source = Path(str(source_value))
        if not source.is_absolute() and artifact_root is not None:
            source = artifact_root / source
        if not source.is_file():
            return str(source)

        destination_dir = self.image_storage_dir / doc_hash
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{image_id}{source.suffix or '.png'}"
        shutil.copy2(source, destination)
        try:
            return str(destination.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(destination.resolve())

    @staticmethod
    def _image_descriptors(raw_images: Any) -> list[dict[str, Any]]:
        if raw_images is None:
            return []
        if isinstance(raw_images, list):
            return [
                dict(item) if isinstance(item, Mapping) else {"path": str(item)}
                for item in raw_images
            ]
        if isinstance(raw_images, Mapping):
            descriptors: list[dict[str, Any]] = []
            for reference, value in raw_images.items():
                descriptor = dict(value) if isinstance(value, Mapping) else {"path": value}
                descriptor.setdefault("reference", str(reference))
                descriptors.append(descriptor)
            return descriptors
        raise RuntimeError("Invalid PaddleOCR artifact: page images must be a list or object")

    @staticmethod
    def _markdown_target(raw_target: str) -> str:
        target = raw_target.strip()
        if target.startswith("<") and ">" in target:
            return target[1 : target.index(">")]
        return re.split(r"\s+[\"']", target, maxsplit=1)[0]

    @staticmethod
    def _combine_pages(
        page_texts: list[str],
        page_images: list[list[dict[str, Any]]],
        page_numbers: list[int],
    ) -> tuple[str, list[dict[str, int]], list[dict[str, Any]]]:
        parts: list[str] = []
        spans: list[dict[str, int]] = []
        images: list[dict[str, Any]] = []
        cursor = 0
        for index, page_text in enumerate(page_texts):
            if index:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(page_text)
            for image in page_images[index]:
                adjusted = dict(image)
                adjusted["text_offset"] = start + image["text_offset"]
                images.append(adjusted)
            cursor += len(page_text)
            spans.append(
                {
                    "page": page_numbers[index],
                    "start_offset": start,
                    "end_offset": cursor,
                }
            )
        return "".join(parts), spans, images

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for line in text.splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or None
        return None
