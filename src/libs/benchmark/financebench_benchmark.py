"""FinanceBench dataset download and normalization."""

from __future__ import annotations

import json
import random
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import requests

from src.libs.benchmark.base_benchmark import (
    BaseBenchmark,
    BenchmarkCase,
    BenchmarkEvidence,
)

DEFAULT_SOURCE_URL = "https://github.com/patronus-ai/financebench"
_DEFAULT_ARCHIVE_URL = f"{DEFAULT_SOURCE_URL}/archive/refs/heads/main.zip"
_DATASET_RELATIVE_PATH = Path("data") / "financebench_open_source.jsonl"
_PDFS_RELATIVE_PATH = Path("pdfs")
_SETTING_UNSET: Any = object()


def _row_error(row_number: int | None, message: str) -> ValueError:
    location = f" at row {row_number}" if row_number is not None else ""
    return ValueError(f"Malformed FinanceBench case{location}: {message}")


def _required_text(raw: Mapping[str, Any], field: str, row_number: int | None) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _row_error(row_number, f"'{field}' must be a non-empty string")
    return value


def normalize_financebench_case(
    raw: Mapping[str, Any], row_number: int | None = None
) -> BenchmarkCase:
    """Convert one official FinanceBench row to the internal benchmark model."""
    if not isinstance(raw, Mapping):
        raise _row_error(row_number, "row must be a JSON object")

    raw_case_id = raw.get("financebench_id")
    if raw_case_id is None or isinstance(raw_case_id, (dict, list)):
        raise _row_error(row_number, "'financebench_id' is required and must be scalar")
    case_id = str(raw_case_id).strip()
    if not case_id:
        raise _row_error(row_number, "'financebench_id' cannot be empty")

    query = _required_text(raw, "question", row_number)
    reference_answer = _required_text(raw, "answer", row_number)

    raw_evidences = raw.get("evidence")
    if not isinstance(raw_evidences, list) or not raw_evidences:
        raise _row_error(row_number, "'evidence' must be a non-empty list")

    fallback_document = raw.get("doc_name")
    evidences: list[BenchmarkEvidence] = []
    for evidence_index, raw_evidence in enumerate(raw_evidences):
        prefix = f"evidence[{evidence_index}]"
        if not isinstance(raw_evidence, Mapping):
            raise _row_error(row_number, f"{prefix} must be an object")

        document_name = raw_evidence.get("evidence_doc_name") or fallback_document
        if not isinstance(document_name, str) or not document_name.strip():
            raise _row_error(
                row_number,
                f"{prefix} requires a non-empty 'evidence_doc_name' or row 'doc_name'",
            )

        page_number = raw_evidence.get("evidence_page_num")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise _row_error(row_number, f"{prefix}.evidence_page_num must be a zero-based integer")
        if page_number < 0:
            raise _row_error(row_number, f"{prefix}.evidence_page_num cannot be negative")

        evidence_text = raw_evidence.get("evidence_text")
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            raise _row_error(row_number, f"{prefix}.evidence_text must be a non-empty string")

        evidences.append(
            BenchmarkEvidence(
                document_name=document_name,
                page_number=page_number + 1,
                text=evidence_text,
            )
        )

    metadata_fields = (
        "doc_name",
        "question_type",
        "question_reasoning",
        "reasoning",
        "company",
        "justification",
    )
    metadata = {field: raw[field] for field in metadata_fields if field in raw}
    if "reasoning" not in metadata and "question_reasoning" in metadata:
        metadata["reasoning"] = metadata["question_reasoning"]

    return BenchmarkCase(
        case_id=case_id,
        query=query,
        reference_answer=reference_answer,
        evidences=evidences,
        metadata=metadata,
    )


class FinanceBenchBenchmark(BaseBenchmark):
    """Adapter for Patronus AI's open-source FinanceBench sample."""

    source_url = DEFAULT_SOURCE_URL

    def __init__(
        self,
        settings: Any = None,
        *,
        data_dir: str | Path | None = None,
        source_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected FinanceBench settings: {unknown}")
        if settings is None:
            if data_dir is None:
                raise ValueError("settings or data_dir is required")
            settings = {"data_dir": data_dir}
        elif isinstance(settings, (str, Path)):
            if data_dir is not None:
                raise ValueError("data_dir was provided twice")
            settings = {"data_dir": settings}
        elif data_dir is not None:
            if not isinstance(settings, Mapping):
                raise ValueError("data_dir override requires mapping-based benchmark settings")
            settings = dict(settings)
            settings["data_dir"] = data_dir

        super().__init__(settings)
        configured_source_url = self._setting("source_url", default=DEFAULT_SOURCE_URL)
        selected_source_url = source_url or configured_source_url
        if not isinstance(selected_source_url, str) or not selected_source_url.strip():
            raise ValueError("settings.source_url must be a non-empty string")
        self.source_url = selected_source_url

    @property
    def dataset_path(self) -> Path:
        """Return the expected normalized location of the source JSONL."""
        return self.data_dir / _DATASET_RELATIVE_PATH

    @property
    def pdfs_dir(self) -> Path:
        """Return the expected normalized location of source PDFs."""
        return self.data_dir / _PDFS_RELATIVE_PATH

    def _locate_artifacts(self, root: Path | None = None) -> tuple[Path, Path]:
        search_root = Path(root or self.data_dir)
        preferred_dataset = search_root / _DATASET_RELATIVE_PATH
        preferred_pdfs = search_root / _PDFS_RELATIVE_PATH
        if preferred_dataset.is_file() and self._valid_pdfs_dir(preferred_pdfs):
            return preferred_dataset, preferred_pdfs

        if not search_root.exists():
            raise FileNotFoundError(f"FinanceBench data directory does not exist: {search_root}")

        candidates = sorted(search_root.rglob(_DATASET_RELATIVE_PATH.name))
        for dataset_path in candidates:
            if dataset_path.parent.name != "data":
                continue
            repository_root = dataset_path.parent.parent
            pdfs_path = repository_root / "pdfs"
            if self._valid_pdfs_dir(pdfs_path):
                return dataset_path, pdfs_path

        raise FileNotFoundError(
            "FinanceBench artifacts are incomplete under "
            f"{search_root}: expected data/{_DATASET_RELATIVE_PATH.name} "
            "and a non-empty pdfs directory"
        )

    @staticmethod
    def _valid_pdfs_dir(path: Path) -> bool:
        return path.is_dir() and any(
            candidate.is_file() and candidate.suffix.lower() == ".pdf"
            for candidate in path.rglob("*")
        )

    def verify(self) -> bool:
        """Validate that the JSONL and PDF corpus are both available."""
        dataset_path, pdfs_path = self._locate_artifacts()
        if dataset_path.stat().st_size == 0:
            raise ValueError(f"FinanceBench JSONL is empty: {dataset_path}")
        if not self._valid_pdfs_dir(pdfs_path):
            raise ValueError(f"FinanceBench PDF directory is empty: {pdfs_path}")
        return True

    def _archive_url(self) -> str:
        source_url = self.source_url.rstrip("/")
        if source_url == DEFAULT_SOURCE_URL:
            return _DEFAULT_ARCHIVE_URL
        if source_url.lower().endswith(".zip"):
            return source_url
        if source_url.startswith("https://github.com/"):
            return f"{source_url}/archive/refs/heads/main.zip"
        return source_url

    def download(self) -> Path:
        """Download and safely install the official FinanceBench repository archive."""
        try:
            dataset_path, _ = self._locate_artifacts()
            self.verify()
            return dataset_path
        except FileNotFoundError:
            pass

        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".financebench-", dir=self.data_dir.parent
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            archive_path = temporary_root / "financebench.zip"
            extraction_root = temporary_root / "extracted"

            try:
                response = requests.get(self._archive_url(), stream=True, timeout=(10, 120))
                response.raise_for_status()
                with archive_path.open("wb") as archive_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            archive_file.write(chunk)
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Failed to download FinanceBench from {self._archive_url()}: {exc}"
                ) from exc

            try:
                with zipfile.ZipFile(archive_path) as archive:
                    self._safe_extract(archive, extraction_root)
            except (OSError, zipfile.BadZipFile) as exc:
                raise RuntimeError(f"Downloaded FinanceBench archive is invalid: {exc}") from exc

            source_dataset, source_pdfs = self._locate_artifacts(extraction_root)
            source_data_dir = source_dataset.parent
            self.data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_data_dir, self.data_dir / "data", dirs_exist_ok=True)
            shutil.copytree(source_pdfs, self.data_dir / "pdfs", dirs_exist_ok=True)

        self.verify()
        return self.dataset_path

    @staticmethod
    def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        destination_resolved = destination.resolve()

        for member in archive.infolist():
            normalized_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(normalized_name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or PureWindowsPath(normalized_name).drive
            ):
                raise ValueError(f"Unsafe path in FinanceBench archive: {member.filename!r}")

            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"Symlinks are not allowed in FinanceBench archive: {member.filename!r}"
                )

            target = destination.joinpath(*member_path.parts).resolve()
            try:
                target.relative_to(destination_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe path in FinanceBench archive: {member.filename!r}"
                ) from exc

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

    def normalize(self, raw: Mapping[str, Any], row_number: int | None = None) -> BenchmarkCase:
        return normalize_financebench_case(raw, row_number=row_number)

    def load_cases(
        self,
        sample_size: int | None = _SETTING_UNSET,
        seed: int = _SETTING_UNSET,
    ) -> list[BenchmarkCase]:
        """Load, validate, normalize, and optionally sample FinanceBench cases."""
        if sample_size is _SETTING_UNSET:
            sample_size = self._setting("sample_size", default=None)
        if seed is _SETTING_UNSET:
            seed = self._setting("seed", default=42)

        try:
            dataset_path, _ = self._locate_artifacts()
        except FileNotFoundError:
            if not self._auto_download:
                raise FileNotFoundError(
                    "FinanceBench artifacts are missing and auto_download is disabled"
                )
            dataset_path = self.download()

        cases: list[BenchmarkCase] = []
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            for row_number, line in enumerate(dataset_file, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed FinanceBench JSONL at {dataset_path}:{row_number}: {exc.msg}"
                    ) from exc
                cases.append(self.normalize(raw, row_number=row_number))

        if not cases:
            raise ValueError(f"FinanceBench JSONL contains no cases: {dataset_path}")
        if sample_size is None:
            return cases
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise ValueError("sample_size must be an integer or None")
        if sample_size < 0:
            raise ValueError("sample_size cannot be negative")
        if sample_size > len(cases):
            raise ValueError(f"sample_size ({sample_size}) exceeds available cases ({len(cases)})")

        indices = random.Random(seed).sample(range(len(cases)), sample_size)
        return [cases[index] for index in indices]

    def document_paths(self) -> list[Path]:
        """Return FinanceBench PDFs in deterministic path order."""
        try:
            _, pdfs_path = self._locate_artifacts()
        except FileNotFoundError:
            if not self._auto_download:
                raise FileNotFoundError(
                    "FinanceBench artifacts are missing and auto_download is disabled"
                )
            self.download()
            _, pdfs_path = self._locate_artifacts()
        return sorted(path for path in pdfs_path.rglob("*") if path.suffix.lower() == ".pdf")

    # Compatibility aliases for minor base-contract naming differences.
    download_data = download
    prepare_data = download
