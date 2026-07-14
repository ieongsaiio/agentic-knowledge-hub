"""Shared data contract and base class for benchmark providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MISSING = object()


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class BenchmarkEvidence:
    """A single piece of reference evidence for a benchmark case."""

    document_name: str
    page_number: int
    text: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.document_name, "document_name")
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
            raise ValueError("page_number must be an integer")
        if self.page_number < 1:
            raise ValueError("page_number must be 1-based (greater than or equal to 1)")
        _validate_non_empty_string(self.text, "text")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the evidence."""
        return {
            "document_name": self.document_name,
            "page_number": self.page_number,
            "text": self.text,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    """Provider-independent benchmark question and its reference evidence."""

    case_id: str
    query: str
    reference_answer: str
    evidences: list[BenchmarkEvidence]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.case_id, "case_id")
        _validate_non_empty_string(self.query, "query")
        _validate_non_empty_string(self.reference_answer, "reference_answer")
        if not isinstance(self.evidences, list):
            raise ValueError("evidences must be a list")
        for index, evidence in enumerate(self.evidences):
            if not isinstance(evidence, BenchmarkEvidence):
                raise ValueError(f"evidences[{index}] must be a BenchmarkEvidence instance")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    @property
    def expected_documents(self) -> list[str]:
        """Document names derived from the reference evidence."""
        return [evidence.document_name for evidence in self.evidences]

    @property
    def expected_pages(self) -> list[int]:
        """One-based page numbers derived from the reference evidence."""
        return [evidence.page_number for evidence in self.evidences]

    @property
    def expected_evidence(self) -> list[str]:
        """Evidence text derived from the reference evidence."""
        return [evidence.text for evidence in self.evidences]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the case."""
        return {
            "case_id": self.case_id,
            "query": self.query,
            "reference_answer": self.reference_answer,
            "evidences": [evidence.to_dict() for evidence in self.evidences],
            "metadata": dict(self.metadata),
        }


class BaseBenchmark(ABC):
    """Base contract for configuration-driven benchmark providers."""

    def __init__(self, settings: Any) -> None:
        if settings is None:
            raise ValueError("settings are required")

        self.settings = settings
        self._data_dir = self._resolve_data_dir(self._setting("data_dir", required=True))
        auto_download = self._setting("auto_download", default=True)
        if not isinstance(auto_download, bool):
            raise ValueError("settings.auto_download must be a boolean")
        self._auto_download = auto_download

    @property
    def data_dir(self) -> Path:
        """Absolute root directory for this provider's cached data."""
        return self._data_dir

    @property
    def pdf_dir(self) -> Path:
        """Directory containing benchmark source PDFs."""
        return self.data_dir / "pdfs"

    def prepare(self) -> list[BenchmarkCase]:
        """Create cache directories, optionally download, and load cases."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        if self._auto_download:
            self.download()

        cases = self.load_cases()
        return self._validate_cases(cases)

    @abstractmethod
    def download(self) -> None:
        """Download and cache provider data.

        Implementations should be idempotent so repeated preparation can reuse
        a valid local cache.
        """
        raise NotImplementedError

    @abstractmethod
    def load_cases(self) -> list[BenchmarkCase]:
        """Load normalized benchmark cases from the provider cache."""
        raise NotImplementedError

    def _setting(
        self,
        name: str,
        *,
        default: Any = _MISSING,
        required: bool = False,
    ) -> Any:
        if isinstance(self.settings, Mapping):
            value = self.settings.get(name, _MISSING)
        else:
            value = getattr(self.settings, name, _MISSING)

        if value is _MISSING:
            if required:
                raise ValueError(f"settings.{name} is required")
            if default is not _MISSING:
                return default
        return value

    @staticmethod
    def _resolve_data_dir(value: Any) -> Path:
        if not isinstance(value, (str, Path)):
            raise ValueError("settings.data_dir must be a path string or Path")
        if isinstance(value, str) and not value.strip():
            raise ValueError("settings.data_dir cannot be empty")

        path = Path(value).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve()

    @staticmethod
    def _validate_cases(cases: Any) -> list[BenchmarkCase]:
        if not isinstance(cases, list):
            raise TypeError("load_cases() must return a list of BenchmarkCase instances")

        seen_ids: set[str] = set()
        for index, case in enumerate(cases):
            if not isinstance(case, BenchmarkCase):
                raise TypeError(f"load_cases() item {index} must be a BenchmarkCase instance")
            if case.case_id in seen_ids:
                raise ValueError(f"Duplicate benchmark case_id: {case.case_id}")
            seen_ids.add(case.case_id)
        return cases
