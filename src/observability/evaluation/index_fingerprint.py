"""Deterministic fingerprints for index-affecting evaluation settings."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any

_MISSING = object()
INDEX_PAYLOAD_VERSION = 2
_UNSAFE_NAME_CHARS = re.compile(r"[^a-z0-9_-]+")
_KEY_PARTS = re.compile(r"[^a-z0-9]+")

_SECRET_PARTS = frozenset(
    {
        "authorization",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
        "token",
    }
)
_LOCATION_PARTS = frozenset(
    {
        "dir",
        "directory",
        "endpoint",
        "file",
        "filepath",
        "folder",
        "path",
        "uri",
        "url",
    }
)


def _get(value: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_mapping(value: Any) -> Mapping[Any, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, Mapping):
        return value_dict
    raise TypeError(f"Expected settings mapping or object, got {type(value).__name__}")


def _key_parts(key: Any) -> tuple[str, ...]:
    text = str(key)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return tuple(part for part in _KEY_PARTS.split(text.lower()) if part)


def _is_secret_key(key: Any) -> bool:
    parts = _key_parts(key)
    compact = "".join(parts)
    return (
        bool(_SECRET_PARTS.intersection(parts))
        or compact in {"apikey", "accesstoken", "authtoken", "privatekey"}
        or ("api" in parts and "key" in parts)
        or ("access" in parts and "key" in parts)
        or ("private" in parts and "key" in parts)
    )


def _is_location_key(key: Any) -> bool:
    parts = _key_parts(key)
    return bool(_LOCATION_PARTS.intersection(parts))


def _normalise_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Index fingerprint settings must contain finite numbers")
        return value
    if isinstance(value, Enum):
        return _normalise_value(value.value)
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(
        f"Index fingerprint settings must be JSON-compatible; got {type(value).__name__}"
    )


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Mapping) or (is_dataclass(value) and not isinstance(value, type)):
        return _sanitise_mapping(_as_mapping(value))
    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalised = [_normalise_value(item) for item in value]
        return sorted(normalised, key=_canonical_json)
    return _normalise_scalar(value)


def _sanitise_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    sanitised: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if _is_secret_key(key) or _is_location_key(key):
            continue
        normalised = _normalise_value(raw_value)
        if isinstance(normalised, dict) and not normalised:
            continue
        sanitised[key] = normalised
    return sanitised


def _section(settings: Any, name: str) -> Any:
    section = _get(settings, name)
    if section is _MISSING or section is None:
        raise ValueError(f"Missing required settings section: {name}")
    return section


def _selected(section: Any, names: tuple[str, ...], section_name: str) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name in names:
        value = _get(section, name)
        if value is _MISSING:
            raise ValueError(f"Missing required setting: {section_name}.{name}")
        selected[name] = _normalise_value(value)
    return selected


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    return bool(value)


def _uses_ingestion_llm(ingestion: Any) -> bool:
    for stage_name in ("chunk_refiner", "metadata_enricher"):
        stage = _get(ingestion, stage_name, None)
        if stage is not None and _enabled(_get(stage, "use_llm", False)):
            return True
    return False


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_index_payload(
    settings: Any,
    benchmark_provider: str = "financebench",
    dataset_version: str = "open_source",
) -> dict[str, Any]:
    """Build a JSON-compatible snapshot of settings that affect the index."""

    ingestion = _section(settings, "ingestion")
    embedding = _section(settings, "embedding")
    vector_store = _section(settings, "vector_store")

    payload: dict[str, Any] = {
        "index_payload_version": INDEX_PAYLOAD_VERSION,
        "benchmark": {
            "provider": str(benchmark_provider),
            "dataset_version": str(dataset_version),
        },
        "ingestion": _sanitise_mapping(_as_mapping(ingestion)),
        "embedding": _selected(
            embedding,
            ("provider", "model", "dimensions"),
            "embedding",
        ),
        "vector_store": _selected(
            vector_store,
            ("provider",),
            "vector_store",
        ),
    }

    if _uses_ingestion_llm(ingestion):
        payload["llm"] = _selected(
            _section(settings, "llm"),
            ("provider", "model", "temperature"),
            "llm",
        )

    vision = _get(settings, "vision_llm", None)
    if vision is not None and _enabled(_get(vision, "enabled", False)):
        payload["vision_llm"] = _sanitise_mapping(_as_mapping(vision))

    return payload


def build_index_fingerprint(
    settings: Any,
    benchmark_provider: str = "financebench",
    dataset_version: str = "open_source",
) -> str:
    """Return the SHA-256 digest of the canonical index payload."""

    payload = build_index_payload(settings, benchmark_provider, dataset_version)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def collection_name(provider: str, fingerprint: str) -> str:
    """Return a lowercase collection name using the first 12 digest characters."""

    safe_provider = _UNSAFE_NAME_CHARS.sub("_", str(provider).strip().lower())
    safe_provider = safe_provider.strip("_-") or "benchmark"
    hash_prefix = str(fingerprint)[:12].lower()
    safe_hash = _UNSAFE_NAME_CHARS.sub("_", hash_prefix).strip("_-")
    if not safe_hash:
        raise ValueError("fingerprint must contain at least one safe character")
    return f"{safe_provider}__{safe_hash}"
