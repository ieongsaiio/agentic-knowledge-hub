#!/usr/bin/env python
"""Safely clear generated knowledge-hub data.

Examples:
    python scripts/clear_data.py --storage
    python scripts/clear_data.py --logs --yes
    python scripts/clear_data.py --all --dry-run
    python scripts/clear_data.py --all --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class ClearTarget:
    """One generated file or directory selected for removal."""

    label: str
    path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clear generated knowledge-hub storage, JSONL logs, or evaluation "
            "outputs. Benchmark source data is always preserved."
        )
    )
    parser.add_argument(
        "--storage",
        action="store_true",
        help="Clear Chroma, BM25, ingestion history, image index, and images.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Clear JSONL files under logs plus the configured trace file.",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Clear generated evaluation reports.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear storage, logs, and evaluation outputs.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Configuration file used to locate configured outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without changing the filesystem.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args(argv)

    if not (args.storage or args.logs or args.evaluation or args.all):
        parser.error(
            "select at least one scope: --storage, --logs, --evaluation, or --all"
        )
    return args


def load_config(config_path: Path) -> dict[str, Any]:
    """Load only path-related YAML data without initializing providers."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return data


def configured_value(
    config: dict[str, Any],
    section: str,
    key: str,
    default: str,
) -> str:
    section_value = config.get(section, {})
    if not isinstance(section_value, dict):
        return default
    value = section_value.get(key, default)
    return value if isinstance(value, str) and value.strip() else default


def resolve_output_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def validate_target(root: Path, target: Path) -> None:
    """Reject paths outside the project or capable of deleting benchmarks."""
    root = root.resolve(strict=False)
    target = target.resolve(strict=False)
    benchmark_root = (root / "data" / "benchmarks").resolve(strict=False)

    if target == root or root not in target.parents:
        raise ValueError(f"Refusing to clear path outside project root: {target}")
    if (
        target == benchmark_root
        or target in benchmark_root.parents
        or benchmark_root in target.parents
    ):
        raise ValueError(
            f"Refusing to clear path that contains benchmark source data: {target}"
        )


def database_targets(root: Path, filename: str, label: str) -> list[ClearTarget]:
    database = (root / "data" / "db" / filename).resolve(strict=False)
    return [
        ClearTarget(label, database),
        ClearTarget(f"{label} WAL", Path(f"{database}-wal")),
        ClearTarget(f"{label} SHM", Path(f"{database}-shm")),
    ]


def build_targets(
    root: Path,
    config: dict[str, Any],
    *,
    storage: bool,
    logs: bool,
    evaluation: bool,
) -> list[ClearTarget]:
    """Build a deduplicated and validated removal plan."""
    root = root.resolve(strict=False)
    targets: list[ClearTarget] = []

    if storage:
        chroma_path = configured_value(
            config,
            "vector_store",
            "persist_directory",
            "./data/db/chroma",
        )
        targets.extend(
            [
                ClearTarget(
                    "Chroma vector store",
                    resolve_output_path(root, chroma_path),
                ),
                ClearTarget("BM25 indexes", root / "data" / "db" / "bm25"),
                ClearTarget("Extracted images", root / "data" / "images"),
            ]
        )
        targets.extend(
            database_targets(root, "ingestion_history.db", "Ingestion history")
        )
        targets.extend(database_targets(root, "image_index.db", "Image index"))

    if logs:
        logs_root = (root / "logs").resolve(strict=False)
        if logs_root.is_dir():
            targets.extend(
                ClearTarget("JSONL log", path)
                for path in logs_root.rglob("*.jsonl")
            )
        trace_path = configured_value(
            config,
            "observability",
            "trace_file",
            "./logs/traces.jsonl",
        )
        targets.append(
            ClearTarget(
                "Configured trace log",
                resolve_output_path(root, trace_path),
            )
        )

    if evaluation:
        output = "./data/evaluation"
        evaluation_section = config.get("evaluation", {})
        if isinstance(evaluation_section, dict):
            output_section = evaluation_section.get("output", {})
            if isinstance(output_section, dict):
                candidate = output_section.get("directory", output)
                if isinstance(candidate, str) and candidate.strip():
                    output = candidate
        targets.append(
            ClearTarget(
                "Evaluation outputs",
                resolve_output_path(root, output),
            )
        )

    deduplicated: dict[Path, ClearTarget] = {}
    for target in targets:
        resolved = target.path.resolve(strict=False)
        validate_target(root, resolved)
        deduplicated.setdefault(resolved, ClearTarget(target.label, resolved))
    return list(deduplicated.values())


def clear_targets(
    targets: Sequence[ClearTarget],
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Remove selected targets and return (removed, already_absent)."""
    removed = 0
    absent = 0
    for target in targets:
        if not target.path.exists():
            print(f"[ABSENT] {target.label}: {target.path}")
            absent += 1
            continue
        if dry_run:
            print(f"[DRY-RUN] {target.label}: {target.path}")
            continue
        if target.path.is_dir():
            shutil.rmtree(target.path)
        else:
            target.path.unlink()
        print(f"[CLEARED] {target.label}: {target.path}")
        removed += 1
    return removed, absent


def confirm(targets: Sequence[ClearTarget]) -> bool:
    print("The following generated paths are selected:")
    for target in targets:
        print(f"  - {target.label}: {target.path}")
    answer = input("Type 'clear' to continue: ").strip().lower()
    return answer == "clear"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = PROJECT_ROOT.resolve(strict=False)

    try:
        config = load_config(Path(args.config).resolve(strict=False))
        select_all = args.all
        targets = build_targets(
            root,
            config,
            storage=args.storage or select_all,
            logs=args.logs or select_all,
            evaluation=args.evaluation or select_all,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not args.dry_run and not args.yes and not confirm(targets):
        print("[CANCELLED] Nothing was removed.")
        return 1

    removed, absent = clear_targets(targets, dry_run=args.dry_run)
    action = "would clear" if args.dry_run else "cleared"
    print(
        f"[SUMMARY] {action}={removed if not args.dry_run else len(targets) - absent}, "
        f"already_absent={absent}"
    )
    print("[PRESERVED] data/benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
