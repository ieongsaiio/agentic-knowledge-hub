#!/usr/bin/env python
"""Ingest exactly the source documents listed in an exported JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import load_settings  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--config", default="config/settings.yaml", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = load_settings(args.config)
    pipeline = IngestionPipeline(
        settings,
        collection=args.collection,
        force=args.force,
    )
    started = time.perf_counter()
    results = []
    try:
        for index, row in enumerate(rows, start=1):
            print(
                f"[MANIFEST {index}/{len(rows)}] {row['document_name']}",
                flush=True,
            )
            result = pipeline.run(str(row["source_path"]))
            results.append(result)
            print(
                f"[RESULT] success={result.success} chunks={result.chunk_count} "
                f"error={result.error}",
                flush=True,
            )
    finally:
        pipeline.close()

    elapsed = time.perf_counter() - started
    print(
        f"[FINAL] documents={len(results)} "
        f"success={sum(result.success for result in results)} "
        f"chunks={sum(result.chunk_count for result in results)} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return 0 if len(results) == len(rows) and all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
