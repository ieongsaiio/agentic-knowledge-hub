"""Validate fixed FinanceBench atomic-fact annotations against their source cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.libs.benchmark.financebench_benchmark import FinanceBenchBenchmark
from src.observability.evaluation.atomic_fact_annotations import (
    load_atomic_fact_annotations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "config" / "evaluation" / "financebench_30_atomic_facts.v1.jsonl"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-dir", default="./data/benchmarks/financebench")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark = FinanceBenchBenchmark(
        {
            "data_dir": args.data_dir,
            "auto_download": False,
            "sample_size": args.sample_size,
            "seed": args.seed,
        }
    )
    cases = benchmark.load_cases()
    annotations = load_atomic_fact_annotations(args.annotations, cases=cases)
    manifest_path = args.manifest or args.annotations.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": "1.0",
        "provider": "financebench",
        "sample_size": len(cases),
        "seed": args.seed,
        "annotation_sha256": _sha256_file(args.annotations),
        "dataset_sha256": _sha256_file(benchmark.dataset_path),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"manifest {field} mismatch: {manifest.get(field)!r} != {expected!r}"
            )
    fact_count = sum(
        len(group.evidence_facts)
        for annotation in annotations
        for group in annotation.evidence_groups
    )
    evidence_count = sum(len(annotation.evidence_groups) for annotation in annotations)
    reasoning_count = sum(len(annotation.reasoning_requirements) for annotation in annotations)
    print(
        f"valid annotations: cases={len(annotations)}, evidences={evidence_count}, "
        f"facts={fact_count}, reasoning_requirements={reasoning_count}"
    )


if __name__ == "__main__":
    main()
