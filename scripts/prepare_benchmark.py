#!/usr/bin/env python
"""Prepare the benchmark dataset configured for this project.

Exit codes:
    0 - Success
    1 - Benchmark preparation failure
    2 - Configuration failure
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

# Match the other project scripts when writing to a Windows console.
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download, cache, and load the configured benchmark dataset."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Configuration file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--sample-size",
        type=_non_negative_int,
        help="Override evaluation.benchmark.sample_size for this run only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Accepted for compatibility; preparation currently retains the "
            "provider's idempotent cache behavior."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the preparation summary as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the configured benchmark and print a non-sensitive summary."""
    args = parse_args(argv)

    try:
        from src.core.settings import load_settings
        from src.libs.benchmark.benchmark_factory import BenchmarkFactory

        settings = load_settings(args.config)
        effective_settings = settings
        if args.sample_size is not None:
            benchmark_settings = settings.evaluation.benchmark
            if benchmark_settings is None:
                raise ValueError("evaluation.benchmark is not configured")
            effective_settings = replace(
                settings,
                evaluation=replace(
                    settings.evaluation,
                    benchmark=replace(
                        benchmark_settings,
                        sample_size=args.sample_size,
                    ),
                ),
            )

        benchmark = BenchmarkFactory.create(effective_settings)
        configured_benchmark = effective_settings.evaluation.benchmark
        if configured_benchmark is None:
            raise ValueError("evaluation.benchmark is not configured")
        provider = configured_benchmark.provider
    except Exception:
        print(
            f"Configuration error: could not load or validate {args.config}",
            file=sys.stderr,
        )
        return 2

    try:
        cases = benchmark.prepare()
        pdf_count = sum(
            1
            for path in benchmark.pdf_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
    except Exception:
        print("Benchmark preparation failed.", file=sys.stderr)
        return 1

    summary = {
        "provider": provider,
        "data_dir": str(benchmark.data_dir),
        "pdf_count": pdf_count,
        "case_count": len(cases),
        "first_case_id": cases[0].case_id if cases else None,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(f"Provider: {summary['provider']}")
        print(f"Data dir: {summary['data_dir']}")
        print(f"PDF count: {summary['pdf_count']}")
        print(f"Case count: {summary['case_count']}")
        print(f"First case ID: {summary['first_case_id']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
