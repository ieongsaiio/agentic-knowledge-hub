#!/usr/bin/env python
"""Export the best historical result for every distinct benchmark config."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _json_value(value: Any) -> str:
    if value in (None, {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _config_columns(settings: dict[str, Any]) -> dict[str, Any]:
    ingestion = _get(settings, "ingestion", default={}) or {}
    embedding = _get(settings, "embedding", default={}) or {}
    retrieval = _get(settings, "retrieval", default={}) or {}
    rerank = _get(settings, "rerank", default={}) or {}
    rerank_api = rerank.get("api_args") or {}
    vision = _get(settings, "vision_llm", default={}) or {}
    llm = _get(settings, "llm", default={}) or {}
    return {
        "length_unit": ingestion.get("length_unit", "characters"),
        "chunk_size": ingestion.get("chunk_size", ""),
        "chunk_overlap": ingestion.get("chunk_overlap", ""),
        "tokenizer_model": ingestion.get("tokenizer_model", ""),
        "chunk_refiner": _json_value(ingestion.get("chunk_refiner")),
        "metadata_enricher": _json_value(ingestion.get("metadata_enricher")),
        "embedding_provider": embedding.get("provider", ""),
        "embedding_model": embedding.get("model", ""),
        "embedding_dimensions": embedding.get("dimensions", ""),
        "dense_enabled": retrieval.get("enable_dense", True),
        "sparse_enabled": retrieval.get("enable_sparse", True),
        "dense_top_k": retrieval.get("dense_top_k", ""),
        "sparse_top_k": retrieval.get("sparse_top_k", ""),
        "fusion_top_k": retrieval.get("fusion_top_k", ""),
        "rrf_k": retrieval.get("rrf_k", ""),
        "dense_weight": retrieval.get("dense_weight", 0.5),
        "sparse_weight": retrieval.get("sparse_weight", 0.5),
        "query_expansions": _json_value(retrieval.get("query_expansions")),
        "document_routing": _json_value(retrieval.get("document_routing")),
        "rerank_enabled": rerank.get("enabled", False),
        "rerank_provider": rerank.get("provider", ""),
        "rerank_model": rerank_api.get("model") or rerank.get("model", ""),
        "rerank_top_k": rerank.get("top_k", ""),
        "rerank_threshold": rerank_api.get("rerank_threshold", ""),
        "rerank_metadata_fields": _json_value(rerank.get("metadata_fields")),
        "vision_enabled": vision.get("enabled", False),
        "llm_provider": llm.get("provider", ""),
        "llm_model": llm.get("model", ""),
    }


def _signature(columns: dict[str, Any]) -> str:
    canonical = json.dumps(columns, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _score(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(row.get("evidence_hit_rate@5") or -1),
        float(row.get("evidence_mrr@5") or -1),
        str(row.get("created_at") or ""),
    )


def export_history(evaluation_dir: Path, output_path: Path) -> tuple[int, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    summary_count = 0

    for summary_path in sorted(evaluation_dir.rglob("*.summary.json")):
        run_path = summary_path.parent / "run.json"
        if not run_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        experiment = str(summary.get("experiment") or summary_path.name.removesuffix(".summary.json"))
        plan = next(
            (item for item in run.get("plans", []) if item.get("name") == experiment),
            None,
        )
        if not isinstance(plan, dict):
            continue

        config = _config_columns(plan.get("effective_settings") or {})
        signature = _signature(config)
        metrics = summary.get("aggregate_metrics") or {}
        row = {
            "config_id": signature,
            "experiment": experiment,
            "created_at": run.get("created_at", ""),
            "sample_size": summary.get("query_count", run.get("case_count", "")),
            "document_hit_rate@5": metrics.get("document_hit_rate@5", ""),
            "document_mrr@5": metrics.get("document_mrr@5", ""),
            "evidence_hit_rate@5": metrics.get("evidence_hit_rate@5", ""),
            "evidence_mrr@5": metrics.get("evidence_mrr@5", ""),
            "page_hit_rate@5": metrics.get("page_hit_rate@5", ""),
            "total_elapsed_ms": summary.get("total_elapsed_ms", ""),
            **config,
        }
        groups.setdefault(signature, []).append(row)
        summary_count += 1

    selected: list[dict[str, Any]] = []
    for rows in groups.values():
        best = max(rows, key=_score).copy()
        best["run_count"] = len(rows)
        best["experiment_aliases"] = "|".join(sorted({str(row["experiment"]) for row in rows}))
        selected.append(best)
    selected.sort(key=_score, reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if selected:
        fieldnames = list(selected[0].keys())
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)
    return summary_count, len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default="data/evaluation")
    parser.add_argument("--output", default="benchmark_history_comparison.csv")
    args = parser.parse_args()
    summary_count, config_count = export_history(
        (PROJECT_ROOT / args.evaluation_dir).resolve(),
        (PROJECT_ROOT / args.output).resolve(),
    )
    print(f"Exported {config_count} distinct configs from {summary_count} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
