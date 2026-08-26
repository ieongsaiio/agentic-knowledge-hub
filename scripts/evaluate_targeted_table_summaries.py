"""Evaluate oracle-selected FinanceBench table summaries with dense retrieval only.

This is a diagnostic experiment, not a benchmark score. Reference evidence is
used only to select the source tables. It is never included in the summarizer
prompt or the retrieval corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from src.core.settings import load_settings, resolve_path
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.splitter.structured_markdown_splitter import _LLMTableSummarizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data/benchmarks/financebench/data/financebench_open_source.jsonl"
DEFAULT_TARGETS = (
    PROJECT_ROOT
    / "config/evaluation/financebench_30_table_targets.v1.json"
)
DEFAULT_FACTS = PROJECT_ROOT / "config/evaluation/financebench_30_atomic_facts.v1.jsonl"
DEFAULT_CHUNKS = PROJECT_ROOT / "tmp/financebench_30_chunking_current/chunks.jsonl"
DEFAULT_CACHE = PROJECT_ROOT / "data/cache/table_summaries/financebench_30_targeted_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "data/analysis/targeted_table_summary_dense_report.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "data/analysis/targeted_table_summary_dense_report.md"
DEFAULT_ALIAS_COLLECTION = "financebench_30_targeted_table_summaries"

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&'’-]*")
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?%?"
    r"(?![A-Za-z0-9])"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _words(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(html.unescape(text))
        if len(token) > 2
    }


def _numbers(text: str) -> set[str]:
    values: set[str] = set()
    normalized_text = re.sub(r"\bFY(?=\d)", "", html.unescape(text), flags=re.I)
    for match in _NUMBER.findall(normalized_text):
        normalized = match.replace(",", "").replace(" ", "")
        suffix = "%" if normalized.endswith("%") else ""
        numeric = normalized[:-1] if suffix else normalized
        if numeric.startswith("(") and numeric.endswith(")"):
            numeric = numeric[1:-1]
        values.add(numeric.removeprefix("-").casefold() + suffix)
    return values


def _coverage(required: set[str], candidate: set[str]) -> float:
    return len(required & candidate) / len(required) if required else 1.0


def _table_source(text: str) -> str:
    start = text.casefold().find("<table")
    end = text.casefold().rfind("</table>")
    if start < 0 or end < start:
        return text
    return text[start : end + len("</table>")]


def _fact_text(annotation: dict[str, Any], evidence_index: int) -> str:
    groups = annotation.get("evidence_groups", [])
    if not 0 < evidence_index <= len(groups):
        return ""
    return "\n".join(
        str(item.get("fact", ""))
        for item in groups[evidence_index - 1].get("evidence_facts", [])
        if item.get("fact")
    )


def select_target_chunk(
    target: dict[str, Any],
    case: dict[str, Any],
    annotation: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Select the strongest current table candidate on the annotated page."""
    evidence_index = int(target["evidence_index"])
    evidence = case["evidence"][evidence_index - 1]
    facts = _fact_text(annotation, evidence_index)
    reference = "\n".join(
        [str(evidence.get("evidence_text", "")), facts, str(case.get("answer", ""))]
    )
    required_numbers = _numbers(facts) or _numbers(reference)
    required_words = _words(reference)
    ranked: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for candidate in candidates:
        candidate_text = "\n".join(
            [
                str(candidate.get("text", "")),
                str(candidate.get("dense_index_text", "")),
            ]
        )
        number_coverage = _coverage(required_numbers, _numbers(candidate_text))
        word_coverage = _coverage(required_words, _words(candidate_text))
        old_id_match = float(
            target.get("parent_table_id")
            in {
                candidate.get("metadata", {}).get("table_group_id"),
                candidate.get("metadata", {}).get("parsed_block_id"),
            }
        )
        ranked.append(((old_id_match, number_coverage, word_coverage), candidate))
    if not ranked:
        raise RuntimeError(
            f"No table candidate for {target['case_id']} evidence "
            f"{target['evidence_index']} on {target['document_name']} "
            f"page {target['page_number']}"
        )
    score, selected = max(ranked, key=lambda item: item[0])
    return selected, {
        "legacy_id_match": score[0],
        "number_coverage": score[1],
        "word_coverage": score[2],
        "candidate_count": float(len(candidates)),
    }


def summary_cache_key(
    *, table_text: str, prompt: str, model_config: dict[str, Any]
) -> str:
    payload = json.dumps(
        {"table_text": table_text, "prompt": prompt, "model": model_config},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge_dense_results(
    baseline: list[tuple[str, float]], aliases: list[tuple[str, float]]
) -> list[tuple[str, float, str]]:
    """Merge cosine-distance rankings from two collections."""
    merged = [(*item, "baseline") for item in baseline]
    merged.extend((*item, "summary_alias") for item in aliases)
    return sorted(merged, key=lambda item: (item[1], item[0]))


def deduplicate_dense_groups(
    merged: list[tuple[str, float, str]],
    group_by_id: dict[str, str],
) -> list[tuple[str, float, str, str]]:
    """Keep the highest-ranked retrieval entry for each table group."""
    seen: set[str] = set()
    unique: list[tuple[str, float, str, str]] = []
    for item_id, distance, source in merged:
        group_id = group_by_id.get(item_id) or item_id
        if group_id in seen:
            continue
        seen.add(group_id)
        unique.append((item_id, distance, source, group_id))
    return unique


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _rank_metrics(ranks: list[int | None], cutoffs: tuple[int, ...]) -> dict[str, Any]:
    total = len(ranks)
    numeric = [float(rank) for rank in ranks if rank is not None]
    result: dict[str, Any] = {
        "count": total,
        "found": len(numeric),
        "median_rank": statistics.median(numeric) if numeric else None,
        "p90_rank": _percentile(numeric, 0.9),
    }
    for cutoff in cutoffs:
        result[f"hit@{cutoff}"] = (
            sum(rank is not None and rank <= cutoff for rank in ranks) / total
            if total
            else 0.0
        )
        result[f"mrr@{cutoff}"] = (
            sum(1 / rank for rank in ranks if rank is not None and rank <= cutoff)
            / total
            if total
            else 0.0
        )
    return result


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    return [row for row in _jsonl(path) if row.get("chunk_type") == "table_group"]


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("entries", {})
    return payload


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summarize_record(summarizer: _LLMTableSummarizer, record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    header_path = metadata.get("header_path") or []
    section_path = (
        " > ".join(str(item) for item in header_path)
        if isinstance(header_path, list)
        else str(header_path)
    )
    page_start, page_end = metadata.get("page_start"), metadata.get("page_end")
    page_range = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    return summarizer.summarize(
        _table_source(str(record["text"])),
        table_title=str(metadata.get("table_title") or "") or None,
        footnotes=[
            item for item in metadata.get("vision_footnotes", []) if isinstance(item, str)
        ],
        previous_context=str(metadata.get("previous_context") or "") or None,
        next_context=str(metadata.get("next_context") or "") or None,
        document_name=str(record.get("document_name") or "") or None,
        section_path=section_path or None,
        page_range=page_range if page_start is not None else None,
        table_units=[item for item in metadata.get("units", []) if isinstance(item, dict)],
        table_unit_count=max(1, int(metadata.get("unit_count") or 1)),
    )


def _query_pairs(collection: Any, vectors: list[list[float]], top_k: int) -> list[list[tuple[str, float]]]:
    result = collection.query(
        query_embeddings=vectors,
        n_results=min(top_k, collection.count()),
        include=["distances"],
    )
    return [
        [(str(chunk_id), float(distance)) for chunk_id, distance in zip(ids, distances)]
        for ids, distances in zip(result["ids"], result["distances"])
    ]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Targeted Table Summary Dense Retrieval",
        "",
        "> Diagnostic oracle experiment: references select target tables only; references are not sent to the summarizer or indexed.",
        "",
        "## Summary",
        "",
        f"- Direct table evidence records: **{report['selection']['target_records']}**",
        f"- Unique current table groups summarized: **{report['selection']['unique_table_groups']}**",
        f"- Cache hits/new calls: **{report['summary_cache']['hits']} / {report['summary_cache']['misses']}**",
        f"- Dense model: **{report['embedding']['model']} ({report['embedding']['dimensions']}d)**",
        "- Sparse/RRF/Reranker: **disabled**",
        "",
        "## Metrics",
        "",
        "| Ranking | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Median rank | P90 rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("Summary aliases only", "summary_only"),
        ("Summary alias itself in full corpus", "mixed_alias"),
        ("Enhanced group (raw or summary, deduplicated)", "enhanced_group"),
        ("Raw parent tables in baseline", "raw_parent"),
    ]:
        metric = report["metrics"][key]
        lines.append(
            f"| {label} | {metric['hit@1']:.3f} | {metric['hit@3']:.3f} | "
            f"{metric['hit@5']:.3f} | {metric['hit@10']:.3f} | {metric['mrr@10']:.3f} | "
            f"{metric['median_rank'] or 'N/A'} | {metric['p90_rank'] or 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "## Per Evidence",
            "",
            "| Case | Evidence | Document/page | Table group | Summary-only | Alias in corpus | Enhanced group | Raw parent |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["results"]:
        lines.append(
            f"| `{row['case_id']}` | {row['evidence_index']} | "
            f"{row['document_name']} / {row['page_number']} | `{row['table_group_id']}` | "
            f"{row['summary_only_rank'] or '>limit'} | {row['mixed_alias_rank'] or '>limit'} | "
            f"{row['enhanced_group_rank'] or '>limit'} | "
            f"{row['raw_parent_rank'] or '>limit'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--alias-collection", default=DEFAULT_ALIAS_COLLECTION)
    parser.add_argument("--baseline-top-k", type=int, default=100)
    parser.add_argument("--summary-workers", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings(resolve_path(args.config))
    cases = {row["financebench_id"]: row for row in _jsonl(resolve_path(args.cases))}
    facts = {row["case_id"]: row for row in _jsonl(resolve_path(args.facts))}
    targets = json.loads(resolve_path(args.targets).read_text(encoding="utf-8"))["targets"]
    table_chunks = _load_chunks(resolve_path(args.chunks))
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in table_chunks:
        by_doc[str(chunk["document_name"])].append(chunk)

    selections: list[dict[str, Any]] = []
    selected_by_group: dict[str, dict[str, Any]] = {}
    for target in targets:
        page = int(target["page_number"])
        candidates = [
            chunk
            for chunk in by_doc[str(target["document_name"])]
            if int(chunk["metadata"].get("page_start", -1)) <= page
            <= int(chunk["metadata"].get("page_end", -1))
        ]
        selected, selection_score = select_target_chunk(
            target, cases[target["case_id"]], facts[target["case_id"]], candidates
        )
        group_id = str(selected["metadata"]["table_group_id"])
        selected_by_group[group_id] = selected
        selections.append({**target, "table_group_id": group_id, "selection": selection_score})

    summary_config = dict(settings.ingestion.structured_chunking.get("table_summary") or {})
    llm_config = dict(summary_config.get("llm") or {})
    prompt = resolve_path(summary_config.get("prompt_path", "config/prompts/table_summary.txt")).read_text(encoding="utf-8")
    model_identity = {
        "provider": llm_config.get("provider", settings.llm.provider),
        "api_mode": llm_config.get("api_mode", settings.llm.api_mode),
        "model": llm_config.get("model", settings.llm.model),
        "temperature": llm_config.get("temperature", settings.llm.temperature),
        "prompt_version": summary_config.get("prompt_version", "v1"),
    }
    cache_path = resolve_path(args.cache)
    cache = _load_cache(cache_path)
    summaries: dict[str, str] = {}
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    selected_cache_keys: set[str] = set()
    cache_hits = 0
    for group_id, record in selected_by_group.items():
        key = summary_cache_key(table_text=_table_source(record["text"]), prompt=prompt, model_config=model_identity)
        selected_cache_keys.add(key)
        entry = cache["entries"].get(key)
        if entry and entry.get("summary"):
            summaries[group_id] = str(entry["summary"])
            cache_hits += 1
        else:
            pending[group_id] = (key, record)

    cache["entries"] = {
        key: value
        for key, value in cache["entries"].items()
        if key in selected_cache_keys
    }
    _write_cache(cache_path, cache)

    if pending:
        summarizer = _LLMTableSummarizer(settings)
        with ThreadPoolExecutor(max_workers=min(args.summary_workers, len(pending))) as executor:
            futures = {
                executor.submit(_summarize_record, summarizer, record): (group_id, key, record)
                for group_id, (key, record) in pending.items()
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                group_id, key, record = futures[future]
                started = time.perf_counter()
                summary = future.result()
                summaries[group_id] = summary
                cache["entries"][key] = {
                    "table_group_id": group_id,
                    "document_name": record["document_name"],
                    "page_start": record["metadata"].get("page_start"),
                    "page_end": record["metadata"].get("page_end"),
                    "source_sha256": hashlib.sha256(_table_source(record["text"]).encode("utf-8")).hexdigest(),
                    "model": model_identity,
                    "summary": summary,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                _write_cache(cache_path, cache)
                print(f"summarized {completed}/{len(pending)}: {group_id} ({time.perf_counter() - started:.3f}s collect)")

    client = chromadb.PersistentClient(path=str(resolve_path(settings.vector_store.persist_directory)))
    baseline = client.get_collection(settings.vector_store.collection_name)
    baseline_metadata = baseline.get(include=["metadatas"])
    baseline_group_by_id = {
        str(chunk_id): str(metadata.get("table_group_id") or "")
        for chunk_id, metadata in zip(
            baseline_metadata["ids"], baseline_metadata["metadatas"]
        )
    }
    try:
        client.delete_collection(args.alias_collection)
    except Exception:
        pass
    aliases = client.create_collection(args.alias_collection, metadata={"hnsw:space": "cosine", "diagnostic": True})
    group_ids = sorted(summaries)
    embedding = EmbeddingFactory.create(settings)
    summary_vectors = embedding.embed([summaries[group_id] for group_id in group_ids])
    aliases.add(
        ids=[f"{group_id}_summary" for group_id in group_ids],
        embeddings=summary_vectors,
        documents=[summaries[group_id] for group_id in group_ids],
        metadatas=[{"table_group_id": group_id, "chunk_role": "table_summary"} for group_id in group_ids],
    )

    case_ids = list(dict.fromkeys(selection["case_id"] for selection in selections))
    query_vectors = embedding.embed([cases[case_id]["question"] for case_id in case_ids])
    baseline_results = _query_pairs(baseline, query_vectors, args.baseline_top_k)
    alias_results = _query_pairs(aliases, query_vectors, len(group_ids))
    query_index = {case_id: index for index, case_id in enumerate(case_ids)}
    alias_group_by_id = {
        f"{group_id}_summary": group_id for group_id in group_ids
    }
    merged_group_by_id = {**baseline_group_by_id, **alias_group_by_id}
    rows: list[dict[str, Any]] = []
    for selection in selections:
        index = query_index[selection["case_id"]]
        target_group = selection["table_group_id"]
        alias_id = f"{target_group}_summary"
        alias_rank = next((rank for rank, (item_id, _) in enumerate(alias_results[index], 1) if item_id == alias_id), None)
        raw_parent_rank = next(
            (
                rank
                for rank, (item_id, _) in enumerate(baseline_results[index], 1)
                if baseline_group_by_id.get(item_id) == target_group
            ),
            None,
        )
        mixed = merge_dense_results(baseline_results[index], alias_results[index])
        mixed_alias_rank = next(
            (rank for rank, (item_id, _, _) in enumerate(mixed, 1) if item_id == alias_id),
            None,
        )
        deduplicated = deduplicate_dense_groups(mixed, merged_group_by_id)
        enhanced_group_rank = next(
            (
                rank
                for rank, (_, _, _, group_id) in enumerate(deduplicated, 1)
                if group_id == target_group
            ),
            None,
        )
        rows.append(
            {
                **selection,
                "summary_only_rank": alias_rank,
                "mixed_alias_rank": mixed_alias_rank,
                "enhanced_group_rank": enhanced_group_rank,
                "raw_parent_rank": raw_parent_rank,
                "summary_chars": len(summaries[target_group]),
            }
        )

    cutoffs = (1, 3, 5, 10, 20, 30)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "oracle table selection; generic summaries; dense-only retrieval",
        "selection": {"target_records": len(selections), "question_count": len(case_ids), "unique_table_groups": len(group_ids)},
        "summary_cache": {"path": str(cache_path), "hits": cache_hits, "misses": len(pending), "entries": len(cache["entries"])},
        "embedding": {"provider": settings.embedding.provider, "model": settings.embedding.model, "dimensions": settings.embedding.dimensions},
        "retrieval": {"baseline_collection": baseline.name, "alias_collection": aliases.name, "baseline_top_k": args.baseline_top_k, "sparse": False, "reranker": False},
        "metrics": {
            "summary_only": _rank_metrics([row["summary_only_rank"] for row in rows], cutoffs),
            "mixed_alias": _rank_metrics([row["mixed_alias_rank"] for row in rows], cutoffs),
            "enhanced_group": _rank_metrics([row["enhanced_group_rank"] for row in rows], cutoffs),
            "raw_parent": _rank_metrics([row["raw_parent_rank"] for row in rows], cutoffs),
        },
        "results": rows,
    }
    report_path, markdown_path = resolve_path(args.report), resolve_path(args.markdown)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"cache={cache_path}\nreport={report_path}\nmarkdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
