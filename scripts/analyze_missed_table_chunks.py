"""Report reference table chunks that did not reach a retrieval run's top-k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    args = parser.parse_args()

    root = Path.cwd()
    client = chromadb.PersistentClient(path=str(root / "data/db/chroma"))
    collection = client.get_collection(args.collection)
    all_records = collection.get(include=["metadatas"])
    metadata_by_id = dict(zip(all_records["ids"], all_records["metadatas"]))
    table_ids = {
        chunk_id
        for chunk_id, metadata in metadata_by_id.items()
        if "table" in str(metadata.get("unit_types", "")).lower()
    }

    final_by_case = {
        item["case_id"]: set(item["retrieved_chunk_ids"])
        for item in _load_jsonl(args.evaluation)
    }
    reference_by_case = {
        item["case_id"]: item for item in _load_jsonl(args.judgements)
    }

    missed: list[dict[str, Any]] = []
    for case in _load_jsonl(args.coverage):
        final_ids = final_by_case.get(case["case_id"], set())
        for evidence_index, evidence in enumerate(case["evidences"], start=1):
            candidates = [
                (float(candidate["coverage"]), candidate["id"])
                for candidate in evidence["top_chunks"]
                if candidate["id"] in table_ids
            ]
            if not candidates:
                continue
            coverage, chunk_id = max(candidates)
            if coverage >= args.minimum_coverage and chunk_id not in final_ids:
                missed.append(
                    {
                        "case_id": case["case_id"],
                        "query": case["query"],
                        "evidence_index": evidence_index,
                        "coverage": coverage,
                        "chunk_id": chunk_id,
                    }
                )

    records = collection.get(
        ids=[item["chunk_id"] for item in missed],
        include=["documents", "metadatas"],
    )
    record_by_id = {
        chunk_id: (document, metadata)
        for chunk_id, document, metadata in zip(
            records["ids"], records["documents"], records["metadatas"]
        )
    }

    lines = [
        "# Missed Table Chunks in the Current 30-Question Evaluation",
        "",
        "## Method",
        "",
        f"- Collection: `{args.collection}`",
        f"- Total chunks: **{collection.count():,}**",
        f"- Table chunks: **{len(table_ids):,}**",
        f"- Minimum normalized token coverage: `{args.minimum_coverage}`",
        "- Missed means the mapped table chunk did not enter the final Top 10.",
        f"- Missed table evidences: **{len(missed)}**",
        "",
        "> Token coverage maps differently formatted FinanceBench evidence back to "
        "the indexed source. It is a diagnostic mapping, not a semantic judgement.",
        "",
        "## Summary",
        "",
        "| Case | Evidence | Coverage | Page | Chars | Chunk ID |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for item in missed:
        document, metadata = record_by_id[item["chunk_id"]]
        lines.append(
            f"| `{item['case_id']}` | {item['evidence_index']} | "
            f"{item['coverage']:.3f} | "
            f"{metadata.get('page_start')}-{metadata.get('page_end')} | "
            f"{len(document):,} | `{item['chunk_id']}` |"
        )

    for number, item in enumerate(missed, start=1):
        document, metadata = record_by_id[item["chunk_id"]]
        reference = reference_by_case.get(item["case_id"], {})
        evidences = reference.get("evidences", [])
        evidence = (
            evidences[item["evidence_index"] - 1]
            if len(evidences) >= item["evidence_index"]
            else {}
        )
        lines.extend(
            [
                "",
                f"## {number}. {item['case_id']} / Evidence {item['evidence_index']}",
                "",
                f"**Question:** {item['query']}",
                "",
                "**Reference document/page:** "
                f"`{evidence.get('document_name', '')}`, "
                f"page {evidence.get('page_number', '')}",
                "",
                "**Reference evidence:**",
                "",
                "````text",
                str(evidence.get("text", "")).strip(),
                "````",
                "",
                f"**Mapped Chunk:** `{item['chunk_id']}`",
                "",
                f"- Coverage: `{item['coverage']:.3f}`",
                f"- Page: `{metadata.get('page_start')}-{metadata.get('page_end')}`",
                f"- Characters: `{len(document):,}`",
                f"- Unit types: `{metadata.get('unit_types')}`",
                f"- Source: `{metadata.get('source_path')}`",
                "",
                "**Original content stored in Chroma `documents`:**",
                "",
                "````html",
                document,
                "````",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"table_chunks={len(table_ids)} missed={len(missed)} "
        f"report={args.output}"
    )


if __name__ == "__main__":
    main()
