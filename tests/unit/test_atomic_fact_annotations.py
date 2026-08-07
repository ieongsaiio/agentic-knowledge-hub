from __future__ import annotations

import hashlib
import json

import pytest

from src.libs.benchmark.base_benchmark import BenchmarkCase, BenchmarkEvidence
from src.observability.evaluation.atomic_fact_annotations import (
    case_signature,
    load_atomic_fact_annotations,
)


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="case-1",
        query="What was revenue?",
        reference_answer="$10 million",
        evidences=[BenchmarkEvidence("report", 3, "Revenue was $10 million.")],
        metadata={},
    )


def _row(case: BenchmarkCase) -> dict:
    evidence_text = case.evidences[0].text
    return {
        "schema_version": "1.0",
        "sample_index": 0,
        "case_id": case.case_id,
        "case_signature_sha256": case_signature(case),
        "evidence_groups": [
            {
                "evidence_id": "e1",
                "source_evidence_sha256": hashlib.sha256(evidence_text.encode()).hexdigest(),
                "evidence_facts": [{"fact_id": "e1_f1", "fact": "Revenue was $10 million."}],
            }
        ],
        "reasoning_requirements": [],
        "annotation_notes": [],
    }


def test_loads_and_verifies_exact_case(tmp_path) -> None:
    case = _case()
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(_row(case)) + "\n", encoding="utf-8")

    annotations = load_atomic_fact_annotations(path, cases=[case])

    assert annotations[0].evidence_groups[0].evidence_facts[0].fact_id == "e1_f1"


def test_rejects_changed_source_evidence(tmp_path) -> None:
    case = _case()
    row = _row(case)
    row["evidence_groups"][0]["source_evidence_sha256"] = "0" * 64
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source evidence hash mismatch"):
        load_atomic_fact_annotations(path, cases=[case])


def test_rejects_non_sequential_fact_ids(tmp_path) -> None:
    case = _case()
    row = _row(case)
    row["evidence_groups"][0]["evidence_facts"][0]["fact_id"] = "fact-99"
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected fact_id e1_f1"):
        load_atomic_fact_annotations(path)
