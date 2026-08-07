"""Versioned, deterministic atomic-fact annotations for retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.libs.benchmark.base_benchmark import BenchmarkCase

SCHEMA_VERSION = "1.0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def case_signature(case: BenchmarkCase) -> str:
    """Hash the immutable benchmark inputs used to create one annotation."""
    payload = {
        "case_id": case.case_id,
        "question": case.query,
        "reference_answer": case.reference_answer,
        "evidences": [evidence.to_dict() for evidence in case.evidences],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)


@dataclass(frozen=True)
class AtomicFact:
    fact_id: str
    fact: str


@dataclass(frozen=True)
class EvidenceFactGroup:
    evidence_id: str
    source_evidence_sha256: str
    evidence_facts: tuple[AtomicFact, ...]


@dataclass(frozen=True)
class ReasoningRequirement:
    reasoning_id: str
    requirement: str


@dataclass(frozen=True)
class AtomicFactAnnotation:
    schema_version: str
    sample_index: int
    case_id: str
    case_signature_sha256: str
    evidence_groups: tuple[EvidenceFactGroup, ...]
    reasoning_requirements: tuple[ReasoningRequirement, ...]
    annotation_notes: tuple[str, ...]


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_annotation(raw: Any, line_number: int) -> AtomicFactAnnotation:
    if not isinstance(raw, dict):
        raise ValueError(f"annotation line {line_number} must be a JSON object")

    schema_version = _required_string(raw.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"annotation line {line_number} uses schema {schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    sample_index = raw.get("sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError(f"annotation line {line_number} has invalid sample_index")

    raw_groups = raw.get("evidence_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(f"annotation line {line_number} requires evidence_groups")
    groups: list[EvidenceFactGroup] = []
    for group_index, raw_group in enumerate(raw_groups, start=1):
        if not isinstance(raw_group, dict):
            raise ValueError(f"line {line_number} evidence group {group_index} must be an object")
        expected_evidence_id = f"e{group_index}"
        evidence_id = _required_string(raw_group.get("evidence_id"), "evidence_id")
        if evidence_id != expected_evidence_id:
            raise ValueError(
                f"line {line_number} expected evidence_id {expected_evidence_id}, got {evidence_id}"
            )
        evidence_hash = _required_string(
            raw_group.get("source_evidence_sha256"), "source_evidence_sha256"
        )
        raw_facts = raw_group.get("evidence_facts")
        if not isinstance(raw_facts, list) or not raw_facts:
            raise ValueError(f"line {line_number} {evidence_id} requires evidence_facts")
        facts: list[AtomicFact] = []
        for fact_index, raw_fact in enumerate(raw_facts, start=1):
            if not isinstance(raw_fact, dict):
                raise ValueError(f"line {line_number} fact {fact_index} must be an object")
            expected_fact_id = f"{evidence_id}_f{fact_index}"
            fact_id = _required_string(raw_fact.get("fact_id"), "fact_id")
            if fact_id != expected_fact_id:
                raise ValueError(
                    f"line {line_number} expected fact_id {expected_fact_id}, got {fact_id}"
                )
            facts.append(AtomicFact(fact_id=fact_id, fact=_required_string(raw_fact.get("fact"), "fact")))
        groups.append(
            EvidenceFactGroup(
                evidence_id=evidence_id,
                source_evidence_sha256=evidence_hash,
                evidence_facts=tuple(facts),
            )
        )

    raw_reasoning = raw.get("reasoning_requirements", [])
    if not isinstance(raw_reasoning, list):
        raise ValueError(f"annotation line {line_number} reasoning_requirements must be a list")
    reasoning: list[ReasoningRequirement] = []
    for reasoning_index, raw_requirement in enumerate(raw_reasoning, start=1):
        if not isinstance(raw_requirement, dict):
            raise ValueError(f"line {line_number} reasoning item {reasoning_index} must be an object")
        expected_reasoning_id = f"r{reasoning_index}"
        reasoning_id = _required_string(raw_requirement.get("reasoning_id"), "reasoning_id")
        if reasoning_id != expected_reasoning_id:
            raise ValueError(
                f"line {line_number} expected reasoning_id {expected_reasoning_id}, got {reasoning_id}"
            )
        reasoning.append(
            ReasoningRequirement(
                reasoning_id=reasoning_id,
                requirement=_required_string(raw_requirement.get("requirement"), "requirement"),
            )
        )

    raw_notes = raw.get("annotation_notes", [])
    if not isinstance(raw_notes, list) or not all(isinstance(note, str) for note in raw_notes):
        raise ValueError(f"annotation line {line_number} annotation_notes must be strings")

    return AtomicFactAnnotation(
        schema_version=schema_version,
        sample_index=sample_index,
        case_id=_required_string(raw.get("case_id"), "case_id"),
        case_signature_sha256=_required_string(
            raw.get("case_signature_sha256"), "case_signature_sha256"
        ),
        evidence_groups=tuple(groups),
        reasoning_requirements=tuple(reasoning),
        annotation_notes=tuple(note.strip() for note in raw_notes if note.strip()),
    )


def load_atomic_fact_annotations(
    path: str | Path,
    *,
    cases: list[BenchmarkCase] | None = None,
) -> list[AtomicFactAnnotation]:
    """Load annotations and optionally verify them against exact benchmark cases."""
    annotation_path = Path(path)
    annotations: list[AtomicFactAnnotation] = []
    with annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line_number, line in enumerate(annotation_file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed annotation JSONL at {annotation_path}:{line_number}: {exc.msg}"
                ) from exc
            annotations.append(_parse_annotation(raw, line_number))

    if not annotations:
        raise ValueError(f"annotation file is empty: {annotation_path}")
    for expected_index, annotation in enumerate(annotations):
        if annotation.sample_index != expected_index:
            raise ValueError(
                f"annotation order mismatch: expected sample_index {expected_index}, "
                f"got {annotation.sample_index}"
            )

    if cases is None:
        return annotations
    if len(annotations) != len(cases):
        raise ValueError(
            f"annotation count {len(annotations)} does not match case count {len(cases)}"
        )
    for annotation, case in zip(annotations, cases, strict=True):
        if annotation.case_id != case.case_id:
            raise ValueError(
                f"sample {annotation.sample_index} case mismatch: "
                f"{annotation.case_id} != {case.case_id}"
            )
        if annotation.case_signature_sha256 != case_signature(case):
            raise ValueError(f"case signature mismatch for {case.case_id}")
        if len(annotation.evidence_groups) != len(case.evidences):
            raise ValueError(f"evidence group count mismatch for {case.case_id}")
        for group, evidence in zip(annotation.evidence_groups, case.evidences, strict=True):
            if group.source_evidence_sha256 != _sha256_text(evidence.text):
                raise ValueError(
                    f"source evidence hash mismatch for {case.case_id}/{group.evidence_id}"
                )
    return annotations
