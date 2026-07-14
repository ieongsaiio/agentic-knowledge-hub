"""Unit tests for deterministic, index-scoped evaluation fingerprints."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from src.core.settings import Settings
from src.observability.evaluation.index_fingerprint import (
    build_index_fingerprint,
    build_index_payload,
    collection_name,
)


def _settings() -> Settings:
    return Settings.from_dict(
        {
            "llm": {
                "provider": "azure_openai",
                "model": "chat-model-v1",
                "temperature": 0.1,
                "max_tokens": 512,
                "api_key": "llm-secret-key",
                "api_version": "2025-01-01",
                "azure_endpoint": "https://llm.secret.invalid",
                "deployment_name": "chat-deployment",
                "base_url": "https://llm-base.secret.invalid",
                "extra_chat_configs": {"seed": 7},
            },
            "embedding": {
                "provider": "azure_openai",
                "model": "embedding-model-v1",
                "dimensions": 1536,
                "api_key": "embedding-secret-key",
                "api_version": "2025-01-01",
                "azure_endpoint": "https://embedding.secret.invalid",
                "deployment_name": "embedding-deployment",
                "base_url": "https://embedding-base.secret.invalid",
            },
            "vector_store": {
                "provider": "chroma",
                "persist_directory": "C:/private/vector-store",
                "collection_name": "private-collection",
            },
            "retrieval": {
                "dense_top_k": 20,
                "sparse_top_k": 15,
                "fusion_top_k": 10,
                "rrf_k": 60,
            },
            "rerank": {
                "enabled": True,
                "provider": "cohere",
                "model": "rerank-model-v1",
                "top_k": 5,
                "api_args": {"api_token": "rerank-secret-token"},
            },
            "evaluation": {
                "enabled": True,
                "provider": "custom",
                "metrics": ["hit_rate", "mrr"],
                "backends": ["local"],
                "benchmark": {
                    "provider": "financebench",
                    "source_url": "https://benchmark.secret.invalid/data.json",
                    "data_dir": "C:/private/benchmark",
                    "split": "open_source",
                    "auto_download": False,
                    "sample_size": 25,
                    "seed": 42,
                },
                "experiments": [
                    {
                        "name": "baseline",
                        "enabled": True,
                        "overrides": {"retrieval": {"fusion_top_k": 8}},
                    }
                ],
                "output": {
                    "directory": "C:/private/evaluation",
                    "save_per_query": True,
                    "formats": ["json", "csv"],
                },
            },
            "observability": {
                "log_level": "INFO",
                "trace_enabled": True,
                "trace_file": "C:/private/traces.jsonl",
                "structured_logging": True,
            },
            "ingestion": {
                "chunk_size": 512,
                "chunk_overlap": 64,
                "splitter": "recursive",
                "batch_size": 16,
                "chunk_refiner": {
                    "use_llm": False,
                    "strategy": "semantic",
                    "api_token": "ingestion-secret-token",
                    "prompt_url": "https://prompt.secret.invalid",
                    "cache_path": "C:/private/refiner-cache",
                },
                "metadata_enricher": {
                    "use_llm": False,
                    "include_headings": True,
                    "source_file": "C:/private/source.pdf",
                },
            },
            "vision_llm": {
                "enabled": False,
                "provider": "azure_openai",
                "model": "vision-model-v1",
                "max_image_size": 2048,
                "api_key": "vision-secret-key",
                "api_version": "2025-01-01",
                "azure_endpoint": "https://vision.secret.invalid",
                "deployment_name": "vision-deployment",
                "base_url": "https://vision-base.secret.invalid",
            },
        }
    )


def test_fingerprint_is_deterministic_across_mapping_order() -> None:
    settings = _settings()
    ingestion = settings.ingestion
    assert ingestion is not None
    reordered = replace(
        settings,
        ingestion=replace(
            ingestion,
            chunk_refiner=dict(reversed(list(ingestion.chunk_refiner.items()))),
            metadata_enricher=dict(reversed(list(ingestion.metadata_enricher.items()))),
        ),
    )

    assert build_index_fingerprint(settings) == build_index_fingerprint(reordered)


def test_query_evaluation_and_api_credentials_do_not_affect_fingerprint() -> None:
    settings = _settings()
    baseline = build_index_fingerprint(settings)

    variants = [
        replace(
            settings,
            retrieval=replace(settings.retrieval, dense_top_k=99, rrf_k=10),
        ),
        replace(
            settings,
            rerank=replace(
                settings.rerank,
                enabled=False,
                model="rerank-model-v2",
                top_k=1,
            ),
        ),
        replace(
            settings,
            evaluation=replace(
                settings.evaluation,
                provider="another-evaluator",
                metrics=["faithfulness"],
                enabled=False,
            ),
        ),
        replace(
            settings,
            llm=replace(settings.llm, api_key="changed-llm-key"),
            embedding=replace(
                settings.embedding,
                api_key="changed-embedding-key",
            ),
            vision_llm=replace(
                settings.vision_llm,
                api_key="changed-vision-key",
            ),
        ),
    ]

    assert all(build_index_fingerprint(variant) == baseline for variant in variants)


def test_ingestion_chunk_size_and_embedding_model_affect_fingerprint() -> None:
    settings = _settings()
    ingestion = settings.ingestion
    assert ingestion is not None
    baseline = build_index_fingerprint(settings)

    changed_chunk_size = replace(
        settings,
        ingestion=replace(ingestion, chunk_size=256),
    )
    changed_embedding_model = replace(
        settings,
        embedding=replace(settings.embedding, model="embedding-model-v2"),
    )

    assert build_index_fingerprint(changed_chunk_size) != baseline
    assert build_index_fingerprint(changed_embedding_model) != baseline


def test_llm_changes_only_affect_fingerprint_when_refinement_uses_llm() -> None:
    settings = _settings()
    ingestion = settings.ingestion
    assert ingestion is not None
    changed_llm = replace(
        settings,
        llm=replace(settings.llm, model="chat-model-v2"),
    )

    assert build_index_fingerprint(changed_llm) == build_index_fingerprint(settings)

    refinement_enabled = replace(
        settings,
        ingestion=replace(
            ingestion,
            chunk_refiner={**ingestion.chunk_refiner, "use_llm": True},
        ),
    )
    changed_enabled_llm = replace(
        refinement_enabled,
        llm=replace(refinement_enabled.llm, model="chat-model-v2"),
    )

    assert build_index_fingerprint(changed_enabled_llm) != build_index_fingerprint(
        refinement_enabled
    )


def test_vision_model_only_affects_fingerprint_when_vision_is_enabled() -> None:
    settings = _settings()
    changed_disabled_model = replace(
        settings,
        vision_llm=replace(settings.vision_llm, model="vision-model-v2"),
    )

    assert build_index_fingerprint(changed_disabled_model) == build_index_fingerprint(settings)

    vision_enabled = replace(
        settings,
        vision_llm=replace(settings.vision_llm, enabled=True),
    )
    changed_enabled_model = replace(
        vision_enabled,
        vision_llm=replace(vision_enabled.vision_llm, model="vision-model-v2"),
    )

    assert build_index_fingerprint(changed_enabled_model) != build_index_fingerprint(vision_enabled)


def test_collection_name_is_safe_and_uses_twelve_hash_characters() -> None:
    fingerprint = build_index_fingerprint(_settings())

    name = collection_name(" Finance Bench / 2026! ", fingerprint)

    assert name == f"finance_bench_2026__{fingerprint[:12]}"
    assert re.fullmatch(r"[a-z0-9_-]+__[0-9a-f]{12}", name)


def test_payload_contains_no_secret_url_or_path_values() -> None:
    settings = _settings()
    ingestion = settings.ingestion
    assert ingestion is not None
    settings = replace(
        settings,
        ingestion=replace(
            ingestion,
            chunk_refiner={**ingestion.chunk_refiner, "use_llm": True},
        ),
        vision_llm=replace(settings.vision_llm, enabled=True),
    )

    payload_json = json.dumps(build_index_payload(settings), sort_keys=True)

    assert "llm-secret-key" not in payload_json
    assert "embedding-secret-key" not in payload_json
    assert "ingestion-secret-token" not in payload_json
    assert "vision-secret-key" not in payload_json
    assert "https://" not in payload_json
    assert "C:/" not in payload_json
    assert "./" not in payload_json
