"""Tests for settings loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.core.settings import SettingsError, load_settings


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_load_settings_success(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
        - mrr
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    ingestion:
      chunk_size: 1000
      chunk_overlap: 200
      splitter: recursive
      batch_size: 100
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.llm.provider == "openai"
    assert settings.embedding.dimensions == 1536
    assert settings.vector_store.collection_name == "knowledge_hub"
    assert settings.retrieval.rrf_k == 60
    assert settings.retrieval.enable_dense is True
    assert settings.retrieval.enable_sparse is True
    assert settings.retrieval.dense_weight == 0.5
    assert settings.retrieval.sparse_weight == 0.5
    assert settings.rerank.provider == "none"
    assert settings.rerank.api_args == {}
    assert settings.evaluation.metrics == ["hit_rate", "mrr"]
    assert settings.observability.log_level == "INFO"
    assert settings.ingestion is not None
    assert settings.ingestion.length_unit == "characters"
    assert settings.ingestion.tokenizer_model is None


def test_load_settings_retrieval_weights(tmp_path: Path) -> None:
    config = """
    llm: {provider: openai, model: gpt-4o-mini, temperature: 0.0, max_tokens: 1024}
    embedding: {provider: openai, model: text-embedding-3-small, dimensions: 1536}
    vector_store: {provider: chroma, persist_directory: ./data/db/chroma, collection_name: test}
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
      dense_weight: 0.7
      sparse_weight: 0.3
    rerank: {enabled: false, provider: none, model: "", top_k: 5}
    evaluation: {enabled: false, provider: custom, metrics: [hit_rate]}
    observability: {log_level: INFO, trace_enabled: false, trace_file: ./logs/test.jsonl, structured_logging: true}
    ingestion: {splitter: recursive, chunk_size: 1000, chunk_overlap: 200, batch_size: 100}
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.retrieval.dense_weight == 0.7
    assert settings.retrieval.sparse_weight == 0.3


def test_load_settings_token_based_ingestion(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: siliconflow
      model: Qwen/Qwen3-Embedding-0.6B
      dimensions: 1024
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: ""
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    ingestion:
      splitter: recursive
      length_unit: tokens
      tokenizer_model: Qwen/Qwen3-Embedding-0.6B
      chunk_size: 512
      chunk_overlap: 80
      batch_size: 100
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.ingestion is not None
    assert settings.ingestion.length_unit == "tokens"
    assert settings.ingestion.tokenizer_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.ingestion.chunk_size == 512
    assert settings.ingestion.chunk_overlap == 80


def test_load_settings_rerank_api_args(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: true
      provider: cross_encoder_api
      model: ""
      top_k: 5
      api_args:
        base_url: https://api.siliconflow.com/v1
        api_key_env: SILICONFLOW_API_KEY
        model: Qwen/Qwen3-Reranker-0.6B
        rerank_threshold: 0.2
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.rerank.provider == "cross_encoder_api"
    assert settings.rerank.model == ""
    assert settings.rerank.api_args == {
        "base_url": "https://api.siliconflow.com/v1",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "rerank_threshold": 0.2,
    }


def test_missing_required_field_raises_error(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(settings_path)
