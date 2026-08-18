"""Provider selection tests for PDF loaders."""

from types import SimpleNamespace

import pytest

from src.libs.loader.loader_factory import LoaderFactory


class _FakeLoader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _settings(provider: str, *, vision: bool = False):
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            loader=SimpleNamespace(
                provider=provider,
                paddle={
                    "backend": "api",
                    "docker": {"engine": "transformers", "merge_tables": True},
                    "api": {
                        "model": "PaddleOCR-VL-1.6",
                        "token_env": "PADDLEOCR_API_TOKEN",
                    },
                },
                mineru={
                    "backend": "api",
                    "api": {
                        "model_version": "vlm",
                        "token_env": "MINERU_API_TOKEN",
                    },
                    "ignored_block_types": ["page_header", "page_footer"],
                },
            )
        ),
        vision_llm=SimpleNamespace(enabled=vision),
    )


@pytest.mark.parametrize("provider", ["default", "paddle", "mineru"])
def test_factory_selects_configured_provider(monkeypatch, provider):
    monkeypatch.setitem(LoaderFactory._PROVIDERS, provider, lambda: _FakeLoader)

    loader = LoaderFactory.create(
        _settings(provider, vision=True),
        image_storage_dir="data/images/test",
    )

    assert loader.kwargs["extract_images"] is True
    assert loader.kwargs["image_storage_dir"] == "data/images/test"
    if provider == "paddle":
        assert loader.kwargs["paddle_config"]["backend"] == "api"
        assert loader.kwargs["paddle_config"]["api"]["model"] == "PaddleOCR-VL-1.6"
    elif provider == "mineru":
        assert loader.kwargs["mineru_config"]["backend"] == "api"
        assert loader.kwargs["mineru_config"]["api"]["model_version"] == "vlm"
    else:
        assert "paddle_config" not in loader.kwargs
        assert "mineru_config" not in loader.kwargs


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported loader provider"):
        LoaderFactory.create(_settings("unknown"))
