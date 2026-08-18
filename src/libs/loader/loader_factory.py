"""Factory for creating configured PDF loaders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.loader.base_loader import BaseLoader


def _get_pdf_loader() -> type[BaseLoader]:
    from src.libs.loader.pdf_loader import PdfLoader

    return PdfLoader


def _get_paddle_pdf_loader() -> type[BaseLoader]:
    from src.libs.loader.paddle_pdf_loader import PaddlePdfLoader

    return PaddlePdfLoader


def _get_mineru_pdf_loader() -> type[BaseLoader]:
    from src.libs.loader.mineru_pdf_loader import MineruPdfLoader

    return MineruPdfLoader


class LoaderFactory:
    """Create the PDF loader selected by ingestion settings."""

    _PROVIDERS: dict[str, Callable[[], type[BaseLoader]]] = {
        "default": _get_pdf_loader,
        "paddle": _get_paddle_pdf_loader,
        "mineru": _get_mineru_pdf_loader,
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        image_storage_dir: str | Path | None = None,
    ) -> BaseLoader:
        """Create a loader from ``settings.ingestion.loader.provider``."""
        try:
            provider = settings.ingestion.loader.provider.lower()
        except AttributeError as exc:
            raise ValueError(
                "Missing required configuration: settings.ingestion.loader.provider"
            ) from exc

        provider_loader = cls._PROVIDERS.get(provider)
        if provider_loader is None:
            available = ", ".join(sorted(cls._PROVIDERS))
            raise ValueError(
                f"Unsupported loader provider: '{provider}'. Available providers: {available}."
            )

        loader_class = provider_loader()
        kwargs = {
            "extract_images": bool(settings.vision_llm is not None and settings.vision_llm.enabled)
        }
        if image_storage_dir is not None:
            kwargs["image_storage_dir"] = image_storage_dir
        if provider == "paddle":
            kwargs["paddle_config"] = dict(settings.ingestion.loader.paddle)
        elif provider == "mineru":
            kwargs["mineru_config"] = dict(settings.ingestion.loader.mineru)

        return loader_class(**kwargs)


__all__ = ["LoaderFactory"]
