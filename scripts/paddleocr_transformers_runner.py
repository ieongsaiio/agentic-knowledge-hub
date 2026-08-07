#!/usr/bin/env python
"""Run PaddleOCR-VL with the Transformers engine and emit one JSON result."""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MARKDOWN_IGNORE_LABELS = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL in a PaddleOCR container using Transformers."
    )
    parser.add_argument("--input", required=True, help="Input image or PDF path.")
    parser.add_argument("--output", required=True, help="Destination JSON file path.")
    parser.add_argument(
        "--config-json",
        default="{}",
        help="JSON object containing PaddleOCR pipeline and restructure settings.",
    )
    parser.add_argument(
        "--vision-enabled",
        action="store_true",
        help="Keep image regions in generated Markdown.",
    )
    return parser.parse_args()


def parse_config(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--config-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--config-json must contain a JSON object")
    return value


def to_jsonable(value: Any) -> Any:
    """Recursively convert common PaddleOCR result values to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(item) for item in value]

    module_name = type(value).__module__
    if module_name.startswith("PIL.") and hasattr(value, "mode") and hasattr(value, "size"):
        return {
            "type": type(value).__name__,
            "mode": value.mode,
            "size": list(value.size),
        }

    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def result_json(result: Any) -> Any:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    return to_jsonable(value)


def result_markdown(result: Any) -> Mapping[str, Any]:
    value = getattr(result, "markdown", {})
    if callable(value):
        value = value()
    return value if isinstance(value, Mapping) else {}


def page_index(result: Any, fallback: int) -> int:
    payload = result_json(result)
    if isinstance(payload, Mapping):
        candidates = [payload]
        nested = payload.get("res")
        if isinstance(nested, Mapping):
            candidates.insert(0, nested)
        for candidate in candidates:
            value = candidate.get("page_index")
            if isinstance(value, int):
                return value
    return fallback


def safe_asset_path(name: Any, page_number: int) -> Path:
    raw = str(name).replace("\\", "/")
    parts = [part for part in PurePosixPath(raw).parts if part not in ("", ".", "..", "/")]
    if not parts:
        parts = ["image.png"]
    path = Path(f"page_{page_number:04d}").joinpath(*parts)
    return path if path.suffix else path.with_suffix(".png")


def save_image(image: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.save(destination)
        return
    if isinstance(image, (bytes, bytearray, memoryview)):
        destination.write_bytes(bytes(image))
        return
    try:
        from PIL import Image

        Image.fromarray(image).save(destination)
    except (ImportError, TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported Markdown image type: {type(image).__name__}") from exc


def export_markdown_images(images: Any, assets_dir: Path, page_number: int) -> dict[str, str]:
    if not isinstance(images, Mapping):
        return {}

    exported: dict[str, str] = {}
    for source_name, image in images.items():
        relative_asset = safe_asset_path(source_name, page_number)
        destination = assets_dir / relative_asset
        save_image(image, destination)
        exported[str(source_name)] = (Path("assets") / relative_asset).as_posix()
    return exported


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        config = parse_config(args.config_json)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not input_path.exists():
        print(f"error: input does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        from paddleocr import PaddleOCRVL
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "error: PaddleOCR with PaddleOCRVL is not installed. "
            "Run this script inside a PaddleOCR-VL Docker image with transformers support.",
            file=sys.stderr,
        )
        print(f"details: {exc}", file=sys.stderr)
        return 2

    ignore_labels = list(DEFAULT_MARKDOWN_IGNORE_LABELS)
    if not args.vision_enabled:
        ignore_labels.append("image")

    started_at = time.perf_counter()
    try:
        engine = str(config.get("engine", "transformers"))
        if engine != "transformers":
            raise ValueError("The Docker runner only supports engine='transformers'")
        pipeline = PaddleOCRVL(
            engine=engine,
            pipeline_version=str(config.get("pipeline_version", "v1.6")),
            markdown_ignore_labels=ignore_labels,
            use_queues=bool(config.get("use_queues", True)),
            device=str(config.get("device", "gpu:0")),
        )
        pages_res = list(pipeline.predict_iter(input=str(input_path)))
        restructured = list(
            pipeline.restructure_pages(
                pages_res,
                merge_tables=bool(config.get("merge_tables", True)),
                relevel_titles=bool(config.get("relevel_titles", True)),
                concatenate_pages=bool(config.get("concatenate_pages", False)),
            )
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        assets_dir = output_path.parent / "assets"
        restructured_pages = []
        for position, result in enumerate(restructured):
            index = page_index(result, position)
            markdown = result_markdown(result)
            markdown_text = markdown.get("markdown_texts", markdown.get("markdown_text", ""))
            images = markdown.get("markdown_images", markdown.get("images", {}))
            image_files = export_markdown_images(
                images,
                assets_dir=assets_dir,
                page_number=index,
            )
            restructured_pages.append(
                {
                    "page_index": index,
                    "markdown_text": to_jsonable(markdown_text),
                    "json": result_json(result),
                    "images": image_files,
                }
            )

        payload = {
            "elapsed_seconds": time.perf_counter() - started_at,
            "config": to_jsonable(config),
            "pages": [result_json(result) for result in pages_res],
            "restructured_pages": restructured_pages,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"error: PaddleOCR-VL processing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote PaddleOCR-VL results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
