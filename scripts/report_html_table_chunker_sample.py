"""Build a reproducible 30-table HTML child-parser validation report."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.libs.splitter.html_table_chunker import (
    HTMLTableChild,
    HTMLTableChunker,
    ParsedHTMLTable,
)


@dataclass(frozen=True)
class TableSample:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    parsed: ParsedHTMLTable
    size_group: str
    detected_blocks: int


@dataclass(frozen=True)
class ValidationResult:
    sample: TableSample
    title: str
    caption_source: str
    children: tuple[HTMLTableChild, ...]
    coverage: float
    cell_fidelity: bool
    local_block_count: int
    warnings: tuple[str, ...]


def _row_range(values: tuple[int, ...]) -> str:
    if not values:
        return "(none)"
    groups: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(groups)


def _select_samples(
    records: list[tuple[str, str, dict[str, Any], ParsedHTMLTable, int]],
    *,
    sample_size: int,
    seed: int,
) -> list[TableSample]:
    if sample_size != 30:
        raise ValueError("This stratified audit currently requires sample_size=30")
    single_block = [record for record in records if record[4] == 1]
    complex_tables = [record for record in records if record[4] > 1]
    ordered = sorted(
        single_block,
        key=lambda item: (len(item[3].rows), len(item[1])),
    )
    boundaries = (
        0,
        len(ordered) // 3,
        (2 * len(ordered)) // 3,
        len(ordered),
    )
    names = ("small", "medium", "large")
    rng = random.Random(seed)
    selected: list[TableSample] = []
    per_group = 8
    for group_index, name in enumerate(names):
        bucket = ordered[boundaries[group_index] : boundaries[group_index + 1]]
        if len(bucket) < per_group:
            raise RuntimeError(f"Not enough {name} tables for sampling")
        for chunk_id, document, metadata, parsed, block_count in rng.sample(
            bucket,
            per_group,
        ):
            selected.append(
                TableSample(
                    chunk_id=chunk_id,
                    document=document,
                    metadata=metadata,
                    parsed=parsed,
                    size_group=name,
                    detected_blocks=block_count,
                )
            )
    ranked_complex = sorted(
        complex_tables,
        key=lambda item: (-item[4], -len(item[3].rows), item[0]),
    )
    for chunk_id, document, metadata, parsed, block_count in ranked_complex[:6]:
        selected.append(
            TableSample(
                chunk_id=chunk_id,
                document=document,
                metadata=metadata,
                parsed=parsed,
                size_group="complex",
                detected_blocks=block_count,
            )
        )
    return selected


def _validate(
    sample: TableSample,
    chunker: HTMLTableChunker,
) -> ValidationResult:
    preferred_title = str(sample.metadata.get("table_title") or "")
    metadata_caption = chunker.extract_caption(
        "<table></table>",
        preferred_title=preferred_title,
    )
    title = chunker.extract_caption(
        sample.document,
        preferred_title=preferred_title,
    )
    caption_source = "metadata" if metadata_caption else "prefix" if title else "none"
    children = tuple(chunker.split(sample.document, title=title))
    covered: set[int] = set()
    cell_fidelity = True
    for child in children:
        indices = child.repeated_context_row_indices + child.source_row_indices
        covered.update(indices)
        expected = tuple(sample.parsed.rows[index] for index in indices)
        if chunker.parse(child.html).rows != expected:
            cell_fidelity = False
    source_rows = set(range(len(sample.parsed.rows)))
    coverage = len(covered & source_rows) / len(source_rows)
    contexts = {child.repeated_context_row_indices for child in children}
    warnings: list[str] = []
    if not title:
        warnings.append("no_reliable_external_caption")
    if len(children) > 5:
        warnings.append("more_than_five_children")
    if coverage != 1.0:
        warnings.append("incomplete_source_row_coverage")
    if not cell_fidelity:
        warnings.append("cell_fidelity_failure")
    if any("Section:" in child.html for child in children):
        warnings.append("section_path_leaked")
    return ValidationResult(
        sample=sample,
        title=title,
        caption_source=caption_source,
        children=children,
        coverage=coverage,
        cell_fidelity=cell_fidelity,
        local_block_count=len(contexts),
        warnings=tuple(warnings),
    )


def _write_report(
    results: list[ValidationResult],
    *,
    collection_name: str,
    seed: int,
    output: Path,
) -> None:
    passed = sum(
        result.coverage == 1.0
        and result.cell_fidelity
        and not any("Section:" in child.html for child in result.children)
        for result in results
    )
    child_total = sum(len(result.children) for result in results)
    lines = [
        "# HTML Table Child Parser 30 表分层抽样报告",
        "",
        "## 测试方法",
        "",
        f"- Collection：`{collection_name}`",
        f"- 固定随机种子：`{seed}`",
        "- 单 Block 的小型、中型、大型表格各抽取 8 个。",
        "- 额外强制抽取 6 个多 Block 复杂表格。",
        "- 规模排序依据：解析后的逻辑行数，其次为原始字符数。",
        "- 每个样本检查：源行覆盖、逐格 Cell Fidelity、HTML 可重解析、Section Path 泄漏。",
        "",
        "## 总结",
        "",
        f"- 样本：`{len(results)}`",
        f"- 结构测试通过：`{passed}/{len(results)}`",
        f"- 生成 Child：`{child_total}`",
        f"- Metadata Caption：`{sum(result.caption_source == 'metadata' for result in results)}`",
        f"- Prefix Caption：`{sum(result.caption_source == 'prefix' for result in results)}`",
        f"- 无可靠 Caption：`{sum(result.caption_source == 'none' for result in results)}`",
        f"- 多于 5 个 Child：`{sum(len(result.children) > 5 for result in results)}`",
        "",
        "> External Caption 缺失不是结构测试失败；这类表仍依靠自己局部 Context Row 表达列含义。",
        "",
        "## 样本汇总",
        "",
        "| # | 规模 | Chunk ID | PDF | 页 | 字符 | 网格 | Block | Child | Caption | Coverage | Fidelity | Warnings |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for index, result in enumerate(results, start=1):
        sample = result.sample
        source_name = Path(str(sample.metadata.get("source_path", ""))).name
        page = f"{sample.metadata.get('page_start')}-{sample.metadata.get('page_end')}"
        warnings = ", ".join(result.warnings) or "none"
        lines.append(
            f"| {index} | {sample.size_group} | `{sample.chunk_id}` | "
            f"{source_name} | {page} | {len(sample.document):,} | "
            f"{len(sample.parsed.rows)}x{sample.parsed.width} | "
            f"{result.local_block_count} | {len(result.children)} | "
            f"{result.caption_source} | "
            f"{result.coverage:.0%} | {'PASS' if result.cell_fidelity else 'FAIL'} | "
            f"{warnings} |"
        )

    for index, result in enumerate(results, start=1):
        sample = result.sample
        page = f"{sample.metadata.get('page_start')}-{sample.metadata.get('page_end')}"
        lines.extend(
            [
                "",
                f"## {index}. {sample.chunk_id}",
                "",
                f"- 规模：`{sample.size_group}`",
                f"- PDF：`{sample.metadata.get('source_path')}`",
                f"- 页码：`{page}`",
                f"- External Caption：`{result.title or '(none)'}`",
                f"- Caption Source：`{result.caption_source}`",
                f"- Metadata header_path（未写入 Child）：`{sample.metadata.get('header_path')}`",
                f"- 原始字符：`{len(sample.document):,}`",
                f"- 解析网格：`{len(sample.parsed.rows)} x {sample.parsed.width}`",
                f"- Local Blocks：`{result.local_block_count}`",
                f"- Child 数：`{len(result.children)}`",
                f"- Source Row Coverage：`{result.coverage:.0%}`",
                f"- Cell Fidelity：`{'PASS' if result.cell_fidelity else 'FAIL'}`",
                f"- Warnings：`{', '.join(result.warnings) or 'none'}`",
                "",
                "### Parent 原始 HTML",
                "",
                "````html",
                sample.document,
                "````",
            ]
        )
        for child in result.children:
            lines.extend(
                [
                    "",
                    f"### Child {child.child_index + 1}",
                    "",
                    f"- Context rows：`{_row_range(child.repeated_context_row_indices)}`",
                    f"- Source rows：`{_row_range(child.source_row_indices)}`",
                    f"- Overlap rows：`{_row_range(child.overlap_row_indices)}`",
                    f"- Characters：`{len(child.html):,}`",
                    "",
                    "````html",
                    child.html,
                    "````",
                ]
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="financebench__e4473342e89c")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/html_table_chunker_sample_30.md"),
    )
    args = parser.parse_args()

    collection = chromadb.PersistentClient(path="data/db/chroma").get_collection(
        args.collection
    )
    chunker = HTMLTableChunker(
        target_children=4,
        overlap_rows=1,
        repeated_context_rows=2,
    )
    stored = collection.get(include=["documents", "metadatas"])
    tables: list[tuple[str, str, dict[str, Any], ParsedHTMLTable, int]] = []
    for chunk_id, document, metadata in zip(
        stored["ids"],
        stored["documents"],
        stored["metadatas"],
    ):
        if "table" not in str(metadata.get("unit_types", "")).casefold():
            continue
        parsed = chunker.parse(document)
        probe_children = chunker.split(document, title="")
        block_count = len(
            {child.repeated_context_row_indices for child in probe_children}
        )
        tables.append((chunk_id, document, metadata, parsed, block_count))

    samples = _select_samples(
        tables,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    results = [_validate(sample, chunker) for sample in samples]
    _write_report(
        results,
        collection_name=args.collection,
        seed=args.seed,
        output=args.output,
    )
    failed = [
        result
        for result in results
        if result.coverage != 1.0 or not result.cell_fidelity
    ]
    print(
        f"samples={len(results)} children={sum(len(r.children) for r in results)} "
        f"failed={len(failed)} report={args.output}"
    )
    if failed:
        raise RuntimeError(f"{len(failed)} sampled tables failed validation")


if __name__ == "__main__":
    main()
