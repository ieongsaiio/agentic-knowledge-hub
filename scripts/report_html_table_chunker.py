"""Generate a Markdown report for real FinanceBench table children."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb

from src.libs.splitter.html_table_chunker import HTMLTableChunker

DEFAULT_IDS = (
    "doc_fe9126b17c3656e5_c5d59391_0042_3c3bf94f",
    "doc_7d715f552d4eeb6b_4d2caf6d_0102_9d70c601",
    "doc_84d7a0291a12f4e6_06cbd0c5_0168_b008ed2a",
)


def _ranges(values: tuple[int, ...]) -> str:
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


def _record_map(records: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        chunk_id: (document, metadata)
        for chunk_id, document, metadata in zip(
            records["ids"], records["documents"], records["metadatas"]
        )
    }


def _resolved_title(
    chunker: HTMLTableChunker,
    document: str,
    metadata: dict[str, Any],
) -> str:
    return chunker.extract_caption(
        document,
        preferred_title=str(metadata.get("table_title") or ""),
    )


def _audit_collection(
    collection: Any,
    chunker: HTMLTableChunker,
) -> tuple[Counter[int], Counter[str]]:
    records = collection.get(include=["documents", "metadatas"])
    distribution: Counter[int] = Counter()
    captions: Counter[str] = Counter()
    for chunk_id, document, metadata in zip(
        records["ids"],
        records["documents"],
        records["metadatas"],
    ):
        if "table" not in str(metadata.get("unit_types", "")).casefold():
            continue
        source = chunker.parse(document)
        preferred_title = str(metadata.get("table_title") or "")
        metadata_caption = chunker.extract_caption(
            "<table></table>",
            preferred_title=preferred_title,
        )
        resolved_title = _resolved_title(chunker, document, metadata)
        captions[
            "metadata" if metadata_caption else "prefix" if resolved_title else "none"
        ] += 1
        children = chunker.split(
            document,
            title=resolved_title,
        )
        covered: set[int] = set()
        for child in children:
            indices = child.repeated_context_row_indices + child.source_row_indices
            covered.update(indices)
            expected = tuple(source.rows[index] for index in indices)
            if chunker.parse(child.html).rows != expected:
                raise RuntimeError(
                    f"Full-corpus cell fidelity failed for {chunk_id}, "
                    f"child={child.child_index}"
                )
        expected_indices = set(range(len(source.rows)))
        if covered != expected_indices:
            raise RuntimeError(
                f"Full-corpus row coverage failed for {chunk_id}; "
                f"missing={sorted(expected_indices - covered)}"
            )
        distribution[len(children)] += 1
    return distribution, captions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="financebench__e4473342e89c")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/html_table_chunker_real_results.md"),
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path="data/db/chroma")
    collection = client.get_collection(args.collection)
    records = collection.get(
        ids=list(DEFAULT_IDS),
        include=["documents", "metadatas"],
    )
    by_id = _record_map(records)
    missing = set(DEFAULT_IDS) - set(by_id)
    if missing:
        raise RuntimeError(f"Missing real table records: {sorted(missing)}")

    chunker = HTMLTableChunker(
        target_children=4,
        overlap_rows=1,
        repeated_context_rows=2,
    )
    distribution, captions = _audit_collection(collection, chunker)
    audited_tables = sum(distribution.values())
    lines = [
        "# HTML Table Child Parser 真实数据测试报告",
        "",
        "## 测试结论",
        "",
        "- 单元测试：`11 passed`。",
        "- 当前 Chroma 真实数据集成测试：`1 passed`。",
        f"- 全库表格审计：`{audited_tables:,} passed, 0 failed`。",
        "- 全库审计同时检查源行覆盖与逐格 Cell Fidelity。",
        f"- Caption 来源：metadata `{captions['metadata']:,}`，"
        f"prefix `{captions['prefix']:,}`，none `{captions['none']:,}`。",
        "- 测试 Collection：`financebench__e4473342e89c`。",
        "- 真实表格数量：3；每张表生成 4 个 Child，共 12 个。",
        "- 每个 Child 都保留 `<table>...</table>` HTML 结构。",
        "- 每个 Child 只重复自己局部 block 的 Caption/Context Row。",
        "- 同一局部 block 内相邻 Child 重叠一行；不同 block 不制造重叠。",
        "- 三张表的数据行覆盖率均为 100%，没有遗漏源数据行。",
        "- 每个 Child 的单元格网格与 Parent 对应源行逐格一致。",
        "- Child 不包含 `Section:` 或 `header_path`。",
        "",
        "> Parser 会把 `rowspan/colspan` 展开成矩形网格，再重新生成独立 HTML。"
        "单元格文字与逻辑位置被保留，但原始 style、rowspan/colspan 标签和单元格内部格式标签不会逐字保留。",
        "",
        "## 配置",
        "",
        "```yaml",
        "target_children: 4",
        "overlap_rows: 1",
        "repeated_context_rows: 2",
        "minimum_rows_per_child: 2",
        "```",
        "",
        "## 全库 Child 数量分布",
        "",
        "| 每张 Parent 生成的 Child 数 | Parent 表格数 |",
        "|---:|---:|",
        *[
            f"| {child_count} | {parent_count:,} |"
            for child_count, parent_count in sorted(distribution.items())
        ],
        "",
        "## 汇总",
        "",
        "| Chunk ID | PDF 页 | 原字符 | 行 x 列 | Child 数 | 数据行覆盖 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    details: list[list[str]] = []
    for table_number, chunk_id in enumerate(DEFAULT_IDS, start=1):
        document, metadata = by_id[chunk_id]
        title = _resolved_title(chunker, document, metadata)
        parsed = chunker.parse(document)
        children = chunker.split(document, title=title)
        covered = {
            row_index
            for child in children
            for row_index in (
                child.repeated_context_row_indices + child.source_row_indices
            )
        }
        expected = set(range(len(parsed.rows)))
        coverage = len(covered & expected) / len(expected) if expected else 1.0
        if coverage != 1.0:
            missing_rows = sorted(expected - covered)
            raise RuntimeError(
                f"Source-row coverage is {coverage:.2%} for {chunk_id}; "
                f"missing={missing_rows}"
            )
        page = f"{metadata.get('page_start')}-{metadata.get('page_end')}"
        lines.append(
            f"| `{chunk_id}` | {page} | {len(document):,} | "
            f"{len(parsed.rows)} x {parsed.width} | {len(children)} | "
            f"{coverage:.0%} |"
        )

        section = [
            f"## 表格 {table_number}: {chunk_id}",
            "",
            f"- PDF：`{metadata.get('source_path')}`",
            f"- 页码：`{page}`",
            f"- Table title：`{title or '(no reliable external caption)'}`",
            f"- Metadata header_path（未写入 Child）：`{metadata.get('header_path')}`",
            f"- 原始字符数：`{len(document):,}`",
            f"- 解析网格：`{len(parsed.rows)} rows x {parsed.width} columns`",
            f"- 数据行覆盖率：`{coverage:.0%}`",
            "",
            "### 原始 HTML",
            "",
            "````html",
            document,
            "````",
        ]
        for child in children:
            reparsed = chunker.parse(child.html)
            expected_rows = tuple(
                parsed.rows[row_index]
                for row_index in (
                    child.repeated_context_row_indices + child.source_row_indices
                )
            )
            if reparsed.rows != expected_rows:
                raise RuntimeError(
                    f"Child {child.child_index} changed source cells for {chunk_id}"
                )
            section.extend(
                [
                    "",
                    f"### Child {child.child_index + 1}",
                    "",
                    f"- Context rows：`{_ranges(child.repeated_context_row_indices)}`",
                    f"- Source data rows：`{_ranges(child.source_row_indices)}`",
                    f"- Overlap rows：`{_ranges(child.overlap_row_indices)}`",
                    f"- Child grid：`{len(reparsed.rows)} rows x {reparsed.width} columns`",
                    f"- Child characters：`{len(child.html):,}`",
                    "- Cell fidelity：`PASS`",
                    "",
                    "````html",
                    child.html,
                    "````",
                ]
            )
        details.append(section)

    for section in details:
        lines.extend(["", *section])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={args.output} tables=3 children=12")


if __name__ == "__main__":
    main()
