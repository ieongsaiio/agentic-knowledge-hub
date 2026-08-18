"""Conservatively merge visually contiguous HTML table blocks."""

from __future__ import annotations

import copy
import hashlib
import html
from dataclasses import dataclass
from html.parser import HTMLParser

from src.libs.loader.parsed_document import ParsedBlock, ParsedDocument, ParsedPage
from src.libs.splitter.html_table_parser import HTMLTableParser


@dataclass(frozen=True)
class TableMergeAssessment:
    """Explain whether two same-page table blocks may be joined."""

    accepted: bool
    reasons: tuple[str, ...]
    horizontal_overlap_ratio: float
    vertical_gap_ratio: float
    left_width: int | None
    right_width: int | None


class _RowExtractor(HTMLParser):
    """Extract top-level rows while preserving their HTML markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[str] = []
        self._table_depth = 0
        self._row_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized == "table":
            self._table_depth += 1
            if self._row_depth:
                self._parts.append(self.get_starttag_text())
            return
        if self._table_depth != 1:
            if self._row_depth:
                self._parts.append(self.get_starttag_text())
            return
        if normalized == "tr" and self._row_depth == 0:
            self._row_depth = 1
            self._parts = [self.get_starttag_text()]
        elif self._row_depth:
            self._row_depth += int(normalized == "tr")
            self._parts.append(self.get_starttag_text())

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag, attrs
        if self._row_depth:
            self._parts.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized == "table":
            if self._row_depth:
                self._parts.append(f"</{tag}>")
            self._table_depth = max(0, self._table_depth - 1)
            return
        if not self._row_depth:
            return
        self._parts.append(f"</{tag}>")
        if normalized == "tr":
            self._row_depth -= 1
            if self._row_depth == 0:
                self.rows.append("".join(self._parts))
                self._parts = []

    def handle_data(self, data: str) -> None:
        if self._row_depth:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._row_depth:
            self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._row_depth:
            self._parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        if self._row_depth:
            self._parts.append(f"<!--{data}-->")


class TableContinuationMerger:
    """Merge adjacent same-page table fragments without inferring headers."""

    def __init__(
        self,
        *,
        minimum_horizontal_overlap: float = 0.85,
        maximum_vertical_gap_ratio: float = 0.08,
    ) -> None:
        if not 0 <= minimum_horizontal_overlap <= 1:
            raise ValueError("minimum_horizontal_overlap must be between 0 and 1")
        if maximum_vertical_gap_ratio < 0:
            raise ValueError("maximum_vertical_gap_ratio cannot be negative")
        self.minimum_horizontal_overlap = minimum_horizontal_overlap
        self.maximum_vertical_gap_ratio = maximum_vertical_gap_ratio
        self._table_parser = HTMLTableParser()

    def process(self, document: ParsedDocument) -> ParsedDocument:
        """Return a copy with visual ordering restored and continuations merged."""
        pages = [self._process_page(page) for page in document.pages]
        return ParsedDocument(
            schema_version=document.schema_version,
            provider=document.provider,
            pages=pages,
            parser_version=document.parser_version,
            raw_markdown=document.raw_markdown,
            raw_artifact=copy.deepcopy(document.raw_artifact),
        )

    def assess(
        self,
        left: ParsedBlock,
        right: ParsedBlock,
        *,
        page_width: float | None,
        page_height: float | None,
    ) -> TableMergeAssessment:
        """Evaluate geometric and structural continuation signals."""
        reasons: list[str] = []
        left_width = self._logical_width(left.content)
        right_width = self._logical_width(right.content)
        if left.type != "table" or right.type != "table":
            reasons.append("non_table_block")
        if left.page_index != right.page_index:
            reasons.append("different_page")
        if left_width is None or right_width is None:
            reasons.append("unparseable_html_table")
        elif left_width != right_width:
            reasons.append("incompatible_logical_width")

        horizontal_overlap = self._horizontal_overlap(left.bbox, right.bbox)
        if horizontal_overlap < self.minimum_horizontal_overlap:
            reasons.append("insufficient_horizontal_overlap")

        vertical_gap = self._vertical_gap_ratio(
            left.bbox,
            right.bbox,
            page_height=page_height,
        )
        if vertical_gap > self.maximum_vertical_gap_ratio:
            reasons.append("vertical_gap_too_large")

        if not reasons:
            reasons.extend(
                (
                    "same_page",
                    "adjacent_visual_blocks",
                    "aligned_horizontal_bounds",
                    "compatible_logical_width",
                )
            )
        return TableMergeAssessment(
            accepted=not any(
                reason
                in {
                    "non_table_block",
                    "different_page",
                    "unparseable_html_table",
                    "incompatible_logical_width",
                    "insufficient_horizontal_overlap",
                    "vertical_gap_too_large",
                }
                for reason in reasons
            ),
            reasons=tuple(reasons),
            horizontal_overlap_ratio=horizontal_overlap,
            vertical_gap_ratio=vertical_gap,
            left_width=left_width,
            right_width=right_width,
        )

    def _process_page(self, page: ParsedPage) -> ParsedPage:
        ordered = sorted(page.blocks, key=self._visual_sort_key)
        output: list[ParsedBlock] = []
        index = 0
        while index < len(ordered):
            block = ordered[index]
            if block.type != "table":
                output.append(copy.deepcopy(block))
                index += 1
                continue

            group = [block]
            assessments: list[TableMergeAssessment] = []
            while index + len(group) < len(ordered):
                candidate = ordered[index + len(group)]
                assessment = self.assess(
                    group[-1],
                    candidate,
                    page_width=page.width,
                    page_height=page.height,
                )
                if not assessment.accepted:
                    break
                group.append(candidate)
                assessments.append(assessment)

            output.append(
                self._merge_group(group, assessments)
                if len(group) > 1
                else copy.deepcopy(block)
            )
            index += len(group)

        for order, block in enumerate(output):
            block.order = order
        return ParsedPage(
            page_index=page.page_index,
            width=page.width,
            height=page.height,
            blocks=output,
        )

    def _merge_group(
        self,
        blocks: list[ParsedBlock],
        assessments: list[TableMergeAssessment],
    ) -> ParsedBlock:
        width = self._logical_width(blocks[0].content)
        assert width is not None
        rows: list[str] = []
        units: list[dict[str, object]] = []
        row_cursor = 0
        for part_index, block in enumerate(blocks):
            if block.caption:
                for caption in block.caption:
                    rows.append(self._context_row(caption, width, "table-section-caption"))
                    row_cursor += 1
            source_rows = self._extract_rows(block.content)
            start = row_cursor
            rows.extend(source_rows)
            row_cursor += len(source_rows)
            for footnote in block.footnotes:
                rows.append(self._context_row(footnote, width, "table-section-footnote"))
                row_cursor += 1
            units.append(
                {
                    "source_block_id": block.block_id,
                    "unit_index": part_index,
                    "row_start": start,
                    "row_end": row_cursor,
                    "caption": "\n".join(block.caption),
                    "footnotes": list(block.footnotes),
                }
            )

        source_block_ids = [block.block_id for block in blocks]
        group_digest = hashlib.sha256(
            ":".join(source_block_ids).encode("utf-8")
        ).hexdigest()[:12]
        table_group_id = (
            f"table_group_p{blocks[0].page_index + 1}_{group_digest}"
        )
        metadata = copy.deepcopy(blocks[0].metadata)
        metadata.update(
            {
                "merged_table": True,
                "chunk_role": "table_group",
                "table_group_id": table_group_id,
                "unit_count": len(blocks),
                "source_block_ids": source_block_ids,
                "source_bboxes": [copy.deepcopy(block.bbox) for block in blocks],
                "units": units,
                "merge_assessments": [
                    {
                        "reasons": list(item.reasons),
                        "horizontal_overlap_ratio": item.horizontal_overlap_ratio,
                        "vertical_gap_ratio": item.vertical_gap_ratio,
                        "logical_width": item.left_width,
                    }
                    for item in assessments
                ],
            }
        )
        bboxes = [block.bbox for block in blocks if block.bbox is not None]
        merged_bbox = None
        if bboxes:
            merged_bbox = [
                min(bbox[0] for bbox in bboxes),
                min(bbox[1] for bbox in bboxes),
                max(bbox[2] for bbox in bboxes),
                max(bbox[3] for bbox in bboxes),
            ]
        return ParsedBlock(
            block_id=f"{blocks[0].block_id}-merged-{len(blocks)}",
            type="table",
            content="<table>" + "".join(rows) + "</table>",
            page_index=blocks[0].page_index,
            bbox=merged_bbox,
            order=blocks[0].order,
            caption=[],
            footnotes=[],
            images=[image for block in blocks for image in copy.deepcopy(block.images)],
            metadata=metadata,
        )

    @staticmethod
    def _context_row(text: str, width: int, class_name: str) -> str:
        escaped = html.escape(text.strip())
        return (
            f'<tr class="{class_name}"><th colspan="{width}">{escaped}</th></tr>'
        )

    @staticmethod
    def _extract_rows(table_html: str) -> list[str]:
        parser = _RowExtractor()
        parser.feed(table_html)
        parser.close()
        return parser.rows

    def _logical_width(self, table_html: str) -> int | None:
        try:
            return self._table_parser.parse(table_html).width
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _visual_sort_key(block: ParsedBlock) -> tuple[float, float, int]:
        if block.bbox is None:
            fallback = block.order if block.order is not None else 10**9
            return (float(fallback), 0.0, fallback)
        fallback = block.order if block.order is not None else 10**9
        return (block.bbox[1], block.bbox[0], fallback)

    @staticmethod
    def _horizontal_overlap(
        left_bbox: list[float] | None,
        right_bbox: list[float] | None,
    ) -> float:
        if left_bbox is None or right_bbox is None:
            return 0.0
        overlap = max(0.0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0]))
        narrower = min(left_bbox[2] - left_bbox[0], right_bbox[2] - right_bbox[0])
        return overlap / narrower if narrower > 0 else 0.0

    @staticmethod
    def _vertical_gap_ratio(
        left_bbox: list[float] | None,
        right_bbox: list[float] | None,
        *,
        page_height: float | None,
    ) -> float:
        if left_bbox is None or right_bbox is None or not page_height:
            return float("inf")
        return max(0.0, right_bbox[1] - left_bbox[3]) / page_height


__all__ = ["TableContinuationMerger", "TableMergeAssessment"]
