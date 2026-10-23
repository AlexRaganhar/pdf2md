"""Turn OCR blocks into Markdown and page-numbered plain text.

All layout heuristics live here: reading order, paragraph grouping, heading
detection and CJK-aware joining. There is no HTML/Docling step.
"""

from __future__ import annotations

import re

from .ocr import TextBlock

PAGE_EMPTY_TEXT = "（本页未识别到文字）"
PAGE_FAILED_TEXT = "（本页 OCR 失败）"

# Characters that glue OCR blocks together without a space: CJK ideographs,
# CJK punctuation, fullwidth forms, and the punctuation marks Chinese OCR
# commonly emits (curly quotes, dashes, ellipsis).
_CJK_CHAR = re.compile(
    "[\u2013-\u201f\u2026\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)


def failed_page_marker(page_no: int) -> str:
    """Visible marker inserted for a page whose OCR failed."""
    return f"（第 {page_no} 页 OCR 失败）"


def _join_blocks(blocks: list[TextBlock]) -> str:
    """Join a paragraph's OCR blocks, inserting no space inside CJK text."""
    texts = [b.text for b in blocks if b.text]
    if not texts:
        return ""
    joined = texts[0]
    for text in texts[1:]:
        if _CJK_CHAR.match(joined[-1]) and _CJK_CHAR.match(text[0]):
            joined += text
        else:
            joined += " " + text
    return joined


def _average_height(blocks: list[TextBlock]) -> float:
    return sum(b.height for b in blocks) / len(blocks) if blocks else 20.0


def _group_paragraphs(
    blocks: list[TextBlock], avg_h: float, gap_ratio: float = 1.3
) -> list[list[TextBlock]]:
    """Start a new paragraph when the vertical gap exceeds ``avg_h * gap_ratio``."""
    if not blocks:
        return []

    threshold = avg_h * gap_ratio
    paragraphs: list[list[TextBlock]] = [[blocks[0]]]
    for block in blocks[1:]:
        if block.top - paragraphs[-1][-1].bottom > threshold:
            paragraphs.append([block])
        else:
            paragraphs[-1].append(block)
    return paragraphs


def _heading_or_body(para: list[TextBlock], avg_h: float) -> str:
    """Return ``h2`` if the paragraph looks like a heading, else ``p``."""
    para_avg_h = sum(b.height for b in para) / len(para)
    if para_avg_h > avg_h * 1.4 and len(para) <= 3:
        return "h2"
    return "p"


def page_paragraphs(
    blocks: list[TextBlock], confidence_threshold: float = 0.5
) -> list[tuple[str, str]]:
    """Group one page's blocks into ``(tag, text)`` paragraphs in reading order."""
    filtered = [b for b in blocks if b.confidence >= confidence_threshold]
    if not filtered:
        return []

    filtered.sort(key=lambda b: (b.top, b.left))
    avg_h = _average_height(filtered)
    return [
        (_heading_or_body(para, avg_h), _join_blocks(para))
        for para in _group_paragraphs(filtered, avg_h)
    ]


def _escape_block_start(text: str) -> str:
    """Neutralise characters that would start a Markdown block.

    Legal text frequently starts lines with "1." or "-", which Markdown would
    otherwise turn into a list. Only the leading marker is escaped; inline
    characters are left untouched so the text stays readable.
    """
    text = re.sub(r"^(\s*)([#>])", r"\1\\\2", text, count=1)
    text = re.sub(r"^(\s*)([-+*])(?=\s)", r"\1\\\2", text, count=1)
    text = re.sub(r"^(\s*)(\d+)([.)])(?=\s)", r"\1\2\\\3", text, count=1)
    return text


def pages_to_markdown(
    pages: list[list[TextBlock]],
    threshold: float = 0.5,
    failed_pages: frozenset[int] = frozenset(),
) -> str:
    """Render per-page OCR blocks as Markdown with ``<!-- Page N -->`` markers."""
    parts: list[str] = []
    for page_no, blocks in enumerate(pages, start=1):
        parts.append(f"<!-- Page {page_no} -->")

        if page_no in failed_pages:
            parts.append(failed_page_marker(page_no))
            continue

        for tag, text in page_paragraphs(blocks, threshold):
            text = _escape_block_start(text.strip())
            if not text:
                continue
            parts.append(f"## {text}" if tag == "h2" else text)

    if not parts:
        return ""
    return "\n\n".join(parts).strip() + "\n"


def blocks_to_text(blocks: list[TextBlock], threshold: float) -> str:
    """One line per block, sorted into reading order (page-numbered output)."""
    ordered = sorted(
        (b for b in blocks if b.confidence >= threshold),
        key=lambda b: (b.top, b.left),
    )
    return "\n".join(b.text for b in ordered).strip() or PAGE_EMPTY_TEXT


def page_numbered_text(
    pages: list[list[TextBlock]],
    total: int,
    threshold: float = 0.5,
    failed_pages: frozenset[int] = frozenset(),
) -> str:
    """Format per-page text with explicit PDF page-number markers."""
    parts: list[str] = []
    for page_no, blocks in enumerate(pages, start=1):
        body = (
            PAGE_FAILED_TEXT
            if page_no in failed_pages
            else blocks_to_text(blocks, threshold)
        )
        parts.append(f"【第 {page_no} 页 / 共 {total} 页】\n{body}")
    return "\n\n".join(parts) + "\n"
