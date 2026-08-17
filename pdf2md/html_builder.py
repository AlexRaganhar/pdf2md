"""
Convert OCR text blocks into structured HTML.

Groups text blocks into paragraphs by vertical proximity,
detects headings based on relative font size, and filters
by confidence threshold.
"""

from __future__ import annotations

from .ocr_engine import TextBlock


def build_html(
    pages: list[list[TextBlock]],
    confidence_threshold: float = 0.5,
) -> str:
    """Build an HTML document from per-page OCR results."""
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head><meta charset='utf-8'></head>",
        "<body>",
    ]

    for page_idx, blocks in enumerate(pages):
        filtered = [b for b in blocks if b.confidence >= confidence_threshold]
        if not filtered:
            continue

        filtered.sort(key=lambda b: (b.top, b.left))
        avg_h = _average_height(filtered)

        parts.append(f'<div class="page" id="page-{page_idx + 1}">')

        for para in _group_paragraphs(filtered, avg_h):
            tag = _heading_or_body(para, avg_h)
            text = " ".join(b.text for b in para)
            parts.append(f"<{tag}>{text}</{tag}>")

        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _average_height(blocks: list[TextBlock]) -> float:
    """Compute the average text-block height on a page."""
    if not blocks:
        return 20.0
    return sum(b.height for b in blocks) / len(blocks)


def _group_paragraphs(
    blocks: list[TextBlock], avg_h: float, gap_ratio: float = 1.3
) -> list[list[TextBlock]]:
    """Group consecutive blocks into paragraphs using a vertical-gap heuristic.

    A new paragraph starts when the vertical gap between two consecutive
    blocks exceeds ``avg_h * gap_ratio``.
    """
    if not blocks:
        return []

    threshold = avg_h * gap_ratio
    paragraphs: list[list[TextBlock]] = [[blocks[0]]]

    for block in blocks[1:]:
        prev_bottom = paragraphs[-1][-1].bottom
        if block.top - prev_bottom > threshold:
            paragraphs.append([block])
        else:
            paragraphs[-1].append(block)

    return paragraphs


def _heading_or_body(para: list[TextBlock], avg_h: float) -> str:
    """Return 'h2' if the paragraph looks like a heading, else 'p'."""
    para_avg_h = sum(b.height for b in para) / len(para)
    if para_avg_h > avg_h * 1.4 and len(para) <= 3:
        return "h2"
    return "p"
