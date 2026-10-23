"""Unit tests for the Markdown/page-text writer (no OCR runtime)."""

from __future__ import annotations

import unittest

from pdf2md.markdown import (
    PAGE_EMPTY_TEXT,
    PAGE_FAILED_TEXT,
    _escape_block_start,
    blocks_to_text,
    page_numbered_text,
    page_paragraphs,
    pages_to_markdown,
)
from pdf2md.ocr import TextBlock


def make_block(text, top=0.0, left=0.0, height=20.0, confidence=0.9):
    bbox = [
        [left, top],
        [left + 100.0, top],
        [left + 100.0, top + height],
        [left, top + height],
    ]
    return TextBlock(text=text, confidence=confidence, bbox=bbox)


class ParagraphTests(unittest.TestCase):
    def test_heading_detected_by_height(self):
        blocks = [
            make_block("第一章 总则", height=40.0),
            make_block("正文一", top=100.0),
            make_block("正文二", top=130.0),
            make_block("正文三", top=160.0),
        ]
        paragraphs = page_paragraphs(blocks)
        self.assertEqual(paragraphs[0], ("h2", "第一章 总则"))
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(paragraphs[1][0], "p")

    def test_cjk_blocks_join_without_space(self):
        paragraphs = page_paragraphs(
            [
                make_block("原告张三", top=80.0),
                make_block("于2026年3月", top=80.0, left=150.0),
            ]
        )
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0][1], "原告张三于2026年3月")

    def test_large_vertical_gap_starts_new_paragraph(self):
        paragraphs = page_paragraphs(
            [make_block("第一段"), make_block("第二段", top=200.0)]
        )
        self.assertEqual([text for _, text in paragraphs], ["第一段", "第二段"])

    def test_low_confidence_blocks_are_dropped(self):
        paragraphs = page_paragraphs(
            [make_block("noise", confidence=0.1), make_block("real")]
        )
        self.assertEqual([text for _, text in paragraphs], ["real"])


class MarkdownTests(unittest.TestCase):
    def test_page_markers_and_paragraph_breaks(self):
        markdown = pages_to_markdown(
            [[make_block("第一段"), make_block("第二段", top=100.0)]]
        )
        self.assertIn("<!-- Page 1 -->", markdown)
        self.assertEqual(markdown, "<!-- Page 1 -->\n\n第一段\n\n第二段\n")

    def test_empty_page_keeps_its_marker(self):
        self.assertEqual(pages_to_markdown([[]]), "<!-- Page 1 -->\n")

    def test_failed_page_gets_visible_marker(self):
        markdown = pages_to_markdown([[]], failed_pages=frozenset({1}))
        self.assertIn("（第 1 页 OCR 失败）", markdown)

    def test_leading_list_marker_is_escaped(self):
        markdown = pages_to_markdown([[make_block("1. 被告人")]])
        self.assertIn("1\\. 被告人", markdown)


class EscapeTests(unittest.TestCase):
    def test_heading_and_list_markers(self):
        self.assertEqual(_escape_block_start("# 标题"), "\\# 标题")
        self.assertEqual(_escape_block_start("1. 被告人"), "1\\. 被告人")
        self.assertEqual(_escape_block_start("- 证据"), "\\- 证据")
        self.assertEqual(_escape_block_start("正文"), "正文")


class PageTextTests(unittest.TestCase):
    def test_page_text_has_page_headers(self):
        text = page_numbered_text([[make_block("a")], []], total=2)
        self.assertIn("【第 1 页 / 共 2 页】\na", text)
        self.assertIn(f"【第 2 页 / 共 2 页】\n{PAGE_EMPTY_TEXT}", text)

    def test_failed_page_uses_failure_text(self):
        text = page_numbered_text([[]], total=1, failed_pages=frozenset({1}))
        self.assertIn(PAGE_FAILED_TEXT, text)

    def test_blocks_to_text_orders_and_filters(self):
        blocks = [
            make_block("b", top=10.0, left=50.0),
            make_block("low", top=0.0, left=0.0, confidence=0.1),
            make_block("a", top=0.0, left=0.0),
        ]
        self.assertEqual(blocks_to_text(blocks, 0.5), "a\nb")


if __name__ == "__main__":
    unittest.main()
