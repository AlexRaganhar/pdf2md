"""Unit tests for the resumable conversion pipeline (OCR/pdfium faked)."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from pdf2md import pipeline
from pdf2md.ocr import TextBlock


def make_block(text="text"):
    return TextBlock(
        text=text,
        confidence=0.9,
        bbox=[[0, 0], [10, 0], [10, 20], [0, 20]],
    )


class _Image:
    def close(self):
        pass


class _Bitmap:
    def to_pil(self):
        return _Image()


class _Page:
    def render(self, scale):
        return _Bitmap()


class _PdfDocument:
    page_count = 1

    def __init__(self, path):
        pass

    def __len__(self):
        return type(self).page_count

    def __getitem__(self, index):
        return _Page()

    def close(self):
        pass


def fake_pdfium(page_count: int) -> dict:
    module = types.ModuleType("pypdfium2")
    module.PdfDocument = type("_Doc", (_PdfDocument,), {"page_count": page_count})
    return {"pypdfium2": module}


class _FakeEngine:
    calls = 0
    fail_on = frozenset()

    def __init__(self):
        type(self).calls = 0

    def recognize(self, image):
        type(self).calls += 1
        if type(self).calls in type(self).fail_on:
            raise RuntimeError("boom")
        return [make_block()]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdf = self.root / "doc.pdf"
        self.pdf.write_bytes(b"%PDF-1.4")
        self.out = self.root / "doc.md"
        self.work = self.root / "doc.work"

    def run_convert(self, pages, fail_on=frozenset(), **kwargs):
        _FakeEngine.calls = 0
        _FakeEngine.fail_on = fail_on
        kwargs.setdefault("log", lambda message: None)
        with mock.patch.dict(sys.modules, fake_pdfium(pages)), mock.patch.object(
            pipeline, "OcrEngine", _FakeEngine
        ):
            return pipeline.convert_pdf(self.pdf, self.out, **kwargs)

    def test_conversion_writes_both_outputs_and_removes_checkpoint(self):
        result = self.run_convert(2)

        self.assertEqual(result.failed_pages, [])
        self.assertEqual(result.total_pages, 2)
        self.assertEqual(result.processed_pages, 2)
        self.assertFalse(self.work.exists())

        markdown = self.out.read_text(encoding="utf-8")
        self.assertIn("<!-- Page 1 -->", markdown)
        self.assertIn("<!-- Page 2 -->", markdown)
        page_text = self.out.with_suffix(".pages.txt").read_text(encoding="utf-8")
        self.assertIn("【第 1 页 / 共 2 页】", page_text)
        self.assertIn("【第 2 页 / 共 2 页】", page_text)

    def test_resume_skips_pages_with_a_checkpoint(self):
        self.run_convert(2, max_pages=1)
        self.assertTrue((self.work / "pages" / "page_0001.json").exists())
        self.assertTrue(self.work.exists())

        result = self.run_convert(2)

        self.assertEqual(_FakeEngine.calls, 1)  # only page 2 was OCR'd
        self.assertEqual(result.processed_pages, 1)
        self.assertFalse(self.work.exists())

    def test_changed_source_discards_the_old_checkpoint(self):
        self.run_convert(2, max_pages=1)
        self.pdf.write_bytes(b"%PDF-1.4 changed")  # different size -> fingerprint

        result = self.run_convert(2)

        self.assertEqual(_FakeEngine.calls, 2)  # page 1 was not reused
        self.assertEqual(result.processed_pages, 2)
        self.assertEqual(result.failed_pages, [])

    def test_failed_page_is_marked_and_retried(self):
        result = self.run_convert(2, fail_on=frozenset({1}))

        self.assertEqual(result.failed_pages, [1])
        self.assertIn("（第 1 页 OCR 失败）", self.out.read_text(encoding="utf-8"))
        self.assertTrue(self.work.exists())
        self.assertFalse((self.work / "pages" / "page_0001.json").exists())
        self.assertTrue((self.work / "pages" / "page_0002.json").exists())
        self.assertEqual(list(self.work.rglob("*.tmp")), [])

        result = self.run_convert(2)  # transient failure is gone

        self.assertEqual(result.failed_pages, [])
        self.assertEqual(_FakeEngine.calls, 1)  # only the failed page
        self.assertFalse(self.work.exists())

    def test_unreadable_page_checkpoint_is_re_ocrd(self):
        self.run_convert(2, max_pages=1)
        (self.work / "pages" / "page_0001.json").write_text("{broken", encoding="utf-8")

        result = self.run_convert(2)

        self.assertEqual(result.failed_pages, [])
        self.assertEqual(_FakeEngine.calls, 2)  # corrupt page 1 + new page 2
        self.assertFalse(self.work.exists())

    def test_max_pages_keeps_the_checkpoint_for_a_trial_run(self):
        self.run_convert(2, max_pages=1)
        self.assertTrue(self.work.exists())
        self.assertEqual(self.out.with_suffix(".pages.txt").exists(), True)

    def test_no_resume_reprocesses_checkpointed_pages(self):
        self.run_convert(2, max_pages=1)
        result = self.run_convert(2, resume=False)
        self.assertEqual(_FakeEngine.calls, 2)
        self.assertEqual(result.processed_pages, 2)

    def test_foreign_work_dir_is_never_deleted_by_no_resume(self):
        self.work.mkdir()
        (self.work / "keep.txt").write_text("important", encoding="utf-8")

        with self.assertRaises(RuntimeError):
            self.run_convert(1, resume=False)

        self.assertTrue((self.work / "keep.txt").exists())

    def test_foreign_work_dir_is_refused(self):
        self.work.mkdir()
        (self.work / "keep.txt").write_text("important", encoding="utf-8")

        with self.assertRaises(RuntimeError):
            self.run_convert(1)

        self.assertTrue((self.work / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
