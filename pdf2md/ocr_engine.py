"""
PaddleOCR PP-OCRv6_small engine via ONNX Runtime (CPU-only).

Decision (2026-08-12): run on ONNX Runtime with pure CPU. On this machine
``engine="onnxruntime"`` measured ~1.0 s/page vs ~3.0 s/page for the paddle
engine (200 dpi A4), with identical block counts. GPU/TensorRT/HPD-Parsing
are intentionally out of scope.

Models are cached under the project-local ``.paddlex_cache`` directory so no
user-home downloads are needed. First run downloads ONNX det/rec weights.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Keep PaddleX models in a project-local cache instead of the user home.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".paddlex_cache"))
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

log = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A single OCR-detected text region."""

    bbox: list[list[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text: str
    confidence: float

    @property
    def top(self) -> float:
        return min(p[1] for p in self.bbox)

    @property
    def bottom(self) -> float:
        return max(p[1] for p in self.bbox)

    @property
    def left(self) -> float:
        return min(p[0] for p in self.bbox)

    @property
    def height(self) -> float:
        return self.bottom - self.top


def _pdf_to_images(pdf_path: str | Path, dpi: int = 200):
    """Yield PDF pages as PIL Images using pypdfium2, one at a time."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72.0)
            yield bitmap.to_pil()
    finally:
        pdf.close()


class PaddleEngine:
    """PP-OCRv6_small OCR via PaddleOCR."""

    def __init__(self, lang: str = "ch"):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            engine="onnxruntime",
            lang=lang,
        )
        log.info("Initialized PaddleOCR (PP-OCRv6_small, ONNX Runtime / CPU)")

    def recognize_image(self, img) -> list[TextBlock]:
        """Run OCR on one rendered page image."""
        results = self.ocr.predict(np.array(img))
        blocks: list[TextBlock] = []
        if not results:
            return blocks
        res = results[0]
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        polys = res.get("rec_polys")
        if polys is None:
            polys = res.get("dt_polys") or []
        for poly, text, score in zip(polys, texts, scores):
            bbox = [[float(pt[0]), float(pt[1])] for pt in poly]
            blocks.append(TextBlock(bbox=bbox, text=str(text), confidence=float(score)))
        return blocks

    def recognize(self, file_path: str | Path) -> list[list[TextBlock]]:
        """Run OCR on a PDF or image; returns one list of TextBlock per page."""
        file_path = Path(file_path)
        if file_path.suffix.lower() != ".pdf":
            return [self.recognize_image(file_path)]
        pages: list[list[TextBlock]] = []
        for page_number, img in enumerate(_pdf_to_images(file_path), start=1):
            try:
                pages.append(self.recognize_image(img))
            finally:
                img.close()
            if page_number % 10 == 0 or page_number == 1:
                print(f"      OCR page {page_number}...", flush=True)
        return pages
