"""Text recognition with RapidOCR (PP-OCRv6_small, ONNX Runtime).

This is the only OCR backend: it runs the same PP-OCRv6_small det/rec ONNX
models on CPU, with no PaddlePaddle and no PyTorch dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class TextBlock:
    """A single OCR-detected text region."""

    text: str
    confidence: float
    bbox: list[list[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

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

    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": self.confidence, "box": self.bbox}

    @classmethod
    def from_dict(cls, data: dict) -> "TextBlock":
        return cls(
            text=str(data["text"]),
            confidence=float(data["confidence"]),
            bbox=[[float(p[0]), float(p[1])] for p in data["box"]],
        )


def blocks_to_json(blocks: list[TextBlock]) -> str:
    return json.dumps([b.to_dict() for b in blocks], ensure_ascii=False)


def blocks_from_json(text: str) -> list[TextBlock]:
    return [TextBlock.from_dict(item) for item in json.loads(text)]


class OcrEngine:
    """RapidOCR wrapper: one loaded model set, one call per page."""

    def __init__(self):
        from rapidocr import RapidOCR

        # Textline orientation classification is off (the models are trained
        # for upright scans); INFO logging would print one line per model.
        self._ocr = RapidOCR(
            params={"Global.log_level": "warning", "Global.use_cls": False}
        )

    def recognize(self, image) -> list[TextBlock]:
        result = self._ocr(np.array(image))
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        if boxes is None or not texts:
            return []

        scores = getattr(result, "scores", None)
        if scores is None:
            scores = [1.0] * len(texts)

        return [
            TextBlock(
                text=str(text),
                confidence=float(score),
                bbox=[[float(pt[0]), float(pt[1])] for pt in poly],
            )
            for poly, text, score in zip(boxes, texts, scores)
        ]
