"""Unit tests for the RapidOCR wrapper and block serialization."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from pdf2md.ocr import OcrEngine, TextBlock, blocks_from_json, blocks_to_json


class _FakeResult:
    def __init__(self, boxes=None, txts=None, scores=None):
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


class _FakeRapidOCR:
    result = _FakeResult()
    params = None

    def __init__(self, params=None):
        type(self).params = params

    def __call__(self, image):
        return type(self).result


def fake_rapidocr_module() -> dict:
    module = types.ModuleType("rapidocr")
    module.RapidOCR = _FakeRapidOCR
    return {"rapidocr": module}


class OcrEngineTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)

    def test_maps_boxes_texts_and_scores(self):
        _FakeRapidOCR.result = _FakeResult(
            boxes=[np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=float)],
            txts=("hello",),
            scores=(0.95,),
        )
        with mock.patch.dict(sys.modules, fake_rapidocr_module()):
            blocks = OcrEngine().recognize(self.image)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "hello")
        self.assertAlmostEqual(blocks[0].confidence, 0.95)
        self.assertEqual(blocks[0].height, 20.0)
        self.assertEqual(blocks[0].left, 0.0)

    def test_empty_result(self):
        _FakeRapidOCR.result = _FakeResult()
        with mock.patch.dict(sys.modules, fake_rapidocr_module()):
            blocks = OcrEngine().recognize(self.image)
        self.assertEqual(blocks, [])

    def test_missing_scores_default_to_one(self):
        _FakeRapidOCR.result = _FakeResult(
            boxes=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)],
            txts=("x",),
            scores=None,
        )
        with mock.patch.dict(sys.modules, fake_rapidocr_module()):
            blocks = OcrEngine().recognize(self.image)
        self.assertEqual(blocks[0].confidence, 1.0)

    def test_engine_disables_orientation_classifier(self):
        with mock.patch.dict(sys.modules, fake_rapidocr_module()):
            OcrEngine()
        self.assertEqual(
            _FakeRapidOCR.params,
            {"Global.log_level": "warning", "Global.use_cls": False},
        )


class SerializationTests(unittest.TestCase):
    def test_json_roundtrip(self):
        block = TextBlock(text="证据", confidence=0.87, bbox=[[0, 0], [1, 0], [1, 2], [0, 2]])
        loaded = blocks_from_json(blocks_to_json([block]))
        self.assertEqual(loaded, [block])


if __name__ == "__main__":
    unittest.main()
