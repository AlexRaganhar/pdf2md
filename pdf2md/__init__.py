"""
pdf2md - scanned PDF to Markdown converter.

The pipeline is deliberately small: render pages with pypdfium2, recognize
text with RapidOCR (PP-OCRv6_small on ONNX Runtime), then group the text into
paragraphs/headings and write Markdown plus page-numbered text.
"""

from .ocr import TextBlock
from .pipeline import Result, convert_pdf

__all__ = ["convert_pdf", "Result", "TextBlock"]
