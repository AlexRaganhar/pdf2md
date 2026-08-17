"""
pdf2md - Scanned PDF to Markdown converter

Uses PP-OCRv6_small for text recognition and Docling for Markdown conversion.
"""

from .converter import PdfToMarkdownConverter

__all__ = ["PdfToMarkdownConverter"]
