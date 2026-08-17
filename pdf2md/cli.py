#!/usr/bin/env python3
"""CLI entry point for scanned-PDF-to-Markdown conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .converter import PdfToMarkdownConverter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert scanned PDFs to Markdown using PP-OCRv6_small + Docling.",
    )
    parser.add_argument("input", type=Path, help="Path to the scanned PDF file.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output Markdown file path (default: <input>.md).")
    parser.add_argument("--lang", default="ch", help="OCR language code (default: ch).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5).")
    parser.add_argument("--no-docling", action="store_true", help="Skip Docling and output raw OCR text.")
    parser.add_argument("--txt", action="store_true", help="Also save page-numbered plain-text OCR output (<stem>.pages.txt).")
    parser.add_argument("--workdir", type=Path, default=None, help="Checkpoint directory (default: <output>.work).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Pages per Docling chunk (default: 100).")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring existing checkpoints.")
    parser.add_argument("--max-pages", type=int, default=None, help="Only process the first N pages (trial runs).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"Error: file not found -- {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.input.with_suffix(".md")
    txt_path = output_path.with_suffix(".pages.txt") if args.txt else None

    converter = PdfToMarkdownConverter(
        lang=args.lang,
        confidence_threshold=args.threshold,
    )
    converter.convert_incremental(
        args.input,
        output_path,
        work_dir=args.workdir,
        chunk_size=args.chunk_size,
        use_docling=not args.no_docling,
        resume=not args.no_resume,
        txt_path=txt_path,
        max_pages=args.max_pages,
    )

    print(f"\nDone -- saved to {output_path}")
    if txt_path is not None:
        print(f"Page-numbered text saved to {txt_path}")


if __name__ == "__main__":
    main()