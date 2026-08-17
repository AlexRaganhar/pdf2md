"""
Main conversion pipeline: PDF -> PP-OCRv6_small -> HTML -> Docling -> Markdown.
"""

from __future__ import annotations

from pathlib import Path

from .html_builder import build_html
from .incremental import docling_incremental, ocr_incremental
from .ocr_engine import PaddleEngine, TextBlock


class PdfToMarkdownConverter:
    """Orchestrates the scanned-PDF-to-Markdown pipeline."""

    def __init__(
        self,
        lang: str = "ch",
        confidence_threshold: float = 0.5,
    ):
        self.lang = lang
        self.confidence_threshold = confidence_threshold

    def convert(self, pdf_path: str | Path, use_docling: bool = True) -> str:
        """Convert a scanned PDF file to Markdown text."""
        markdown, _ = self.convert_with_pages(pdf_path, use_docling)
        return markdown

    def convert_with_pages(
        self,
        pdf_path: str | Path,
        use_docling: bool = True,
    ) -> tuple[str, str]:
        """Convert to Markdown; also return page-numbered plain OCR text.

        Both outputs come from a single OCR pass. The page-numbered text is
        meant for case-file review (阅卷), where each block of text must be
        locatable at a specific page of the original PDF.
        """
        pdf_path = Path(pdf_path)

        # Step 1 -- OCR
        print(f"[1/3] Running PP-OCRv6 OCR on: {pdf_path.name}")
        engine = PaddleEngine(lang=self.lang)
        pages = engine.recognize(pdf_path)
        total_blocks = sum(len(p) for p in pages)
        print(f"      {len(pages)} page(s), {total_blocks} text blocks detected.")

        # Step 2 -- Build HTML
        print("[2/3] Building structured HTML from OCR results...")
        html = build_html(pages, self.confidence_threshold)

        if use_docling:
            # Step 3 -- Docling -> Markdown
            print("[3/3] Converting to Markdown via Docling...")
            markdown = self._docling_convert(html, pdf_path.stem)
        else:
            print("[3/3] Skipping Docling, generating Markdown directly...")
            markdown = self._fallback_markdown(pages)

        return markdown, self._page_numbered_text(pages)

    def convert_incremental(
        self,
        pdf_path: str | Path,
        output_path: str | Path,
        work_dir: str | Path | None = None,
        chunk_size: int = 100,
        use_docling: bool = True,
        resume: bool = True,
        txt_path: str | Path | None = None,
        max_pages: int | None = None,
    ) -> tuple[str, str]:
        """Resumable conversion for large scanned PDFs.

        Per-page OCR text is saved to ``work_dir/pages/`` immediately, so an
        interrupted run never loses finished pages and a rerun continues from
        the checkpoint. Docling runs in page chunks (``work_dir/chunks/``).
        Returns ``(markdown, page_numbered_text)`` and writes both files.
        """
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)
        if work_dir is None:
            work_dir = output_path.with_name(output_path.stem + ".work")
        work_dir = Path(work_dir)

        print(f"[1/3] Running PP-OCRv6 OCR (incremental) on: {pdf_path.name}")
        pages = ocr_incremental(
            pdf_path,
            work_dir,
            lang=self.lang,
            threshold=self.confidence_threshold,
            resume=resume,
            max_pages=max_pages,
        )
        total_blocks = sum(len(p) for p in pages)
        print(f"      {len(pages)} page(s), {total_blocks} text blocks.")

        if use_docling:
            print("[2/3] Converting to Markdown via Docling (chunked)...")
            markdown = docling_incremental(
                pages,
                work_dir,
                output_path.stem,
                threshold=self.confidence_threshold,
                chunk_size=chunk_size,
                resume=resume,
            )
        else:
            print("[2/3] Skipping Docling, generating Markdown directly...")
            markdown = self._fallback_markdown(pages)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

        page_text = self._combined_page_text(work_dir, len(pages))
        if txt_path is not None:
            Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(txt_path).write_text(page_text, encoding="utf-8")
        return markdown, page_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _docling_convert(self, html: str, name: str) -> str:
        """Feed HTML into Docling and return Markdown."""
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()

        try:
            result = converter.convert_string(
                content=html,
                format=InputFormat.HTML,
                name=name,
            )
            return result.document.export_to_markdown()
        except Exception as exc:
            print(f"      [warn] Docling conversion failed ({exc}), using fallback.")
            return self._fallback_markdown_from_html(html)

    def _fallback_markdown(self, pages: list[list[TextBlock]]) -> str:
        """Produce plain Markdown directly from OCR blocks."""
        parts: list[str] = []
        for idx, blocks in enumerate(pages):
            filtered = [b for b in blocks if b.confidence >= self.confidence_threshold]
            filtered.sort(key=lambda b: (b.top, b.left))
            parts.append(f"<!-- Page {idx + 1} -->\n")
            for b in filtered:
                parts.append(b.text)
            parts.append("")
        return "\n".join(parts)

    def _page_numbered_text(self, pages: list[list[TextBlock]]) -> str:
        """Format per-page OCR text with explicit PDF page-number markers."""
        parts: list[str] = []
        total = len(pages)
        for idx, blocks in enumerate(pages, start=1):
            ordered = sorted(
                (b for b in blocks if b.confidence >= self.confidence_threshold),
                key=lambda b: (b.top, b.left),
            )
            body = "\n".join(b.text for b in ordered).strip()
            if not body:
                body = "（本页未识别到文字）"
            parts.append(f"【第 {idx} 页 / 共 {total} 页】\n{body}")
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _combined_page_text(work_dir: Path, total: int) -> str:
        """Concatenate per-page checkpoint text files in PDF page order."""
        pages_dir = work_dir / "pages"
        parts: list[str] = []
        for page_no in range(1, total + 1):
            p = pages_dir / f"page_{page_no:04d}.txt"
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
            else:
                parts.append(f"【第 {page_no} 页 / 共 {total} 页】\n（本页 OCR 失败）")
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _fallback_markdown_from_html(html: str) -> str:
        """Strip tags from the generated HTML as a last resort."""
        import re
        text = re.sub(r"<[^>]+>", "", html)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
