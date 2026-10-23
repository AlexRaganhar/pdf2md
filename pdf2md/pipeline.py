"""The single conversion pipeline: PDF -> OCR -> Markdown, resumable by page.

There is no second code path: every conversion (CLI, batch, Python API) goes
through :func:`convert_pdf`. Each finished page is checkpointed as JSON; a
missing checkpoint simply means "OCR this page again", which removes the need
for a progress database entirely.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .markdown import page_numbered_text, pages_to_markdown
from .ocr import OcrEngine, TextBlock, blocks_from_json, blocks_to_json

PAGE_DIR = "pages"
SOURCE_FILE = "source.json"
PAGE_DPI = 200


@dataclass
class Result:
    markdown: str
    page_text: str
    failed_pages: list[int]
    total_pages: int
    processed_pages: int
    output_path: Path
    work_dir: Path


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _write_atomic(path: Path, text: str) -> None:
    """Write text so an interrupted run never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_pages(
    pages_dir: Path, target: int, log
) -> list[list[TextBlock] | None]:
    """Load page checkpoints; ``None`` means the page still needs OCR."""
    pages: list[list[TextBlock] | None] = []
    for page_no in range(1, target + 1):
        path = pages_dir / f"page_{page_no:04d}.json"
        if not path.exists():
            pages.append(None)
            continue
        try:
            pages.append(blocks_from_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log(f"  page {page_no}: unreadable checkpoint ({exc}); will re-OCR")
            pages.append(None)
    return pages


def _is_checkpoint_dir(work_dir: Path) -> bool:
    return (work_dir / SOURCE_FILE).exists() or (work_dir / PAGE_DIR).is_dir()


def convert_pdf(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    source_path: str | Path | None = None,
    threshold: float = 0.5,
    max_pages: int | None = None,
    resume: bool = True,
    log=print,
) -> Result:
    """Convert one PDF; page-level checkpoints live in ``<output>.work``.

    ``source_path`` is the file used for the checkpoint fingerprint when
    ``pdf_path`` is a temporary decrypted copy. Failed pages are left
    unchecked (no JSON) so the next run retries exactly those pages.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    source_path = Path(source_path) if source_path is not None else pdf_path
    work_dir = output_path.with_name(output_path.stem + ".work")
    pages_dir = work_dir / PAGE_DIR
    fingerprint = _fingerprint(source_path)

    if work_dir.exists() and not work_dir.is_dir():
        raise RuntimeError(f"refusing to use {work_dir}: it is not a directory")

    if work_dir.is_dir():
        if not _is_checkpoint_dir(work_dir):
            if any(work_dir.iterdir()):
                raise RuntimeError(
                    f"refusing to use {work_dir}: it exists, is not empty, and "
                    "is not a pdf2md checkpoint directory"
                )
        elif not resume:
            shutil.rmtree(work_dir)
        else:
            try:
                previous = json.loads(
                    (work_dir / SOURCE_FILE).read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                previous = None
            if previous is not None and previous != fingerprint:
                log("  source file changed; discarding the old checkpoint")
                shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(exist_ok=True)
    if not (work_dir / SOURCE_FILE).exists():
        _write_atomic(work_dir / SOURCE_FILE, json.dumps(fingerprint))

    import pypdfium2 as pdfium

    processed = 0
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        total = len(pdf)
        target = min(max_pages, total) if max_pages else total
        pages = _load_pages(pages_dir, target, log)
        pending = [page_no for page_no, blocks in enumerate(pages, start=1) if blocks is None]
        engine = OcrEngine() if pending else None

        for page_no in pending:
            image = None
            try:
                image = pdf[page_no - 1].render(scale=PAGE_DPI / 72.0).to_pil()
                blocks = engine.recognize(image)
            except Exception as exc:  # noqa: BLE001 - one bad page must not abort the run
                log(f"  page {page_no} failed: {exc}; will retry next run")
                continue
            finally:
                if image is not None:
                    image.close()

            _write_atomic(
                pages_dir / f"page_{page_no:04d}.json", blocks_to_json(blocks)
            )
            pages[page_no - 1] = blocks
            processed += 1
            if processed == 1 or processed % 5 == 0:
                log(f"  {page_no}/{target} pages  ({time.strftime('%H:%M:%S')})")
    finally:
        pdf.close()

    failed = [page_no for page_no, blocks in enumerate(pages, start=1) if blocks is None]
    failed_set = frozenset(failed)
    render_pages = [blocks or [] for blocks in pages]

    markdown = pages_to_markdown(render_pages, threshold, failed_set)
    page_text = page_numbered_text(render_pages, total, threshold, failed_set)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_path, markdown)
    _write_atomic(output_path.with_suffix(".pages.txt"), page_text)

    # The checkpoint only exists to resume an unfinished run. A --max-pages
    # trial keeps it so the full run can resume from it.
    if max_pages is None and not failed:
        shutil.rmtree(work_dir, ignore_errors=True)

    return Result(
        markdown=markdown,
        page_text=page_text,
        failed_pages=failed,
        total_pages=total,
        processed_pages=processed,
        output_path=output_path,
        work_dir=work_dir,
    )
