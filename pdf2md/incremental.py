"""
Incremental, resumable processing for large scanned PDFs.

Every completed page is written to disk immediately, so an interrupted run
never loses finished work and a rerun continues from the checkpoint instead
of starting over. Docling runs in page chunks so a failure only wastes one
chunk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .html_builder import build_html
from .ocr_engine import PaddleEngine, TextBlock

PAGE_DIR = "pages"
CHUNK_DIR = "chunks"
STATE_FILE = "progress.json"


def _write_state(state: dict, state_path: Path) -> None:
    """Atomically persist the checkpoint (write temp, then replace)."""
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(state_path)


def _load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _blocks_to_text(blocks: list[TextBlock], threshold: float) -> str:
    ordered = sorted(
        (b for b in blocks if b.confidence >= threshold),
        key=lambda b: (b.top, b.left),
    )
    return "\n".join(b.text for b in ordered).strip() or "（本页未识别到文字）"


def _blocks_to_json(blocks: list[TextBlock]) -> str:
    return json.dumps(
        [{"box": b.bbox, "text": b.text, "confidence": b.confidence} for b in blocks],
        ensure_ascii=False,
    )


def _load_pages(pages_dir: Path, total: int) -> list[list[TextBlock]]:
    pages: list[list[TextBlock]] = []
    for page_no in range(1, total + 1):
        jp = pages_dir / f"page_{page_no:04d}.json"
        if not jp.exists():
            pages.append([])
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            pages.append(
                [
                    TextBlock(bbox=b["box"], text=b["text"], confidence=float(b["confidence"]))
                    for b in data
                ]
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pages.append([])
    return pages


def _progress(page_no: int, total: int, start: float, max_pages: int | None) -> None:
    elapsed = time.time() - start
    target = max_pages or total
    eta = elapsed / page_no * (target - page_no) if page_no else 0.0
    print(
        f"      {page_no}/{target} pages  "
        f"{elapsed:5.0f}s elapsed  ~{eta:5.0f}s remaining  ({time.strftime('%H:%M:%S')})",
        flush=True,
    )


def ocr_incremental(
    pdf_path: str | Path,
    work_dir: str | Path,
    lang: str = "ch",
    threshold: float = 0.5,
    resume: bool = True,
    max_pages: int | None = None,
) -> list[list[TextBlock]]:
    """OCR a scanned PDF page-by-page, saving every page immediately.

    Returns one list of TextBlock per page (loaded back from disk). Pages
    that fail OCR are recorded in the checkpoint and retried on the next
    run; they come back as empty lists here.
    """
    work_dir = Path(work_dir)
    pages_dir = work_dir / PAGE_DIR
    pages_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / STATE_FILE

    state = _load_state(state_path) if resume else {}
    done = set(state.get("done", []))
    failed = set(state.get("failed", []))

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    total = state.get("total") or len(pdf)
    target = min(max_pages, total) if max_pages else total
    engine = PaddleEngine(lang=lang)

    start = time.time()
    for page_no in range(1, target + 1):
        if page_no in done:
            continue
        img = None
        try:
            img = pdf[page_no - 1].render(scale=200 / 72.0).to_pil()
            blocks = engine.recognize_image(img)
            text = _blocks_to_text(blocks, threshold)
            (pages_dir / f"page_{page_no:04d}.txt").write_text(
                f"【第 {page_no} 页 / 共 {total} 页】\n{text}\n",
                encoding="utf-8",
            )
            (pages_dir / f"page_{page_no:04d}.json").write_text(
                _blocks_to_json(blocks),
                encoding="utf-8",
            )
            done.add(page_no)
            failed.discard(page_no)
            _write_state(
                {"total": total, "done": sorted(done), "failed": sorted(failed)},
                state_path,
            )
            if page_no == 1 or page_no % 5 == 0:
                _progress(page_no, total, start, max_pages)
        except Exception as exc:
            failed.add(page_no)
            _write_state(
                {"total": total, "done": sorted(done), "failed": sorted(failed)},
                state_path,
            )
            print(f"      page {page_no} FAILED ({exc}); will retry on next run", flush=True)
        finally:
            if img is not None:
                img.close()

    pdf.close()

    if failed:
        print(f"      WARNING: {len(failed)} page(s) failed OCR: {sorted(failed)}", flush=True)
    return _load_pages(pages_dir, target)


def docling_incremental(
    pages: list[list[TextBlock]],
    work_dir: str | Path,
    stem: str,
    threshold: float = 0.5,
    chunk_size: int = 100,
    resume: bool = True,
) -> str:
    """Convert per-page blocks to Markdown in page chunks via Docling.

    Each chunk is written to ``work_dir/chunks/`` with its page range in the
    filename, so a rerun skips exactly the chunks that already succeeded.
    Returns the combined Markdown of all chunks.
    """
    work_dir = Path(work_dir)
    chunks_dir = work_dir / CHUNK_DIR
    chunks_dir.mkdir(parents=True, exist_ok=True)

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    n_chunks = (len(pages) + chunk_size - 1) // chunk_size
    chunk_paths: list[Path] = []

    for ci in range(1, n_chunks + 1):
        start_i = (ci - 1) * chunk_size
        end_i = min(ci * chunk_size, len(pages))
        chunk_file = chunks_dir / f"chunk_{ci:03d}_p{start_i + 1:04d}-{end_i:04d}.md"

        if resume and chunk_file.exists() and chunk_file.stat().st_size > 0:
            chunk_paths.append(chunk_file)
            continue

        html = build_html(pages[start_i:end_i], threshold)
        try:
            result = converter.convert_string(
                content=html,
                format=InputFormat.HTML,
                name=f"{stem}-part{ci}",
            )
            md = result.document.export_to_markdown()
        except Exception as exc:
            print(f"      chunk {ci} FAILED ({exc}); using fallback text", flush=True)
            md = _fallback_from_html(html)

        chunk_file.write_text(md, encoding="utf-8")
        chunk_paths.append(chunk_file)
        print(
            f"      chunk {ci}/{n_chunks} done  ({time.strftime('%H:%M:%S')})",
            flush=True,
        )

    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in chunk_paths)


def _fallback_from_html(html: str) -> str:
    """Strip tags from generated HTML as a last resort."""
    import re

    text = re.sub(r"<[^>]+>", "", html)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
