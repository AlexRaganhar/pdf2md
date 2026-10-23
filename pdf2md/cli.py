"""Command-line interface: one PDF or a folder of PDFs.

One command covers both shapes. A folder is converted recursively, keeping the
source layout under the output directory; each PDF gets its own page-level
checkpoint next to its Markdown file.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .pipeline import Result, convert_pdf

REQUIRED_MODULES = {
    "rapidocr": "rapidocr",
    "onnxruntime": "onnxruntime",
    "pypdfium2": "pypdfium2",
    "pypdf": "pypdf",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description=(
            "Convert scanned PDFs to Markdown (RapidOCR PP-OCRv6_small). "
            "Pass a PDF file or a folder to convert recursively."
        ),
    )
    parser.add_argument("input", type=Path, help="PDF file, or a folder of PDFs.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .md file (file input) or folder (folder input). "
        "Defaults: <input>.md or <input>/markdown_output.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="OCR confidence filter (0-1, default: 0.5).")
    parser.add_argument("--password", default=None, help="Password for encrypted PDFs (optional).")
    parser.add_argument("--max-pages", type=int, default=None, help="Only process the first N pages (trial runs).")
    parser.add_argument("--no-resume", action="store_true", help="Discard existing page checkpoints.")
    parser.add_argument("--pattern", default="*.pdf", help="Folder mode: filename pattern (default: *.pdf).")
    parser.add_argument("--log", type=Path, default=None, help="Folder mode: append a run log to this file.")
    parser.add_argument("--dry-run", action="store_true", help="Only list the jobs, convert nothing.")
    return parser


def _missing_dependencies() -> list[str]:
    missing = []
    for module, package in REQUIRED_MODULES.items():
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(package)
        except (ImportError, ValueError):
            missing.append(package)
    return missing


def _ensure_dependencies() -> bool:
    missing = _missing_dependencies()
    if not missing:
        return True
    print(
        "Error: missing Python package(s): "
        + ", ".join(missing)
        + "\nInstall them with: python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    return False


def _readable_source(path: Path, password: str | None) -> tuple[Path, Path | None]:
    """Return (path to open, temporary decrypted copy or ``None``)."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(path))
    if not reader.is_encrypted:
        return path, None

    if not password:
        raise RuntimeError(f"{path.name} is password-protected; pass --password")
    if not reader.decrypt(password):
        raise RuntimeError(f"{path.name}: the supplied password did not unlock it")

    fd, name = tempfile.mkstemp(prefix="pdf2md-", suffix=".pdf")
    os.close(fd)
    tmp = Path(name)
    try:
        writer = PdfWriter()
        writer.append_pages_from_reader(reader)
        with tmp.open("wb") as handle:
            writer.write(handle)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return tmp, tmp


def _convert_one(pdf: Path, output: Path, args: argparse.Namespace) -> Result:
    source, tmp = _readable_source(pdf, args.password)
    try:
        return convert_pdf(
            source,
            output,
            source_path=pdf,
            threshold=args.threshold,
            max_pages=args.max_pages,
            resume=not args.no_resume,
        )
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


class _Log:
    """Print to stdout and optionally append to a log file."""

    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        if self.path is not None:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] {message}\n")


def _discover(src: Path, dst: Path, pattern: str) -> list[tuple[Path, Path]]:
    """Map every matching PDF under ``src`` to its output path under ``dst``."""
    from fnmatch import fnmatch

    jobs: list[tuple[Path, Path]] = []
    for path in src.rglob("*"):
        if not path.is_file() or not fnmatch(path.name.lower(), pattern.lower()):
            continue
        if any(part.endswith(".work") for part in path.parts):
            continue
        if dst.is_relative_to(src) and path.is_relative_to(dst):
            continue
        jobs.append((path, dst / path.relative_to(src).with_suffix(".md")))
    return sorted(jobs, key=lambda pair: str(pair[0]).lower())


def _run_file(args: argparse.Namespace) -> int:
    pdf = args.input
    output = args.output or pdf.with_suffix(".md")
    if output.resolve() == pdf.resolve():
        print(f"Error: output path is the input file -- {pdf}", file=sys.stderr)
        return 2
    if output.suffix.lower() != ".md":
        output = output.with_suffix(".md")
    if output.is_dir():
        print(f"Error: output path is a directory -- {output}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"{pdf} -> {output}")
        return 0
    if not _ensure_dependencies():
        return 2

    try:
        result = _convert_one(pdf, output, args)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.pages.txt')}")
    if result.failed_pages:
        print(
            f"WARNING: {len(result.failed_pages)} page(s) failed OCR: "
            f"{result.failed_pages}; the checkpoint was kept, rerun to retry them.",
            file=sys.stderr,
        )
        return 1
    return 0


def _run_folder(args: argparse.Namespace) -> int:
    src = args.input.resolve()
    dst = (args.output or (args.input / "markdown_output")).resolve()
    if src == dst:
        print("Error: output folder must differ from the input folder.", file=sys.stderr)
        return 2
    if args.output is not None and args.output.exists() and not args.output.is_dir():
        print("Error: -o/--output must be a folder for folder input.", file=sys.stderr)
        return 2

    jobs = _discover(src, dst, args.pattern)
    if not jobs:
        print(f"Error: no files matching {args.pattern!r} under {src}", file=sys.stderr)
        return 1

    print(f"Found {len(jobs)} PDF(s) under {src}")
    if args.dry_run:
        for index, (pdf, output) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {pdf} -> {output}")
        return 0
    if not _ensure_dependencies():
        return 2

    log = _Log(args.log)
    done = failed = 0
    for index, (pdf, output) in enumerate(jobs, 1):
        output.parent.mkdir(parents=True, exist_ok=True)
        log(f"[{index}/{len(jobs)}] {pdf}")
        try:
            result = _convert_one(pdf, output, args)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            failed += 1
            log(f"  FAILED: {exc}")
            continue

        if result.failed_pages:
            failed += 1
            log(f"  PARTIAL: {len(result.failed_pages)} page(s) failed: {result.failed_pages}")
        else:
            done += 1
            log(f"  OK: {output}")

    log(f"Done: OK={done} FAILED={failed} TOTAL={len(jobs)}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be greater than 0")
    if not args.input.exists():
        print(f"Error: path not found -- {args.input}", file=sys.stderr)
        return 2

    if args.input.is_dir():
        return _run_folder(args)
    return _run_file(args)


if __name__ == "__main__":
    raise SystemExit(main())
