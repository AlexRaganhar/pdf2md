"""Batch-convert scanned PDFs in a folder to Markdown, recursively.

Every PDF gets its own ``.work`` checkpoint directory next to the output, so
an interrupted batch run resumes from where it stopped when re-run. Output
Markdown keeps the source relative folder layout under the destination
directory (default: ``<src>/markdown_output``).
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_NAME = "markdown_output"


def _configure_console() -> None:
    """Make status output readable when Python inherits a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            # Redirected/non-standard streams may not allow reconfiguration.
            pass


def _module_available(module_name: str) -> bool:
    """Check a dependency without importing its (potentially heavy) runtime."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _missing_dependencies(use_docling: bool) -> list[str]:
    required = {
        "pypdfium2": "pypdfium2",
        "paddleocr": "paddleocr",
        "paddle": "paddlepaddle",
        "pypdf": "pypdf",
    }
    if use_docling:
        required["docling"] = "docling"
    return [package for module, package in required.items() if not _module_available(module)]


def _project_python() -> Path | None:
    """Return the bundled project interpreter, if one exists."""
    candidates = [
        Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe",
        Path(__file__).resolve().parent / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _maybe_relaunch_in_project_env(argv: list[str] | None = None) -> int | None:
    """Relaunch with the project venv when the caller used a bare Python."""
    requested_args = list(sys.argv[1:] if argv is None else argv)
    if "--dry-run" in requested_args or "--help" in requested_args or "-h" in requested_args:
        return None
    if os.environ.get("PDF2MD_AUTO_RELAUNCHED") == "1":
        return None

    use_docling = "--no-docling" not in requested_args
    if not _missing_dependencies(use_docling):
        return None

    project_python = _project_python()
    if project_python is None:
        return None
    try:
        current_python = Path(sys.executable).resolve()
        project_python = project_python.resolve()
    except OSError:
        return None
    if current_python == project_python:
        return None

    print(
        "[INFO] Current Python is missing pdf2md dependencies; "
        f"relaunching with {project_python}",
        flush=True,
    )
    env = os.environ.copy()
    env["PDF2MD_AUTO_RELAUNCHED"] = "1"
    completed = subprocess.run(
        [str(project_python), str(Path(__file__).resolve()), *requested_args],
        env=env,
    )
    return completed.returncode


def _print_dependency_error(missing: list[str]) -> None:
    joined = ", ".join(missing)
    project_python = _project_python()
    print(
        f"Error: missing Python package(s): {joined}\n"
        f"Current interpreter: {sys.executable}",
        file=sys.stderr,
    )
    if project_python is not None:
        print(
            "Use the project environment, for example:\n"
            f'  & "{project_python}" "{Path(__file__).resolve()}"',
            file=sys.stderr,
        )
    else:
        print(
            "Install the project dependencies with:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt',
            file=sys.stderr,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively convert scanned PDFs under a source folder to Markdown. "
            "Output preserves the source relative layout under <dst>."
        )
    )
    parser.add_argument("src", type=Path, help="Source folder containing the PDFs.")
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=None,
        help=f"Output folder (default: <src>/{DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument("--pattern", default="*.pdf", help="Filename pattern, matched case-insensitively (default: *.pdf).")
    parser.add_argument("--password", default=None, help="Password for encrypted PDFs (optional).")
    parser.add_argument("--lang", default="ch", help="PaddleOCR language code (default: ch).")
    parser.add_argument("--threshold", type=float, default=0.5, help="OCR confidence threshold (default: 0.5).")
    parser.add_argument("--txt", action="store_true", help="Also save page-numbered plain-text OCR output per PDF.")
    parser.add_argument("--no-docling", action="store_true", help="Skip Docling and output raw OCR Markdown.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Pages per Docling chunk (default: 100).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoints and reprocess everything.")
    parser.add_argument("--max-pages", type=int, default=None, help="Only process the first N pages per PDF (trial runs).")
    parser.add_argument("--dry-run", action="store_true", help="Only list the jobs to be run, without converting.")
    parser.add_argument("--log", type=Path, default=None, help="Append a run log to this file.")
    return parser.parse_args(argv)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_jobs(src: Path, dst: Path, pattern: str) -> list[tuple[Path, Path]]:
    """Find PDFs below ``src`` and map each to its output path under ``dst``."""
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    exclude_output = _is_relative_to(dst_resolved, src_resolved)

    jobs: list[tuple[Path, Path]] = []
    for path in src.rglob("*"):
        if not path.is_file() or not fnmatch.fnmatch(path.name.lower(), pattern.lower()):
            continue
        if exclude_output and _is_relative_to(path.resolve(), dst_resolved):
            continue
        relative = path.relative_to(src_resolved)
        output = dst_resolved / relative.with_suffix(".md")
        jobs.append((path, output))
    return sorted(jobs, key=lambda pair: str(pair[0]).lower())


def _ensure_readable_source(pdf: Path, work_dir: Path, password: str | None) -> Path:
    """Return a decrypted, password-free copy that the OCR pipeline can open."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError("pypdf is required to open password-protected PDFs") from exc

    reader = PdfReader(str(pdf))
    if not reader.is_encrypted:
        return pdf

    if not password:
        raise RuntimeError(
            f"{pdf.name} is password-protected; pass --password to unlock it"
        )

    decrypted = work_dir / "decrypted_source.pdf"
    if decrypted.exists():
        try:
            check = PdfReader(str(decrypted))
            if not check.is_encrypted:
                return decrypted
        except Exception:
            pass
        decrypted.unlink()

    if not reader.decrypt(password):
        raise RuntimeError(
            f"{pdf.name} is password-protected and password {password!r} "
            "did not unlock it"
        )

    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    try:
        if reader.metadata:
            writer.add_metadata(reader.metadata)
    except Exception:
        pass

    work_dir.mkdir(parents=True, exist_ok=True)
    with decrypted.open("wb") as handle:
        writer.write(handle)

    try:
        check = PdfReader(str(decrypted))
        if check.is_encrypted:
            raise RuntimeError("decrypted copy is still encrypted")
    except Exception:
        decrypted.unlink()
        raise
    return decrypted


def _write_log(message: str, log_path: Path | None) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        print(f"[WARN] Could not write log {log_path}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    relaunch_code = _maybe_relaunch_in_project_env(argv)
    if relaunch_code is not None:
        return relaunch_code
    args = parse_args(argv)

    src = args.src
    dst = args.dst or src / DEFAULT_OUTPUT_NAME
    if not src.is_dir():
        print(f"Error: source folder not found -- {src}", file=sys.stderr)
        return 1
    if src.resolve() == dst.resolve():
        print(
            "Error: output folder must be different from the source folder "
            f"(use the default {DEFAULT_OUTPUT_NAME} folder or pass another path).",
            file=sys.stderr,
        )
        return 1
    if not 0 <= args.threshold <= 1:
        print("Error: --threshold must be between 0 and 1.", file=sys.stderr)
        return 1
    if args.chunk_size <= 0:
        print("Error: --chunk-size must be greater than 0.", file=sys.stderr)
        return 1
    if args.max_pages is not None and args.max_pages <= 0:
        print("Error: --max-pages must be greater than 0.", file=sys.stderr)
        return 1

    jobs = discover_jobs(src, dst, args.pattern)
    if not jobs:
        print(f"Error: no files matching {args.pattern!r} under {src}", file=sys.stderr)
        return 1

    print(f"Found {len(jobs)} PDF(s) under {src}")
    if args.dry_run:
        for index, (pdf, output) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {pdf} -> {output}")
        return 0

    missing = _missing_dependencies(use_docling=not args.no_docling)
    if missing:
        _print_dependency_error(missing)
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    log_path = args.log or dst / "batch_convert.log"
    _write_log(
        f"Batch start: {len(jobs)} file(s); source={src}; output={dst}",
        log_path,
    )

    # Import lazily so --dry-run remains useful even when OCR/Docling
    # dependencies are not installed yet.
    try:
        from pdf2md.converter import PdfToMarkdownConverter
    except Exception as exc:  # pragma: no cover - depends on local install
        _write_log(f"Could not load pdf2md dependencies: {exc}", log_path)
        return 1

    converter = PdfToMarkdownConverter(
        lang=args.lang,
        confidence_threshold=args.threshold,
    )
    succeeded = 0
    failed = 0
    for index, (pdf, output) in enumerate(jobs, 1):
        output.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output.with_name(f"{output.stem}.work")
        txt_path = output.with_suffix(".pages.txt") if args.txt else None
        _write_log(f"[{index}/{len(jobs)}] {pdf}", log_path)
        try:
            source = _ensure_readable_source(pdf, work_dir, args.password)
            converter.convert_incremental(
                source,
                output,
                work_dir=work_dir,
                chunk_size=args.chunk_size,
                use_docling=not args.no_docling,
                resume=not args.no_resume,
                txt_path=txt_path,
                max_pages=args.max_pages,
            )
        except Exception as exc:
            failed += 1
            _write_log(f"  FAILED: {exc}", log_path)
        else:
            succeeded += 1
            _write_log(f"  OK: {output}", log_path)

    _write_log(
        f"Batch done: OK={succeeded} FAILED={failed} TOTAL={len(jobs)}",
        log_path,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())