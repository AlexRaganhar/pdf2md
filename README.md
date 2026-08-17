# pdf2md

Convert scanned PDFs to Markdown using **PP-OCRv6_small** (PaddleOCR) + **Docling**.

- Renders scanned pages with pypdfium2 and recognizes text with PP-OCRv6_small (ONNX Runtime, CPU-only).
- Rebuilds page structure (paragraph grouping, heading detection) before feeding Docling, so output keeps reading order.
- Incremental, resumable processing: finished pages and Docling chunks are checkpointed to disk; interrupted runs continue where they stopped.
- Optional page-numbered plain-text output for locating any line of text in the original PDF.

## Workflow

```
Scanned PDF
  |
  v
pypdfium2 (render pages to images)
  |
  v
PP-OCRv6_small (PaddleOCR text recognition, ONNX CPU)
  |
  v
HTML Builder (sort by position, group paragraphs, detect headings)
  |
  v
Docling (HTML -> Markdown)
  |
  v
Output .md file (+ optional page-numbered .txt)
```

## Install

Requires Python 3.9 - 3.12.

```bash
# 1. PaddlePaddle (CPU) from the official index
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 2. This project (editable install also registers the `pdf2md` command)
python -m pip install -e .

# or, without a full install, just the dependencies:
python -m pip install -r requirements.txt
```

The PP-OCRv6_small models are downloaded automatically on the first run and
cached in the project-local `.paddlex_cache/` directory (no user-home writes).

## Usage

```bash
pdf2md input.pdf                     # -> input.md
pdf2md input.pdf -o output/result.md
pdf2md input.pdf --txt               # also save <stem>.pages.txt (page-numbered text)
pdf2md input.pdf --no-docling
pdf2md input.pdf --chunk-size 200
pdf2md input.pdf --max-pages 50      # trial run: first 50 pages only
pdf2md input.pdf --no-resume         # ignore checkpoints, start fresh
```

`python -m pdf2md input.pdf` and `python main.py input.pdf` are equivalent
entry points that work without installing the package.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | -- | Input PDF file path |
| `-o, --output` | `<input>.md` | Output Markdown file path |
| `--lang` | `ch` | OCR language code |
| `--threshold` | `0.5` | Confidence filter (0-1) |
| `--no-docling` | off | Skip Docling, output raw OCR text |
| `--txt` | off | Also save page-numbered plain OCR text as `<stem>.pages.txt` |
| `--workdir` | `<output>.work` | Checkpoint directory (per-page text/JSON, chunks, progress) |
| `--chunk-size` | `100` | Pages per Docling chunk |
| `--no-resume` | off | Ignore existing checkpoints and reprocess everything |
| `--max-pages` | all | Only process the first N pages (trial runs) |

The page-numbered text (`*.pages.txt`) marks every PDF page explicitly
(`【第 N 页 / 共 M 页】`) so document review can locate each line of text in
the original scanned PDF. It is produced in the same OCR pass as the Markdown
output.

## Batch conversion

```bash
# Convert every PDF under <src> (recursively) into <src>/markdown_output
python batch_convert.py <src>

# Explicit output folder, page-numbered text, and a run log
python batch_convert.py <src> <dst> --txt --log batch.log

# Text only (skip Docling)
python batch_convert.py <src> <dst> --no-docling --txt

# Preview the jobs without converting (and without loading OCR dependencies)
python batch_convert.py <src> --dry-run
```

- Output keeps the source folder layout under the destination directory.
- Every PDF is converted independently with its own checkpoint workdir
  (`<dst>/<stem>.work`); interrupted runs resume from where they stopped.
- Encrypted PDFs are decrypted on the fly with `--password`; a password-free
  copy is kept in each per-file `.work` directory so runs stay resumable and
  the original files are never modified. Without `--password`, encrypted PDFs
  are reported as failures and the batch continues.
- Per-file failures are reported without stopping the batch.

## Large files and interruption

Processing is incremental and resumable. Every OCR'd page is written to the
work directory immediately (`pages/page_XXXX.txt` + `.json` plus
`progress.json`), and Docling converts the document in page chunks
(`chunks/chunk_XXX_pXXXX-XXXX.md`). If a run is interrupted (crash, power
loss, closed window), simply run the same command again: finished pages and
chunks are skipped and work continues from the checkpoint instead of
restarting from page 1.

## Project Structure

```
pdf2md/
|-- pyproject.toml        # packaging metadata + `pdf2md` console script
|-- requirements.txt
|-- LICENSE
|-- README.md
|-- main.py               # thin launcher for `python main.py`
|-- batch_convert.py      # recursive batch converter
|-- pdf2md/
|   |-- __init__.py
|   |-- cli.py            # CLI entry point
|   |-- __main__.py       # enables `python -m pdf2md`
|   |-- ocr_engine.py     # PP-OCRv6_small + pypdfium2 PDF rendering
|   |-- html_builder.py   # OCR results -> structured HTML
|   |-- incremental.py    # page-level checkpoints + chunked Docling
|   `-- converter.py      # main pipeline
`-- output/               # generated files (gitignored)
```

## Python API

```python
from pdf2md.converter import PdfToMarkdownConverter

converter = PdfToMarkdownConverter(lang="ch")
markdown = converter.convert("scan.pdf")
```

## License

[MIT](LICENSE)