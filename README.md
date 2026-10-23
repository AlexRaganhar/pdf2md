# pdf2md

[![CI](https://github.com/AlexRaganhar/pdf2md/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexRaganhar/pdf2md/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

Scanned PDF → Markdown, locally, on CPU.

The project is deliberately one pipeline with one job:

- **One OCR engine** — RapidOCR running the PP-OCRv6_small ONNX det/rec models
  on ONNX Runtime. No PyTorch, no PaddlePaddle.
- **One output path** — paragraph/heading grouping straight to Markdown, plus
  a page-numbered `.pages.txt` for locating text in the original PDF.
- **One checkpoint rule** — a page is done when its JSON exists; a missing or
  corrupt JSON means "OCR this page again". No progress database.
- **One command** — a PDF file or a folder, same flags.

## Workflow

```
PDF
 |
 v
pypdfium2 (render each page at 200 dpi)
 |
 v
RapidOCR (PP-OCRv6_small ONNX, CPU)
 |
 v
reading order -> paragraphs -> headings
 |
 v
<name>.md  +  <name>.pages.txt
```

## Install

Requires Python 3.10 - 3.12.

```bash
python -m pip install -e .
```

Or install straight from GitHub:

```bash
python -m pip install "git+https://github.com/AlexRaganhar/pdf2md.git"
```

Dependencies are just `rapidocr`, `onnxruntime`, `pypdfium2`, `pypdf` and
`pillow` (about 0.3 GB installed). The OCR models ship inside the RapidOCR
wheel, so there is no first-run download.

## Usage

```bash
pdf2md input.pdf                      # -> input.md + input.pages.txt
pdf2md input.pdf -o out/result.md
pdf2md input.pdf --max-pages 50       # trial run; keeps the checkpoint
pdf2md input.pdf --no-resume          # discard checkpoints, start over

pdf2md scan_folder/                   # -> scan_folder/markdown_output/
pdf2md scan_folder/ out/ --dry-run    # list jobs, convert nothing
pdf2md scan_folder/ out/ --log run.log
pdf2md scan_folder/ out/ --pattern "*.PDF"
```

`python -m pdf2md ...` and `python main.py ...` are equivalent entry points.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | -- | PDF file, or folder of PDFs |
| `-o, --output` | `<input>.md` / `<input>/markdown_output` | Output file or folder |
| `--threshold` | `0.5` | Drop OCR blocks below this confidence |
| `--password` | -- | Password for encrypted PDFs |
| `--max-pages` | all | Only process the first N pages (trial run) |
| `--no-resume` | off | Discard existing page checkpoints |
| `--pattern` | `*.pdf` | Folder mode: filename pattern |
| `--log` | -- | Folder mode: append a run log to this file |
| `--dry-run` | off | Only list the jobs |

## Output

`<name>.md` contains `<!-- Page N -->` markers, paragraphs separated by blank
lines and `##` headings. Leading list markers such as `1.` are escaped
(`1\.`) so legal numbering stays literal.

`<name>.pages.txt` contains the same text with explicit
`【第 N 页 / 共 M 页】` headers, intended for case-file review where every
line must be locatable in the original scan.

A page whose OCR fails is never silently dropped: it is marked
`（第 N 页 OCR 失败）`, the command exits non-zero, and its checkpoint is kept
so a rerun retries just that page.

## Resume

Each page is checkpointed as soon as it is recognized:

```
<name>.work/
|-- source.json              # size + mtime_ns of the source PDF
`-- pages/page_0001.json     # OCR blocks for page 1 (presence = done)
```

- A missing or unreadable page file is simply re-OCR'd on the next run.
- If the source PDF changed (size/mtime differ), the checkpoint is discarded.
- On a fully successful run the `.work` directory is deleted. A `--max-pages`
  trial or a run with failed pages keeps it.

The work directory is always derived from the output path; there is no
`--workdir` flag and therefore no way to accidentally point it at user data.

## Encrypted PDFs

`--password` decrypts an encrypted PDF to a temporary file in the system temp
directory, which is deleted immediately after the conversion. The original
file is never modified and no plaintext copy is left next to the output.

## Folder mode

Folders are converted recursively, preserving the source layout under the
output directory. The output folder itself is skipped during discovery.
Per-file failures are logged and the batch continues; the exit code is 1 if
any file failed or had failed pages.

## Python API

```python
from pdf2md import convert_pdf

result = convert_pdf("scan.pdf", "scan.md", threshold=0.5)
print(result.total_pages, result.failed_pages)
```

`convert_pdf` writes both outputs and returns a `Result` with the Markdown,
page text, failed pages and counters.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite fakes the OCR engine and PDF renderer, so it runs without loading
models.

## Performance

On a 10-page, 200 dpi test document: about 11 s total, of which ~1.5 s is
RapidOCR initialization and the rest is roughly 1 s per page (render + OCR).
Longer documents scale linearly with pages.

## Limits

- Reading order is single-column `(top, left)`; two-column pages may interleave.
- Headings are inferred from block height, not font metadata.
- Tables are not reconstructed; OCR output is line-oriented.
- All pages are held in memory for the final render (hundreds of pages are
  fine; tens of thousands are not).

## What was deliberately left out

- **Docling/HTML**: it only ever received `<p>`/`<h2>`; a second renderer and
  a PyTorch dependency bought formatting this project does not use. The
  Markdown writer covers the actual output.
- **PaddleOCR**: it runs the same PP-OCRv6_small models, so a second backend
  only added 0.4 GB and 13 s of startup.
- **`progress.json`**: `done` and `failed` are derivable from the page files;
  storing them separately created two sources of truth.
- **Per-page `.txt` files**: the page-numbered output is generated from the
  JSON at the end, so writing a second copy per page was dead weight.
- **`--workdir`/marker files**: work paths are derived from the output, which
  removes a whole class of "did I just delete the user's directory" bugs.
- **`--lang`**: PP-OCRv6_small is a unified multilingual model, so the flag
  never changed the model.

## Credits

- [RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR runtime.
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCRv6_small models.
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) — PDF rendering.
- [pypdf](https://github.com/py-pdf/pypdf) — encrypted-PDF handling.
- [Pillow](https://python-pillow.org/) — image hand-off.

## Contributing

Issues and pull requests are welcome. Before sending a PR:

```bash
python -m unittest discover -s tests -v
```

Keep the pipeline single-path: no second OCR backend and no optional renderer
unless there is a demonstrated capability gap.

## License

MIT — see [LICENSE](LICENSE).
