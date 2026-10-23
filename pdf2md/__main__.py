"""Allow running the CLI via `python -m pdf2md`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
