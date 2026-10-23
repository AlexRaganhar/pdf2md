#!/usr/bin/env python3
"""Compatibility launcher so `python main.py <path>` works like the installed CLI."""

from pdf2md.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
