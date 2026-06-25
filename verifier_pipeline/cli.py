#!/usr/bin/env python3
"""
cli.py — Command-line entry point for the PDF-to-Markdown Table Verifier.

Usage
-----
    python cli.py \
        --examples  examples.json \
        --pdf       source.pdf \
        --markdown  converted.md \
        --output-report    verification_report.json \
        --output-markdown  annotated_converted.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PDF-to-Markdown Table Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--examples",        type=Path, default=Path("examples.json"))
    p.add_argument("--pdf",             type=Path, default=Path("source.pdf"))
    p.add_argument("--markdown",        type=Path, default=Path("converted.md"))
    p.add_argument("--output-report",   type=Path, default=Path("verification_report.json"))
    p.add_argument("--output-markdown", type=Path, default=Path("annotated_converted.md"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    for attr, flag in [
        ("examples", "--examples"),
        ("pdf",      "--pdf"),
        ("markdown", "--markdown"),
    ]:
        p = getattr(args, attr)
        if not p.exists():
            print(f"[error] {flag}: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    main(
        examples_path        = args.examples,
        pdf_path             = args.pdf,
        markdown_path        = args.markdown,
        output_report_path   = args.output_report,
        output_markdown_path = args.output_markdown,
    )
