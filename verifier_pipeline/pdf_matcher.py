"""
pdf_matcher.py — PDF table reference extraction and fuzzy matching.

Optional dependency: pdfplumber.  If not installed, extract_pdf_refs returns [].

Public API
----------
extract_pdf_refs(pdf_path)               -> list[PDFTableRef]
match_table_to_pdf(parsed, pdf_refs)     -> (page, section, pdf_table_index, confidence)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from models import PDFTableRef

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import pdfplumber          # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from rapidfuzz import fuzz  # type: ignore
    def _fuzzy(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b) / 100.0
except ImportError:
    import difflib
    def _fuzzy(a: str, b: str) -> float:   # type: ignore
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf_refs(pdf_path: Path) -> list[PDFTableRef]:
    """
    Extract table locations from the PDF using pdfplumber.
    Returns an empty list if pdfplumber is not installed or extraction fails.
    """
    if not _HAS_PDFPLUMBER:
        return []

    refs: list[PDFTableRef] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                # Guess section: first ALL-CAPS line or "Letter + digit" heading
                section = ""
                for line in page_text.split("\n")[:5]:
                    if re.match(r"^[A-Z][0-9\.\s]", line.strip()) or line.isupper():
                        section = line.strip()[:80]
                        break

                for t_idx, table in enumerate(page.extract_tables() or [], start=1):
                    header_text = " ".join(
                        cell or "" for cell in (table[0] if table else [])
                    ).strip()
                    refs.append(PDFTableRef(
                        page        = page_num,
                        section     = section or f"Page {page_num}",
                        pdf_table_index = t_idx,
                        header_text = header_text,
                        page_text   = page_text[:400],
                    ))
    except Exception as exc:
        print(f"[warn] PDF extraction failed: {exc}", file=sys.stderr)

    return refs


# ── fuzzy matching ────────────────────────────────────────────────────────────

def match_table_to_pdf(
    parsed: dict,
    pdf_refs: list[PDFTableRef],
) -> tuple[Optional[int], Optional[str], Optional[int], float]:
    """
    Fuzzy-match the Markdown table's header/content to the closest PDF table ref.

    Returns
    -------
    (page, section, pdf_table_index, match_confidence)
    All values are None / 0.0 when no refs are available.
    """
    if not pdf_refs:
        return None, None, None, 0.0

    md_header = " ".join(parsed.get("header", []))
    md_sample = md_header + " " + " ".join(
        c for row in (parsed.get("body_rows") or [])[:3] for c in row
    )

    best_score = 0.0
    best_ref: Optional[PDFTableRef] = None

    for ref in pdf_refs:
        score = _fuzzy(md_sample, ref.header_text + " " + ref.page_text)
        if score > best_score:
            best_score = score
            best_ref   = ref

    if best_ref:
        return (
            best_ref.page,
            best_ref.section,
            best_ref.pdf_table_index,
            round(best_score, 3),
        )
    return None, None, None, 0.0
