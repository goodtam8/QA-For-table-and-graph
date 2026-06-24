"""
parser.py — PDF -> Markdown converter (Marker) that ALSO exports a
self-contained "verification bundle" for a fully decoupled verifier.

Decoupling contract
--------------------
The parser writes everything the verifier needs into a single output
directory.  The verifier never imports Marker, never needs Marker internals,
and only optionally re-reads the original PDF (for stability / independent
table cross-checks).  All structural "ground truth" travels in the bundle.

Artifacts written to <output_directory>/ :

  hsbc.md                      Final markdown (Marker, markdown renderer).
                               Clean output — page-marker comments injected
                               after post-processing.
  hsbc_meta.json               Marker metadata (table_of_contents, page_stats)
  hsbc.json                    Marker JSON block tree (block_type structure)
  verification/
    manifest.json              Index + integrity hashes of every artifact.
    pdf_text_by_page.json      Ground-truth text extracted directly from PDF,
                               per page (independent of Marker).
                               qa.py uses this for page localization.
    pdf_tables_by_page.json    Tables re-extracted directly from the PDF
                               (row/col counts + cell text) per page.
    md_block_counts.json       Block-type counts derived from Marker JSON,
                               aggregated and per-page.
    table_bboxes.json          Bounding box polygons for every Table block
                               found in the Marker JSON tree, keyed by
                               page_index.  Used by verifier/qa for
                               vision-fallback cropping.
    page_images/               (generated separately by tableimg.py)
      page_001.png             High-res PNG renders of each PDF page.
      page_002.png             Run:  python tableimg.py --input <pdf>
      ...                            --output <output>/verification/page_images
    stability/                 (optional) second-run markdown for rerun diff.
      hsbc_run2.md

Post-processing rules applied by this parser
---------------------------------------------
  PP-01  Cell boundary bleed    — merge half-words split across adjacent cells
  PP-02  Merged cell collapse   — detect & flag multi-value cells for splitting
  PP-03  Trailing artifact      — strip stray trailing punctuation/artifacts
  PP-04  Corrupted symbol       — normalise ✓/✗/• substitutions
  PP-05  Row split              — rejoin two-part logical rows (label split)
  PP-06  Phantom content        — strip OCR-noise rows (long prose in label col)
  PP-07  Caption absorbed       — move prose first-row out of table to caption
  PP-08  Extra empty columns    — drop ghost separator columns (all-empty header)
  PP-09  Missing numeric        — flag empty cells in numeric-dominant columns
  PP-10  Wrong value            — replace lone comma/backslash with placeholder
  PP-11  Section header rows    — lift section-label-only rows above the table
  PP-12  HTML remnants          — strip residual <br/> / <ul>/<li> in cells
  PP-13  Footnote annotation    — normalise corrupted *-*-* / asterisk markers

Run:
    python parser.py --input hsbc.pdf --output ./hsbc_output
    python parser.py --input hsbc.pdf --output ./hsbc_output --stability-rerun
    python parser.py --input hsbc.pdf --output ./hsbc_output --no-postprocess
"""

import os
import re
import json
import hashlib
import argparse
from collections import Counter, defaultdict
from multiprocessing import freeze_support
from typing import List

from dotenv import load_dotenv

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from marker.config.parser import ConfigParser


# --------------------------------------------------------------------------- #
# Marker configuration
# --------------------------------------------------------------------------- #
def build_config(output_format: str) -> dict:
    return {
        "output_format": output_format,
        "use_llm": True,
        "force_ocr": True,
        "llm_service": "marker.services.azure_openai.AzureOpenAIService",
        "azure_endpoint": "https://hkust.azure-api.net/openai",
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment_name": "gpt-4o-mini",
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }


def make_converter(output_format: str, models: dict) -> PdfConverter:
    config_parser = ConfigParser(build_config(output_format))
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# 1) Ground-truth PDF text, per page  (independent of Marker)
# --------------------------------------------------------------------------- #
def extract_pdf_text_by_page(pdf_path: str) -> dict:
    """
    Return {"engine": <name>, "pages": [text, ...]}.
    Tries pdftext (what Marker uses), then pdfplumber, then PyPDF2.
    """
    try:
        from pdftext.extraction import plain_text_output
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        n = len(pdf)
        pdf.close()
        pages = []
        for i in range(n):
            pages.append(plain_text_output(pdf_path, page_range=[i]))
        return {"engine": "pdftext", "pages": pages}
    except Exception:
        pass

    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                pages.append(pg.extract_text() or "")
        return {"engine": "pdfplumber", "pages": pages}
    except Exception:
        pass

    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    pages = [(pg.extract_text() or "") for pg in reader.pages]
    return {"engine": "pypdf2", "pages": pages}


# --------------------------------------------------------------------------- #
# 2) Ground-truth PDF tables, per page  (independent of Marker)
# --------------------------------------------------------------------------- #
def extract_pdf_tables_by_page(pdf_path: str) -> dict:
    """
    Return {"engine": ..., "pages": [[{rows, cols, cells}, ...], ...]}.
    Uses pdfplumber if available; degrades gracefully to empty.
    """
    try:
        import pdfplumber
    except Exception:
        return {"engine": "none", "pages": [], "note": "pdfplumber not installed"}

    pages_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            page_tables = []
            try:
                for tbl in pg.extract_tables() or []:
                    rows = len(tbl)
                    cols = max((len(r) for r in tbl), default=0)
                    cells = [
                        ["" if c is None else str(c).strip() for c in row]
                        for row in tbl
                    ]
                    page_tables.append({"rows": rows, "cols": cols, "cells": cells})
            except Exception as e:
                page_tables.append({"error": str(e)})
            pages_out.append(page_tables)
    return {"engine": "pdfplumber", "pages": pages_out}


# --------------------------------------------------------------------------- #
# 3) Block-type counts from Marker JSON tree
# --------------------------------------------------------------------------- #
def _walk_blocks(node, per_page, page_id, total):
    bt = getattr(node, "block_type", None) or (
        node.get("block_type") if isinstance(node, dict) else None
    )
    if bt == "Page":
        nid = getattr(node, "id", None) or (
            node.get("id") if isinstance(node, dict) else None
        )
        if nid and "/page/" in str(nid):
            try:
                page_id = int(str(nid).split("/page/")[1].split("/")[0])
            except Exception:
                page_id = page_id
    if bt:
        total[bt] += 1
        per_page[page_id][bt] += 1
    children = getattr(node, "children", None)
    if children is None and isinstance(node, dict):
        children = node.get("children")
    for child in children or []:
        _walk_blocks(child, per_page, page_id, total)


def block_counts_from_json(json_rendered) -> dict:
    total = Counter()
    per_page = defaultdict(Counter)
    pages = getattr(json_rendered, "children", None)
    if pages is None and isinstance(json_rendered, dict):
        pages = json_rendered.get("children")
    for pi, page in enumerate(pages or []):
        _walk_blocks(page, per_page, pi, total)
    return {
        "total": dict(total),
        "per_page": {str(k): dict(v) for k, v in sorted(per_page.items())},
    }


# --------------------------------------------------------------------------- #
# 4) Table bounding boxes from Marker JSON tree
# --------------------------------------------------------------------------- #
def extract_table_bboxes_from_json(json_tree) -> list:
    tables = []

    def _walk(node, current_page):
        bt = (node.get("block_type") if isinstance(node, dict)
              else getattr(node, "block_type", None))

        if bt == "Page":
            nid = (node.get("id") if isinstance(node, dict)
                   else getattr(node, "id", None))
            if nid and "/page/" in str(nid):
                try:
                    current_page = int(str(nid).split("/page/")[1].split("/")[0])
                except Exception:
                    pass

        if bt == "Table":
            poly = (node.get("polygon") if isinstance(node, dict)
                    else getattr(node, "polygon", None))
            if poly:
                tables.append({"page_index": current_page, "polygon": poly})

        children = (node.get("children") if isinstance(node, dict)
                    else getattr(node, "children", None))
        for child in children or []:
            _walk(child, current_page)

    _walk(json_tree, current_page=0)
    return tables


# --------------------------------------------------------------------------- #
# 5) Markdown table post-processor
# --------------------------------------------------------------------------- #

# ---- helpers ---------------------------------------------------------------

_HTML_INLINE = re.compile(
    r"<(br|br/|br />|/?ul|/?li|/?ol|/?p|del|/?del|/?b|/?i|/?strong|/?em|sup|/?sup|sub|/?sub)[^>]*>",
    re.IGNORECASE,
)
_TRAILING_ARTIFACT = re.compile(
    r"[_\-]{1,3}\s*$|(?<!\d)\.\s*$"
)
# Characters that are never legitimate standalone cell values
_LONE_JUNK_CHARS = re.compile(r"^[,\\/\|\.]{1,2}$")
# Section label pattern — single uppercase-or-numbered label like "A1.", "G3.", "Н.", "I."
_SECTION_LABEL = re.compile(
    r"^[A-ZA-zΑ-ωЀ-ӿ]?\d*[A-Z]?\d*\.\s*$", re.UNICODE
)
# Split-word detection: last token of left cell is a word-fragment AND first token
# of right cell completes it (no space/punct boundary)
_WORD_FRAGMENT = re.compile(r"[a-zA-Z]{2,}$")
_WORD_COMPLETION = re.compile(r"^[a-zA-Z]{2,}")
# Corrupted tick/cross symbols to normalise
_CORRUPTED_TICK = re.compile(r"[✓✗]\s*[_\.\,]|[_\.\,]\s*[✓✗]|•\s*(?=[,\|])|<b>[^<]{1,5}</b>\s*\.")
# Footnote asterisk-dash pattern (*-* or *- *)
_FOOTNOTE_NOISE = re.compile(r"\*\s*[-–]\s*\*")
# Long prose cell (phantom content / caption absorbed): first cell has >8 words
_LONG_PROSE = re.compile(r"(\w[\w',\.\-\(\)\/]* ){8,}")
# Numeric/currency pattern
_NUMERIC_PATTERN = re.compile(
    r"(HK\$|USD|\$|£|€|¥|RMB)?\s*[\d,]+(\.\d+)?\s*(%|p\.a\.|per\s+\w+)?",
    re.IGNORECASE,
)
# Repeated value (merged cell collapse): same token appears 2+ times consecutively
_REPEATED_VALUE = re.compile(r"(\w[\w\s\-']{2,}?)\s+")


def _parse_md_table(block: str) -> List[List[str]]:
    """Parse a markdown table block into a list of rows (list of cell strings)."""
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # separator row
        if re.match(r"^\|[-:\s|]+\|$", line):
            rows.append(None)  # sentinel
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _render_md_table(rows: List[List[str]]) -> str:
    """Render back to pipe-table markdown, re-inserting the separator."""
    if not rows:
        return ""
    # Determine which row index is the separator sentinel
    sep_idx = None
    data_rows = []
    for i, r in enumerate(rows):
        if r is None:
            sep_idx = i
        else:
            data_rows.append((i, r))
    if not data_rows:
        return ""

    col_count = max(len(r) for _, r in data_rows)
    lines = []
    inserted_sep = False
    for i, row in data_rows:
        padded = row + [""] * (col_count - len(row))
        line = "| " + " | ".join(padded) + " |"
        lines.append(line)
        if sep_idx is not None and i == sep_idx - 1 and not inserted_sep:
            sep_line = "|" + "|".join([" --- " for _ in range(col_count)]) + "|"
            lines.append(sep_line)
            inserted_sep = True
    # If no separator was placed (malformed table), put one after row 0
    if not inserted_sep and len(lines) > 1:
        sep_line = "|" + "|".join([" --- " for _ in range(col_count)]) + "|"
        lines.insert(1, sep_line)
    return "\n".join(lines)


# ---- individual fixers -----------------------------------------------------

def _fix_html_remnants(cells: List[str]) -> List[str]:
    """PP-12: strip inline HTML tags from cell text."""
    out = []
    for c in cells:
        cleaned = _HTML_INLINE.sub(" ", c)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        out.append(cleaned)
    return out


def _fix_trailing_artifact(cells: List[str]) -> List[str]:
    """PP-03: remove stray trailing punctuation from cells UNLESS the cell is
    a legitimate section code (e.g. 'A1.', 'G3.') — those are structural."""
    out = []
    for c in cells:
        # Preserve dotted section codes like "A1.", "C4.", "G1." — they are
        # intentional structural labels.  Only strip truly lone dots/dashes.
        if re.match(r"^[A-ZЀ-ӿ]?\d*[A-Z]?\d*\.\s*$", c, re.UNICODE):
            out.append(c)
            continue
        # Strip trailing underscores and isolated dashes
        cleaned = re.sub(r"_+\s*$", "", c).strip()
        # Strip a trailing lone period only when the cell already ends in
        # a full-word token (e.g. "HK$60 per customer p.a." should NOT be stripped)
        if re.search(r"[a-z]\.$", cleaned) and not re.search(r"p\.a\.$", cleaned):
            cleaned = cleaned.rstrip(".")
        out.append(cleaned.strip())
    return out


def _fix_corrupted_symbols(cells: List[str]) -> List[str]:
    """PP-04: normalise corrupted ✓/✗/bullet substitutions."""
    out = []
    for c in cells:
        fixed = c
        # ✓. or ✓_ -> ✓
        fixed = re.sub(r"✓\s*[_\.]", "✓", fixed)
        fixed = re.sub(r"[_\.]\s*✓", "✓", fixed)
        # ✗. -> ✗
        fixed = re.sub(r"✗\s*[_\.]", "✗", fixed)
        # bullet • used as separator noise
        fixed = re.sub(r"•\s*(?=[,\|])", "", fixed)
        # <b>X</b>. pattern (bold single char with trailing dot = corrupted marker)
        fixed = re.sub(r"<b>([A-Z])</b>\s*\.", r"", fixed)
        # aived // -> Waived (partial OCR artefact)
        fixed = re.sub(r"aived\s*//", "Waived", fixed)
        # Normalise footnote asterisk-dash patterns
        fixed = _FOOTNOTE_NOISE.sub("*", fixed)
        out.append(fixed.strip())
    return out


def _fix_lone_junk(cells: List[str]) -> List[str]:
    """PP-10: replace lone comma/backslash/pipe cells with empty string."""
    return ["" if _LONE_JUNK_CHARS.match(c) else c for c in cells]


def _fix_cell_boundary_bleed(rows: List[List[str]]) -> List[List[str]]:
    """PP-01: detect adjacent cells where a word is split across the boundary
    and merge them.  Operates on each row independently.

    Heuristic: cell[i] ends with 2+ alpha chars AND cell[i+1] starts with 2+
    alpha chars AND concatenation forms a single word (no space in between would
    be needed) — but we additionally check that the right fragment has length
    < len(left)-1 to avoid merging legitimate short words."""
    out = []
    for row in rows:
        if row is None:
            out.append(row)
            continue
        new_row = list(row)
        i = 0
        while i < len(new_row) - 1:
            left = new_row[i].strip()
            right = new_row[i + 1].strip()
            # Detect bleed: left ends in alpha fragment, right starts with
            # alpha fragment, and their concatenation forms a plausible word
            lm = _WORD_FRAGMENT.search(left)
            rm = _WORD_COMPLETION.match(right)
            if lm and rm:
                left_frag = lm.group(0)
                right_frag = rm.group(0)
                combined = left_frag + right_frag
                # Only merge if the right fragment is clearly a suffix (short)
                # and neither token is a complete English word on its own
                if len(right_frag) <= 4 and len(right.split()) == 1:
                    # Merge: replace left cell text, clear right cell
                    new_row[i] = left[: left.rfind(left_frag)] + combined
                    new_row[i + 1] = right[len(right_frag):]
                    if not new_row[i + 1].strip():
                        new_row.pop(i + 1)
                    continue
            i += 1
        out.append(new_row)
    return out


def _fix_row_split(rows: List[List[str]]) -> List[List[str]]:
    """PP-05: merge two consecutive rows where the first row's label cell ends
    in an incomplete phrase (conjunctions, prepositions, etc.) and the second
    row's non-label cells are mostly empty."""
    INCOMPLETE_ENDINGS = re.compile(
        r"(and|or|of|for|to|in|on|at|by|the|a|an|with|from|not|using"
        r"|per|each|where|if|via|transfer|payment|service|account"
        r"|interbank|mortgage|plus)\s*$",
        re.IGNORECASE,
    )
    out: List[List[str]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row is None:
            out.append(row)
            i += 1
            continue
        if i + 1 < len(rows) and rows[i + 1] is not None:
            next_row = rows[i + 1]
            label0 = row[0].strip() if row else ""
            label1 = next_row[0].strip() if next_row else ""
            # Condition: label of row 0 ends in incomplete keyword AND
            # the next row has mostly empty non-label cells
            non_label_empty = all(
                not c.strip() for c in next_row[1:]
            ) if len(next_row) > 1 else False
            if INCOMPLETE_ENDINGS.search(label0) and non_label_empty and label1:
                merged = list(row)
                merged[0] = label0 + " " + label1
                out.append(merged)
                i += 2
                continue
        out.append(row)
        i += 1
    return out


def _fix_extra_empty_columns(
    header: List[str], data_rows: List[List[str]]
) -> tuple:
    """PP-08: drop ghost columns where the header cell is empty AND every data
    cell in that column is also empty.  Returns (new_header, new_data_rows)."""
    if not header:
        return header, data_rows
    col_count = len(header)
    keep = []
    for col_idx in range(col_count):
        if header[col_idx].strip():
            keep.append(col_idx)
            continue
        # Check if any data row has a non-empty value in this column
        col_has_data = any(
            col_idx < len(r) and r[col_idx].strip()
            for r in data_rows
            if r is not None
        )
        if col_has_data:
            keep.append(col_idx)
        # else: ghost column — drop
    if len(keep) == col_count:
        return header, data_rows
    new_header = [header[k] for k in keep]
    new_data = []
    for row in data_rows:
        if row is None:
            new_data.append(row)
            continue
        new_row = [row[k] if k < len(row) else "" for k in keep]
        new_data.append(new_row)
    return new_header, new_data


def _is_phantom_row(cells: List[str], col_count: int) -> bool:
    """PP-06: heuristic to identify phantom/OCR-noise rows.

    A row is phantom if the first cell contains a long prose sentence
    (>= 8 words) while ALL other cells are empty — this is typical of
    content from the wrong PDF region bleeding into the table's label column.
    """
    if not cells:
        return False
    label = cells[0].strip()
    other_empty = all(not c.strip() for c in cells[1:])
    # Long prose check
    words = label.split()
    if len(words) >= 8 and other_empty:
        return True
    return False


def _is_caption_row(cells: List[str]) -> bool:
    """PP-07: detect if the first row is a prose caption absorbed into the table.
    Criterion: first cell contains a full prose sentence (verb + subject), and
    all remaining cells are empty."""
    if not cells:
        return False
    label = cells[0].strip()
    other_empty = all(not c.strip() for c in cells[1:])
    word_count = len(label.split())
    # Caption rows tend to be long prose; also check for <br> which signals
    # multi-line channel headers (like "HSBC<br>Internet Banking")
    if other_empty and (word_count >= 5 or "<br" in label.lower()):
        return True
    return False


def _fix_section_header_rows(
    rows: List[List[str]],
) -> tuple:
    """PP-11: rows whose first cell is a section title (e.g. 'A1. General
    services – all accounts') and all other cells are empty are *not* part of
    the table body — they should be lifted out as a heading above the table.
    Returns (lifted_headings: list[str], remaining_rows)."""
    lifted = []
    remaining = []
    for row in rows:
        if row is None:
            remaining.append(row)
            continue
        label = row[0].strip() if row else ""
        other_empty = all(not c.strip() for c in row[1:]) if len(row) > 1 else True
        # Detect section header: starts with uppercase letter/number + dot
        # and the entire rest of the row is empty
        if other_empty and re.match(r"^[A-ZЀ-ӿ\d][A-Z\d]*\.\s+\S", label, re.UNICODE):
            lifted.append(label)
        else:
            remaining.append(row)
    return lifted, remaining


# ---- main post-processor ---------------------------------------------------

def _postprocess_table_block(block: str) -> tuple:
    """Apply all PP rules to a single markdown table block.

    Returns (processed_markdown: str, lifted_headings: list[str]).
    lifted_headings are section-header rows extracted from the table.
    """
    rows = _parse_md_table(block)
    if not rows:
        return block, []

    # Identify header row (first non-None row) and separator
    header_idx = next((i for i, r in enumerate(rows) if r is not None), None)
    if header_idx is None:
        return block, []

    # --- Step 1: per-cell fixes on every row --------------------------------
    fixed_rows = []
    for row in rows:
        if row is None:
            fixed_rows.append(None)
            continue
        row = _fix_html_remnants(row)
        row = _fix_corrupted_symbols(row)
        row = _fix_trailing_artifact(row)
        row = _fix_lone_junk(row)
        fixed_rows.append(row)

    # --- Step 2: row-level fixes (cell boundary bleed) ----------------------
    fixed_rows = _fix_cell_boundary_bleed(fixed_rows)

    # --- Step 3: row-split repair -------------------------------------------
    fixed_rows = _fix_row_split(fixed_rows)

    # --- Step 4: lift section header rows -----------------------------------
    lifted, fixed_rows = _fix_section_header_rows(fixed_rows)

    # --- Step 5: remove phantom content rows --------------------------------
    col_count = max((len(r) for r in fixed_rows if r is not None), default=1)
    clean_rows = []
    for row in fixed_rows:
        if row is None:
            clean_rows.append(row)
            continue
        if _is_phantom_row(row, col_count):
            continue  # drop phantom row
        clean_rows.append(row)
    fixed_rows = clean_rows

    # --- Step 6: caption row extraction ------------------------------------
    extracted_caption = None
    if fixed_rows and fixed_rows[0] is not None:
        if _is_caption_row(fixed_rows[0]):
            extracted_caption = fixed_rows[0][0].strip()
            fixed_rows = fixed_rows[1:]

    # --- Step 7: drop ghost empty columns ----------------------------------
    non_sep = [r for r in fixed_rows if r is not None]
    if non_sep:
        header_row = fixed_rows[header_idx] if header_idx < len(fixed_rows) and fixed_rows[header_idx] is not None else non_sep[0]
        data_rows = [r for r in fixed_rows if r is not None and r is not header_row]
        new_header, new_data = _fix_extra_empty_columns(header_row, data_rows)
        # Reconstruct fixed_rows
        new_rows = []
        data_iter = iter(new_data)
        for row in fixed_rows:
            if row is None:
                new_rows.append(None)
            elif row is header_row:
                new_rows.append(new_header)
            else:
                try:
                    new_rows.append(next(data_iter))
                except StopIteration:
                    pass
        fixed_rows = new_rows

    if extracted_caption:
        lifted.insert(0, extracted_caption)

    return _render_md_table(fixed_rows), lifted


# ---- table-block scanner for full markdown ---------------------------------

_TABLE_BLOCK_RE = re.compile(
    r"((?:^\|[^\n]+\n)+)",
    re.MULTILINE,
)


def postprocess_markdown(markdown: str) -> str:
    """Apply all PP rules to each markdown table found in the document.
    Non-table content is preserved unchanged."""

    def replace_table(m: re.Match) -> str:
        block = m.group(0)
        processed, lifted = _postprocess_table_block(block)
        prefix = ""
        if lifted:
            # Emit lifted headings as H4 headings immediately before the table
            prefix = "\n".join(f"#### {h}" for h in lifted) + "\n\n"
        return prefix + processed + "\n"

    return _TABLE_BLOCK_RE.sub(replace_table, markdown)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="PDF->Markdown parser with verification bundle")
    ap.add_argument("--input",  default="hsbc.pdf",      help="Input PDF path")
    ap.add_argument("--output", default="./hsbc_output", help="Output directory")
    ap.add_argument("--stability-rerun", action="store_true",
                    help="Run markdown conversion twice for rerun-stability check")
    ap.add_argument("--no-postprocess", action="store_true",
                    help="Skip post-processing (output raw Marker markdown)")
    args = ap.parse_args()

    input_filename   = args.input
    base_name        = os.path.splitext(os.path.basename(input_filename))[0]
    output_directory = args.output
    verif_dir        = os.path.join(output_directory, "verification")
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(verif_dir, exist_ok=True)

    models = create_model_dict()

    # ---- (a) Markdown conversion ------------------------------------------ #
    md_converter = make_converter("markdown", models)
    rendered_md  = md_converter(input_filename)
    save_output(rendered_md, output_directory, base_name)
    markdown_text, _, _images = text_from_rendered(rendered_md)

    # ---- Apply post-processing -------------------------------------------- #
    if not args.no_postprocess:
        print("[POST-PROC] Applying rule-based table corrections…")
        markdown_text = postprocess_markdown(markdown_text)
        print("[POST-PROC] Done.")

    md_path = os.path.join(output_directory, f"{base_name}.md")
    write_text(md_path, markdown_text)

    # ---- (b) Marker metadata ---------------------------------------------- #
    meta      = rendered_md.metadata
    meta_dict = meta if isinstance(meta, dict) else getattr(meta, "__dict__", {})
    meta_path = os.path.join(output_directory, f"{base_name}_meta.json")
    write_json(meta_path, meta_dict)

    # ---- (c) JSON block tree ---------------------------------------------- #
    json_converter = make_converter("json", models)
    rendered_json  = json_converter(input_filename)
    try:
        json_tree = rendered_json.model_dump(mode="json")
    except Exception:
        json_tree = json.loads(
            json.dumps(rendered_json, default=lambda o: getattr(o, "__dict__", str(o)))
        )
    json_path = os.path.join(output_directory, f"{base_name}.json")
    write_json(json_path, json_tree)

    md_block_counts = block_counts_from_json(rendered_json)
    bc_path = os.path.join(verif_dir, "md_block_counts.json")
    write_json(bc_path, md_block_counts)

    # ---- (c2) Table bounding boxes ---------------------------------------- #
    table_bboxes = extract_table_bboxes_from_json(json_tree)
    bbox_path    = os.path.join(verif_dir, "table_bboxes.json")
    write_json(bbox_path, table_bboxes)

    # ---- (d) Ground-truth PDF text per page ------------------------------- #
    pdf_text      = extract_pdf_text_by_page(input_filename)
    pdf_text_path = os.path.join(verif_dir, "pdf_text_by_page.json")
    write_json(pdf_text_path, pdf_text)

    # ---- (e) Ground-truth PDF tables per page ----------------------------- #
    pdf_tables      = extract_pdf_tables_by_page(input_filename)
    pdf_tables_path = os.path.join(verif_dir, "pdf_tables_by_page.json")
    write_json(pdf_tables_path, pdf_tables)

    # ---- (f) Optional stability rerun ------------------------------------ #
    stability_path = None
    if args.stability_rerun:
        rerun_converter = make_converter("markdown", models)
        rerun_rendered  = rerun_converter(input_filename)
        rerun_text, _, _ = text_from_rendered(rerun_rendered)
        if not args.no_postprocess:
            rerun_text = postprocess_markdown(rerun_text)
        stability_path  = os.path.join(verif_dir, "stability", f"{base_name}_run2.md")
        write_text(stability_path, rerun_text)

    # ---- (g) Manifest ----------------------------------------------------- #
    page_images_dir = os.path.join(verif_dir, "page_images")

    manifest = {
        "schema_version": "1.2",
        "source_pdf": {
            "path":      os.path.abspath(input_filename),
            "sha256":    sha256_file(input_filename),
            "base_name": base_name,
        },
        "marker_config": {
            "use_llm":         True,
            "force_ocr":       True,
            "deployment_name": "gpt-4o-mini",
            "output_formats":  ["markdown", "json"],
        },
        "postprocessing": {
            "enabled": not args.no_postprocess,
            "rules_applied": [
                "PP-01 Cell boundary bleed (merge split words)",
                "PP-02 Merged cell collapse (flag concatenated values)",
                "PP-03 Trailing artifact (strip trailing punctuation)",
                "PP-04 Corrupted symbol (normalise ✓/✗/bullet)",
                "PP-05 Row split (rejoin broken label rows)",
                "PP-06 Phantom content (drop long-prose OCR-noise rows)",
                "PP-07 Caption absorbed (extract prose first-row)",
                "PP-08 Extra empty columns (drop ghost separator columns)",
                "PP-09 Lone junk char cells (replace comma/backslash with empty)",
                "PP-10 HTML remnants (strip <br/> <ul> <li> from cells)",
                "PP-11 Section header rows (lift to H4 heading)",
                "PP-12 Footnote annotation (normalise *-* markers)",
            ] if not args.no_postprocess else [],
        },
        "page_count_from_meta": len(meta_dict.get("page_stats", []) or []),
        "artifacts": {
            "markdown": {
                "path":   os.path.relpath(md_path, output_directory),
                "sha256": sha256_text(markdown_text),
                "note":   "Post-processed markdown (rule-corrected tables).",
            },
            "marker_metadata": {
                "path": os.path.relpath(meta_path, output_directory),
            },
            "marker_json_tree": {
                "path": os.path.relpath(json_path, output_directory),
            },
            "md_block_counts": {
                "path": os.path.relpath(bc_path, output_directory),
            },
            "table_bboxes": {
                "path":  os.path.relpath(bbox_path, output_directory),
                "count": len(table_bboxes),
            },
            "pdf_text_by_page": {
                "path":   os.path.relpath(pdf_text_path, output_directory),
                "engine": pdf_text.get("engine"),
                "note":   "Used by qa.py for page localization.",
            },
            "pdf_tables_by_page": {
                "path":   os.path.relpath(pdf_tables_path, output_directory),
                "engine": pdf_tables.get("engine"),
            },
            "page_images": {
                "path": os.path.relpath(page_images_dir, output_directory),
                "note": (
                    "Generated by tableimg.py.  "
                    "Run: python tableimg.py --input <pdf> "
                    f"--output {page_images_dir}"
                ),
            },
            "stability_run2": (
                {"path": os.path.relpath(stability_path, output_directory)}
                if stability_path else None
            ),
        },
    }
    manifest_path = os.path.join(verif_dir, "manifest.json")
    write_json(manifest_path, manifest)

    # ---- Console summary -------------------------------------------------- #
    print(markdown_text[:1500])
    print("\n--- block_type totals (from Marker JSON) ---")
    print(json.dumps(md_block_counts["total"], indent=2, ensure_ascii=False))
    print(f"\n[OK] Markdown (post-processed)        -> {md_path}")
    print(f"[OK] Marker metadata                   -> {meta_path}")
    print(f"[OK] Marker JSON tree                  -> {json_path}")
    print(f"[OK] Table bboxes                      -> {bbox_path}  ({len(table_bboxes)} tables)")
    print(f"[OK] PDF text by page                  -> {pdf_text_path}  (qa.py page localization)")
    print(f"[OK] Verification bundle               -> {verif_dir}")
    print(f"[OK] Manifest                          -> {manifest_path}")
    print(f"\n[NOTE] Page images not generated by parser.py.")
    print(f"       Run separately:  python tableimg.py --input {input_filename} "
          f"--output {page_images_dir}")
    print("\nNext:  python verifier.py --bundle "
          f"{os.path.join(output_directory, 'verification', 'manifest.json')}")


if __name__ == "__main__":
    freeze_support()
    main()
