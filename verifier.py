"""
verifier.py — Confidence-scoring verifier for PDF->Markdown conversions.

Reads ONLY the verification bundle produced by parser.py (manifest.json and
the artifacts it points to).  It does not import Marker and does not need the
PDF for the core checks — everything structural travels in the bundle.  The
PDF is touched only for the optional independent table re-check / stability.

What it does
------------
Scores each document on five dimensions and, critically, reports WHICH
sections of the markdown are inaccurate vs. the corresponding PDF region
(per page), then routes the document:

    PASS            high overall score, no critical rule failed
    RETRY_OCR_LLM   text/table weak -> re-run with --force_ocr / --use_llm
    MANUAL_REVIEW   low overall score or a critical rule failed

Dimensions (0..1, weighted into overall 0..100):
    text         normalized token recall: PDF text vs markdown text (per page)
    structure    overlap of block-type counts (headings/tables/lists/...) vs
                 a structural expectation derived from the PDF text
    table        per-page table presence + row/col plausibility vs PDF tables
    stability    diff between two markdown runs (if stability bundle present)
    cleanliness  penalties: OCR garbage, repeated headers/footers, empty
                 sections, broken/blank links, replacement chars

Failure severities:
    critical  -> forces MANUAL_REVIEW   (e.g. whole page missing, table dropped)
    major     -> strong score penalty
    minor     -> small score penalty / informational

Usage
-----
    python verifier.py --bundle ./hsbc_output/verification/manifest.json
    python verifier.py --bundle .../manifest.json --report ./report.json
    # thresholds tunable for RAG vs archival/legal use:
    python verifier.py --bundle ... --min-overall 85 --min-table 70
"""

import os
import re
import json
import argparse
import difflib
from collections import Counter


# --------------------------------------------------------------------------- #
# Tokenisation / normalisation
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[0-9a-z\u00c0-\uffff]+", re.IGNORECASE)


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokens(text: str) -> Counter:
    return Counter(_WORD_RE.findall(normalize(text)))


def strip_markdown(md: str) -> str:
    """Remove markdown syntax so we compare on visible text only."""
    md = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)        # code fences
    md = re.sub(r"`[^`]*`", " ", md)                            # inline code
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)               # images
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)            # links -> text
    md = re.sub(r"<[^>]+>", " ", md)                            # html tags
    md = re.sub(r"[#>*_~|`-]", " ", md)                         # md punctuation
    return md


def recall(reference: str, candidate: str) -> float:
    """Token recall: fraction of reference tokens present in candidate."""
    ref, cand = tokens(reference), tokens(candidate)
    if not ref:
        return 1.0
    covered = sum(min(c, cand.get(t, 0)) for t, c in ref.items())
    return covered / max(1, sum(ref.values()))


# --------------------------------------------------------------------------- #
# Bundle loading
# --------------------------------------------------------------------------- #
def load_bundle(manifest_path: str) -> dict:
    base = os.path.dirname(os.path.abspath(manifest_path))
    out_dir = os.path.dirname(base)  # verification/ -> output dir
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    def rel(p):
        return os.path.normpath(os.path.join(out_dir, p))

    arts = manifest["artifacts"]

    def load_json(key):
        a = arts.get(key)
        if not a:
            return None
        with open(rel(a["path"]), encoding="utf-8") as f:
            return json.load(f)

    def load_text(key):
        a = arts.get(key)
        if not a:
            return None
        with open(rel(a["path"]), encoding="utf-8") as f:
            return f.read()

    # marker_json: the full Marker structured output (hsbc.json / <name>.json).
    # parser.py writes it alongside the markdown; we probe by name pattern if
    # it is not yet registered in the manifest artifacts dict.
    marker_json = load_json("marker_json")
    if marker_json is None:
        # Fallback: look for <basename>.json next to the markdown artifact.
        md_art = arts.get("markdown")
        if md_art:
            candidate = os.path.splitext(rel(md_art["path"]))[0] + ".json"
            if os.path.exists(candidate):
                with open(candidate, encoding="utf-8") as f:
                    try:
                        marker_json = json.load(f)
                    except Exception:
                        marker_json = None

    return {
        "manifest": manifest,
        "out_dir": out_dir,
        "markdown": load_text("markdown"),
        "marker_metadata": load_json("marker_metadata"),
        "md_block_counts": load_json("md_block_counts"),
        "pdf_text_by_page": load_json("pdf_text_by_page"),
        "pdf_tables_by_page": load_json("pdf_tables_by_page"),
        "stability_run2": load_text("stability_run2"),
        "marker_json": marker_json,   # NEW: full Marker structured output
    }


# --------------------------------------------------------------------------- #
# Map markdown to pages (using paginate markers if present, else heuristics)
# --------------------------------------------------------------------------- #
_PAGE_MARKER = re.compile(r"\n\{(\d+)\}-{10,}\n")  # Marker --paginate_output style


def split_markdown_by_page(md: str, n_pages: int) -> list:
    """
    Best-effort split of the markdown into n_pages chunks so we can do
    per-page recall.  If pagination markers exist, use them; otherwise split
    proportionally by length (recall is robust to coarse alignment).
    """
    if not md:
        return ["" for _ in range(n_pages)]
    parts = _PAGE_MARKER.split(md)
    if len(parts) > 1:
        # parts = [pre, pageno, chunk, pageno, chunk, ...]
        chunks = []
        i = 1
        while i < len(parts) - 1:
            chunks.append(parts[i + 1])
            i += 2
        if parts[0].strip():
            chunks.insert(0, parts[0])
        # pad/truncate to n_pages
        if len(chunks) >= n_pages:
            return chunks[:n_pages]
        return chunks + [""] * (n_pages - len(chunks))
    # No markers: proportional split.
    if n_pages <= 1:
        return [md]
    L = len(md)
    step = L / n_pages
    return [md[int(i * step):int((i + 1) * step)] for i in range(n_pages)]


# --------------------------------------------------------------------------- #
# Markdown table parsing (for table fidelity)
# --------------------------------------------------------------------------- #
def parse_md_tables(md: str) -> list:
    """Return [{rows, cols}] for every markdown pipe-table found."""
    tables = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            block = [lines[i]]
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            header = [c for c in lines[i].split("|") if c.strip() != ""]
            tables.append({"rows": len(block) - 1, "cols": len(header)})  # minus separator
            i = j
        else:
            i += 1
    # also count HTML tables
    for m in re.finditer(r"<table.*?</table>", md, flags=re.DOTALL | re.IGNORECASE):
        rows = len(re.findall(r"<tr", m.group(0), flags=re.IGNORECASE))
        cols = max((len(re.findall(r"<t[dh]", r, flags=re.IGNORECASE))
                    for r in re.split(r"</tr>", m.group(0), flags=re.IGNORECASE)), default=0)
        tables.append({"rows": rows, "cols": cols})
    return tables


# --------------------------------------------------------------------------- #
# Table CONTENT fidelity (cell text + numbers), not just counts
# --------------------------------------------------------------------------- #
# Glyph-only cells (checkmarks, bullets, OCR noise) are not comparable text and
# are dropped before token comparison.
_MARK_CHARS = "\u2713\u2714\u221a~vV\u2022\u00b7.,-\u2013\u2014\ufffd*"
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _clean_cell(s: str) -> str:
    """Normalise a cell so PDF and markdown renderings of the same value match."""
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)   # markdown line breaks
    s = re.sub(r"<[^>]+>", " ", s)                          # html tags (<b>,<sup>)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)                    # strip $...$ math wrappers
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _cell_word_tokens(cells) -> Counter:
    """Word-token multiset over a 2D cell grid (pure-glyph cells skipped)."""
    c = Counter()
    for row in cells:
        for cell in row:
            cleaned = _clean_cell(cell)
            if cleaned and all(ch in _MARK_CHARS + " " for ch in cleaned):
                continue
            c.update(_WORD_RE.findall(cleaned))
    return c


def _cell_num_tokens(cells) -> Counter:
    """Numeric-token multiset (commas stripped so 1,200 == 1200)."""
    c = Counter()
    for row in cells:
        for cell in row:
            for m in _NUM_RE.findall(_clean_cell(cell)):
                c[m.replace(",", "")] += 1
    return c


def _cell_recall(ref: Counter, cand: Counter):
    """Return (recall, missing_keys). recall=1.0 when nothing to match."""
    if not ref:
        return 1.0, []
    covered = sum(min(c, cand.get(t, 0)) for t, c in ref.items())
    missing = [t for t in ref if cand.get(t, 0) == 0]
    return covered / max(1, sum(ref.values())), missing


def parse_md_tables_with_cells(md: str) -> list:
    """Like parse_md_tables, but also returns the cell grid for content checks."""
    tables = []
    lines = md.splitlines()
    i = 0

    def split_row(ln):
        parts = ln.split("|")
        if parts and parts[0].strip() == "":
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]
        return [p.strip() for p in parts]

    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            block = [lines[i]]
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            header = split_row(block[0])
            body = [split_row(b) for b in block[1:]]   # separator already skipped
            tables.append({"rows": len(body) + 1, "cols": len(header),
                           "cells": [header] + body})
            i = j
        else:
            i += 1
    return tables


def table_content_fidelity(pdf_tbl_pages: list, md: str, cfg: dict):
    """
    Verify the ACTUAL CONTENT of tables (cell text + numbers), not just counts.

    Comparison is document-level on purpose: pdfplumber over-segments logical
    tables into many fragments, and per-page markdown alignment is unreliable
    without pagination markers -- so page-scoped matching produces false
    'table missing' criticals. Document-level cell/number recall is robust to
    both and directly answers "did the table values survive into the markdown?"

    Returns (score 0..1 | None, findings, debug).
    """
    md_tables = parse_md_tables_with_cells(md)

    pdf_tok, pdf_num = Counter(), Counter()
    for p in pdf_tbl_pages:
        for t in p:
            if isinstance(t, dict) and "cells" in t:
                pdf_tok += _cell_word_tokens(t["cells"])
                pdf_num += _cell_num_tokens(t["cells"])
    md_tok, md_num = Counter(), Counter()
    for t in md_tables:
        md_tok += _cell_word_tokens(t["cells"])
        md_num += _cell_num_tokens(t["cells"])

    cell_recall, missing_terms = _cell_recall(pdf_tok, md_tok)
    num_recall, missing_nums = _cell_recall(pdf_num, md_num)

    pages_with_pdf = sum(1 for p in pdf_tbl_pages
                         if any(isinstance(t, dict) and "cells" in t for t in p))
    total_md = len(md_tables)
    presence = min(1.0, total_md / max(1, pages_with_pdf))

    # numbers dominate for fee/charge documents
    score = 0.20 * presence + 0.30 * cell_recall + 0.50 * num_recall

    findings = []
    if pdf_tok and cell_recall < cfg["table_cell_critical"]:
        findings.append({"dimension": "table", "severity": "critical",
                         "detail": "Table cell-text recall {:.0%}: table CONTENT largely missing.".format(cell_recall),
                         "sample_missing_terms": sorted(missing_terms)[:20]})
    elif pdf_tok and cell_recall < cfg["table_cell_major"]:
        findings.append({"dimension": "table", "severity": "major",
                         "detail": "Table cell-text recall {:.0%} below target.".format(cell_recall),
                         "sample_missing_terms": sorted(missing_terms)[:20]})
    if sum(pdf_num.values()) >= 5 and num_recall < cfg["table_numeric_critical"]:
        findings.append({"dimension": "table", "severity": "critical",
                         "detail": "Only {:.0%} of table NUMBERS preserved -- {} numeric value(s) "
                                   "missing from markdown tables.".format(num_recall, len(missing_nums)),
                         "sample_missing_numbers": sorted(missing_nums)[:20]})
    elif sum(pdf_num.values()) >= 5 and num_recall < cfg["table_numeric_major"]:
        findings.append({"dimension": "table", "severity": "major",
                         "detail": "{:.0%} of table numbers preserved; {} value(s) missing.".format(
                             num_recall, len(missing_nums)),
                         "sample_missing_numbers": sorted(missing_nums)[:20]})

    debug = {
        "pdf_table_fragments": sum(len(p) for p in pdf_tbl_pages),
        "pdf_pages_with_tables": pages_with_pdf,
        "md_tables": total_md,
        "presence": round(presence, 3),
        "cell_text_recall": round(cell_recall, 3),
        "numeric_recall": round(num_recall, 3),
        "pdf_numbers": sum(pdf_num.values()),
        "missing_numbers": sorted(missing_nums)[:20],
    }
    return round(score, 4), findings, debug


# --------------------------------------------------------------------------- #
# NEW: Cell-boundary bleed detection
#
# Catches the class of error where the last character(s) of a cell are the
# START of a word that continues in the NEXT cell on the same row, producing
# a dangling uppercase letter in cell[i] and a lowercase fragment in cell[i+1].
#
# Classic example from the HSBC document:
#   cell = "Non-HSBC Credit Card/ Bank Account     F"   (bleeds "F")
#   next  = "ree"                                        (orphan suffix)
#   → should be "Free" in the charge column
#
# The check looks for two patterns:
#   A) left_cell ends with a single uppercase letter AND right_cell is a
#      pure-lowercase word of >= 2 chars (high-precision signal of a bleed).
#   B) left_cell ends with a space + uppercase letter sequence AND right_cell
#      starts with a lowercase continuation that, when concatenated, forms a
#      dictionary-plausible word (heuristic: all alpha, length 3-20).
# --------------------------------------------------------------------------- #
_COMMON_SUFFIXES = re.compile(
    r"^(ree|ull|ull|nit|harging|tion|ment|ount|arge|ange|ine|ice|ate|oss|ank|ee|er|"
    r"ing|ed|es|ly|al|or|ar|le|re|ge|se|ve|ce|st|nd|rd|th|ng|nt|ny|ne|nk|ck|sk|"
    r"ss|tt|ll|ff|pp|rr|mm|nn|bb|dd|gg|zz)$",
    re.IGNORECASE,
)

# Pattern: cell ends with whitespace + one or more uppercase letters that are
# the beginning of a word whose tail appears in the next cell.
_BLEED_LEFT_RE = re.compile(r"\s+([A-Z]{1,4})\s*$")


def check_cell_boundary_bleed(md_tables: list) -> list:
    """
    Scan every consecutive cell pair in each table row for character-bleed
    artifacts.  Returns a list of findings (severity=major).

    Two signals are detected:
      1. UPPERCASE_SUFFIX_BLEED: left cell ends with 1-4 uppercase letters
         and the right cell starts with 2+ lowercase letters that together
         form a plausible word (common suffix match OR purely alpha 3-15 chars).
      2. SHORT_ISOLATED_FRAGMENT: a non-first, non-last cell consists of only
         2-4 lowercase letters — a strong indicator that a word was split
         across cell boundaries.
    """
    findings = []
    for ti, t in enumerate(md_tables):
        for ri, row in enumerate(t["cells"]):
            # -- Signal 1: UPPERCASE_SUFFIX_BLEED --
            for ci in range(len(row) - 1):
                left = row[ci]
                right = row[ci + 1].strip()
                m = _BLEED_LEFT_RE.search(left)
                if m:
                    bleed_chars = m.group(1)
                    # right cell should start with lowercase letters
                    if right and right[0].islower() and right[:8].isalpha():
                        candidate = bleed_chars + right
                        # Accept if it forms an all-alpha word <= 20 chars or
                        # the right fragment matches a common suffix
                        if (
                            re.fullmatch(r"[A-Za-z]{3,20}", candidate)
                            or _COMMON_SUFFIXES.match(right)
                        ):
                            row_label = _clean_cell(row[0])[:50] if row else ""
                            findings.append({
                                "dimension": "table",
                                "severity": "major",
                                "table_index": ti + 1,
                                "row": ri,
                                "col": ci,
                                "detail": (
                                    f"Table {ti+1} row '{row_label}' col {ci}: "
                                    f"likely cell-boundary bleed — left cell ends "
                                    f"with {bleed_chars!r}, right cell starts with "
                                    f"{right[:10]!r}. "
                                    f"Reconstructed word candidate: {candidate!r}."
                                ),
                                "left_cell_tail": left[-20:],
                                "right_cell_head": right[:20],
                                "candidate_word": candidate,
                            })

            # -- Signal 2: SHORT_ISOLATED_FRAGMENT --
            # A cell in the middle of a row that is purely 2-5 lowercase
            # alpha characters is almost always a split-off word fragment.
            for ci in range(1, len(row) - 1):   # skip first and last cells
                cell = row[ci].strip()
                if re.fullmatch(r"[a-z]{2,5}", cell):
                    row_label = _clean_cell(row[0])[:50] if row else ""
                    # Check left neighbour: does it end with a capital?
                    left_tail = row[ci - 1].rstrip()[-1] if row[ci - 1].strip() else ""
                    hint = (f" Left cell ends with {left_tail!r}." if left_tail.isupper() else "")
                    findings.append({
                        "dimension": "table",
                        "severity": "major",
                        "table_index": ti + 1,
                        "row": ri,
                        "col": ci,
                        "detail": (
                            f"Table {ti+1} row '{row_label}' col {ci}: "
                            f"isolated short fragment {cell!r} — likely a word "
                            f"split across cell boundaries.{hint}"
                        ),
                        "fragment": cell,
                        "left_cell_tail": row[ci - 1][-20:] if ci > 0 else "",
                    })

    return findings


# --------------------------------------------------------------------------- #
# NEW: Per-row cell recall (catches token-aggregation masking)
#
# The global table_content_fidelity() pools all cell tokens across the whole
# document.  If a word like "free" appears in any OTHER row, the missing
# "free" from a corrupted cell is silently covered.  Per-row recall compares
# each PDF table row's tokens against the nearest-matching markdown row,
# flagging rows where the match is suspiciously poor.
# --------------------------------------------------------------------------- #

def _row_tokens(row: list) -> Counter:
    """Word tokens for a single table row (list of cell strings)."""
    c = Counter()
    for cell in row:
        cleaned = _clean_cell(cell)
        if cleaned and all(ch in _MARK_CHARS + " " for ch in cleaned):
            continue
        c.update(_WORD_RE.findall(cleaned))
    return c


def check_per_row_recall(pdf_tbl_pages: list, md_tables: list,
                          low_recall_threshold: float = 0.5) -> list:
    """
    For every PDF table row, find the best-matching markdown row by token
    overlap and flag pairs where word-recall is below low_recall_threshold.

    This catches cases where the document-level aggregation hides a
    local corruption: the missing token may exist elsewhere but is absent
    from the specific row in question.

    Only rows with >= 2 meaningful word tokens are evaluated (short/empty
    rows generate too many false positives).
    """
    findings = []
    # Flatten all markdown rows across all tables for matching.
    all_md_rows = []
    for ti, t in enumerate(md_tables):
        for ri, row in enumerate(t["cells"]):
            all_md_rows.append((ti, ri, row, _row_tokens(row)))

    if not all_md_rows:
        return findings

    row_idx = 0  # PDF table fragment counter (for labelling only)
    for page_tables in pdf_tbl_pages:
        for tbl in page_tables:
            if not isinstance(tbl, dict) or "cells" not in tbl:
                continue
            row_idx += 1
            for ri, pdf_row in enumerate(tbl["cells"]):
                pdf_tok = _row_tokens(pdf_row)
                total = sum(pdf_tok.values())
                if total < 2:
                    continue

                # Find best-matching markdown row by token recall.
                best_rec = 0.0
                best_match = None
                for (mti, mri, md_row, md_tok) in all_md_rows:
                    covered = sum(min(c, md_tok.get(t, 0)) for t, c in pdf_tok.items())
                    r = covered / total
                    if r > best_rec:
                        best_rec = r
                        best_match = (mti, mri, md_row)

                if best_rec < low_recall_threshold:
                    pdf_label = " | ".join(_clean_cell(c) for c in pdf_row)[:80]
                    md_label = ""
                    if best_match:
                        md_label = " | ".join(_clean_cell(c) for c in best_match[2])[:80]
                    missing = [t for t in pdf_tok if (best_match[2] and
                                _row_tokens(best_match[2]).get(t, 0) == 0)]
                    findings.append({
                        "dimension": "table",
                        "severity": "major",
                        "detail": (
                            f"PDF table fragment {row_idx} row {ri}: best markdown row match "
                            f"has only {best_rec:.0%} token recall. "
                            f"PDF row: {pdf_label!r}. "
                            f"Best MD row: {md_label!r}."
                        ),
                        "pdf_row_preview": pdf_label,
                        "best_md_row_preview": md_label,
                        "per_row_recall": round(best_rec, 3),
                        "missing_tokens": sorted(missing)[:15],
                    })

    return findings


# --------------------------------------------------------------------------- #
# NEW: Marker JSON cell fidelity
#
# parser.py exports the full Marker JSON (hsbc.json).  Its Table blocks carry
# the INTENDED cell text as Marker understood it.  Comparing Marker's own cell
# text against the re-parsed markdown pipe-table cells catches rendering
# artifacts that occur AFTER Marker has done its work — e.g. the pipe-table
# serialiser splitting a word across two cells.
#
# We compare at the cell level (exact or near-exact string match) rather than
# token level to catch partial-word bleeds that survive token-bag recall.
# --------------------------------------------------------------------------- #

def _extract_marker_table_cells(marker_json: dict) -> list:
    """
    Walk the Marker JSON tree and return a list of 2-D cell grids, one per
    Table block.  Handles both the legacy flat list format and the nested
    children/blocks format emitted by recent Marker versions.
    """
    tables = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        btype = node.get("block_type") or node.get("type") or ""
        if "table" in btype.lower():
            # Try to extract a cell grid from rows/cells sub-structure.
            grid = _extract_grid_from_table_node(node)
            if grid:
                tables.append(grid)
        # Recurse into children / content / blocks
        for key in ("children", "content", "blocks", "rows"):
            child = node.get(key)
            if isinstance(child, list):
                for item in child:
                    _walk(item)

    def _extract_grid_from_table_node(node):
        # Format A: node["rows"] is a list of row objects, each with "cells"
        rows = node.get("rows")
        if isinstance(rows, list):
            grid = []
            for row in rows:
                if isinstance(row, dict):
                    cells = row.get("cells", [])
                    grid.append([_get_cell_text(c) for c in cells])
                elif isinstance(row, list):
                    grid.append([_get_cell_text(c) for c in row])
            if grid:
                return grid
        # Format B: node["html"] / node["markdown"] — fall back to re-parsing
        # the embedded markdown snippet if present.
        md_snippet = node.get("markdown") or node.get("md") or ""
        if md_snippet and "|" in md_snippet:
            parsed = parse_md_tables_with_cells(md_snippet)
            if parsed:
                return parsed[0]["cells"]
        # Format C: flat "cells" list of lists
        cells = node.get("cells")
        if isinstance(cells, list) and cells and isinstance(cells[0], list):
            return [[_get_cell_text(c) for c in row] for row in cells]
        return None

    def _get_cell_text(cell) -> str:
        if isinstance(cell, str):
            return cell
        if isinstance(cell, dict):
            return (cell.get("text") or cell.get("content") or
                    cell.get("value") or cell.get("html") or "")
        return ""

    if isinstance(marker_json, list):
        for item in marker_json:
            _walk(item)
    elif isinstance(marker_json, dict):
        # Top-level may be {"pages": [...]} or directly a block tree
        pages = marker_json.get("pages") or marker_json.get("children") or []
        if pages:
            for page in pages:
                _walk(page)
        else:
            _walk(marker_json)

    return tables


def check_marker_cell_fidelity(marker_json: dict, md: str,
                                bleed_threshold: float = 0.85) -> list:
    """
    Compare Marker's own structured table cells against the cells in the
    re-parsed pipe-table markdown.  Flags cells where the Marker text and the
    markdown cell text diverge, with special focus on character-bleed artifacts.

    A bleed artifact is characterised by:
      - Marker cell text is a complete word (e.g. "Free")
      - Markdown cell is only the tail of that word (e.g. "ree") or empty
      - The missing prefix was concatenated to the PREVIOUS cell

    Returns a list of findings (severity=major).
    """
    if not marker_json:
        return []

    marker_tables = _extract_marker_table_cells(marker_json)
    md_tables = parse_md_tables_with_cells(md)

    if not marker_tables or not md_tables:
        return []

    findings = []
    # Match marker tables to md tables by proximity (best-token-overlap).
    for mki, mk_grid in enumerate(marker_tables):
        if not mk_grid:
            continue
        mk_tok = _cell_word_tokens(mk_grid)
        if not mk_tok:
            continue

        # Find best matching MD table
        best_idx, best_score = 0, -1.0
        for mdi, md_t in enumerate(md_tables):
            md_tok = _cell_word_tokens(md_t["cells"])
            score, _ = _cell_recall(mk_tok, md_tok)
            if score > best_score:
                best_score, best_idx = score, mdi

        if best_score < 0.3:
            # Tables are too dissimilar — likely no corresponding MD table.
            continue

        md_grid = md_tables[best_idx]["cells"]

        # Row-level comparison: for each Marker row, find best matching MD row.
        for ri, mk_row in enumerate(mk_grid):
            mk_row_tok = _row_tokens(mk_row)
            if sum(mk_row_tok.values()) < 2:
                continue

            best_r, best_mrow = 0.0, None
            for md_row in md_grid:
                md_row_tok = _row_tokens(md_row)
                covered = sum(min(c, md_row_tok.get(t, 0)) for t, c in mk_row_tok.items())
                r = covered / max(1, sum(mk_row_tok.values()))
                if r > best_r:
                    best_r, best_mrow = r, md_row

            if best_r < bleed_threshold and best_mrow is not None:
                # Cell-level diff within the matched row pair.
                for ci, mk_cell in enumerate(mk_row):
                    mk_clean = _clean_cell(mk_cell)
                    if not mk_clean or len(mk_clean) < 2:
                        continue
                    # Find closest MD cell in matched row.
                    md_cell = best_mrow[ci] if ci < len(best_mrow) else ""
                    md_clean = _clean_cell(md_cell)

                    # Check if the MD cell is a strict suffix of the Marker cell
                    # (the characteristic bleed pattern: "free" in MD, "free" should
                    # be in a short cell but the leading "F" was in the previous col).
                    if mk_clean and md_clean and mk_clean != md_clean:
                        # Is md_clean a proper suffix of mk_clean?
                        if mk_clean.endswith(md_clean) and len(md_clean) < len(mk_clean):
                            bleed = mk_clean[: len(mk_clean) - len(md_clean)]
                            findings.append({
                                "dimension": "table",
                                "severity": "major",
                                "table_index": mki + 1,
                                "row": ri,
                                "col": ci,
                                "detail": (
                                    f"Marker table {mki+1} row {ri} col {ci}: "
                                    f"Marker cell text is {mk_clean!r} but markdown "
                                    f"cell is only {md_clean!r} — first {len(bleed)} "
                                    f"char(s) {bleed!r} appear to have bled into the "
                                    f"previous cell (cell-boundary bleed artifact)."
                                ),
                                "marker_cell": mk_clean,
                                "markdown_cell": md_clean,
                                "bleed_chars": bleed,
                            })
                        # Is md_clean empty but mk_clean is not?  Possible full bleed.
                        elif not md_clean and mk_clean:
                            findings.append({
                                "dimension": "table",
                                "severity": "major",
                                "table_index": mki + 1,
                                "row": ri,
                                "col": ci,
                                "detail": (
                                    f"Marker table {mki+1} row {ri} col {ci}: "
                                    f"Marker cell text is {mk_clean!r} but "
                                    f"markdown cell is EMPTY — content entirely "
                                    f"missing or bled into adjacent cell."
                                ),
                                "marker_cell": mk_clean,
                                "markdown_cell": "",
                            })

    return findings


# --------------------------------------------------------------------------- #
# Tick-mark consistency: catch cells that SHOULD be a tick but were OCR-mangled
# into stray glyphs (',', '.', '#', '~', 'V', '_', ...). Recall-based checks
# cannot see this because the corrupted glyph "matches" the (equally corrupted)
# ground truth. This is a content-internal sanity check, no ground truth needed.
# --------------------------------------------------------------------------- #
_GOOD_TICKS = set("\u2713\u2714\u221a")               # checkmark glyphs
_SUSPECT_GLYPHS = set(",.\\u2022\\u00b7~^*_-\\u2013\\u2014") | set("vVlI")


def _tick_strip_html(cell: str) -> str:
    s = re.sub(r"<[^>]+>", "", cell)
    return re.sub(r"\s+", " ", s).strip()


def _classify_tick_cell(cell: str) -> str:
    c = _tick_strip_html(cell)
    if c == "":
        return "EMPTY"
    core = c.rstrip(".,_ -").lstrip()
    if core in _GOOD_TICKS:
        return "GOOD_TICK"
    if core and core[0] in _GOOD_TICKS and core[1:].strip(" .,_-").isdigit():
        return "GOOD_TICK"                            # tick + footnote digit
    if len(c) == 1 and c in _SUSPECT_GLYPHS:
        return "SUSPECT"
    stripped = c.replace(" ", "")
    if 1 <= len(stripped) <= 2 and all(ch in _SUSPECT_GLYPHS for ch in stripped):
        return "SUSPECT"
    return "TEXT"


def check_tick_columns(md_tables, tick_col_ratio=0.5, min_ticks=3):
    """Flag suspect cells inside columns that are dominated by valid ticks."""
    findings = []
    for ti, t in enumerate(md_tables):
        cells = t["cells"]
        ncols = max((len(r) for r in cells), default=0)
        for col in range(ncols):
            classes = [(ri, _classify_tick_cell(row[col]))
                       for ri, row in enumerate(cells) if col < len(row)]
            n_good = sum(1 for _, c in classes if c == "GOOD_TICK")
            n_suspect = sum(1 for _, c in classes if c == "SUSPECT")
            n_nonempty = sum(1 for _, c in classes if c != "EMPTY")
            if n_nonempty == 0:
                continue
            tick_share = (n_good + n_suspect) / n_nonempty
            if n_good >= min_ticks and tick_share >= tick_col_ratio:
                for ri, klass in classes:
                    if klass == "SUSPECT":
                        raw = cells[ri][col]
                        rowlabel = _tick_strip_html(cells[ri][0]) if cells[ri] else ""
                        findings.append({
                            "dimension": "table", "severity": "major",
                            "table_index": ti + 1, "row": ri, "col": col,
                            "detail": "Table {} row '{}' col {}: cell={!r} looks like a "
                                      "corrupted tick (column is {}/{} valid checkmarks).".format(
                                          ti + 1, rowlabel[:40], col, raw, n_good, n_nonempty),
                        })
    return findings


# --------------------------------------------------------------------------- #
# Cleanliness signals
# --------------------------------------------------------------------------- #
def cleanliness_signals(md: str) -> dict:
    issues = []
    score = 1.0

    # replacement / control chars => OCR garbage
    repl = md.count("\ufffd")
    if repl:
        issues.append({"severity": "major", "type": "ocr_replacement_chars",
                       "detail": f"{repl} U+FFFD replacement characters"})
        score -= min(0.3, repl * 0.01)

    # long runs of non-word gibberish
    gib = len(re.findall(r"[^\w\s]{8,}", md))
    if gib > 5:
        issues.append({"severity": "minor", "type": "gibberish_runs",
                       "detail": f"{gib} long punctuation/symbol runs"})
        score -= min(0.1, gib * 0.005)

    # repeated header/footer lines (same short line many times)
    line_counts = Counter(l.strip() for l in md.splitlines() if 3 < len(l.strip()) < 80)
    repeated = [(l, c) for l, c in line_counts.items() if c >= 5]
    if repeated:
        issues.append({"severity": "minor", "type": "repeated_lines",
                       "detail": f"{len(repeated)} lines repeated >=5x (likely running header/footer)",
                       "examples": [l for l, _ in repeated[:3]]})
        score -= min(0.1, len(repeated) * 0.01)

    # empty sections: heading immediately followed by another heading / nothing
    headings = [(m.start(), m.group(0)) for m in re.finditer(r"^#{1,6}\s.*$", md, flags=re.MULTILINE)]
    empty = 0
    for idx, (pos, h) in enumerate(headings):
        nxt = headings[idx + 1][0] if idx + 1 < len(headings) else len(md)
        body = strip_markdown(md[pos + len(h):nxt]).strip()
        if len(body) < 3:
            empty += 1
    if empty:
        issues.append({"severity": "minor", "type": "empty_sections",
                       "detail": f"{empty} headings with empty bodies"})
        score -= min(0.15, empty * 0.02)

    # blank / broken links
    blank_links = len(re.findall(r"\]\(\s*\)", md)) + len(re.findall(r"\]\(#\)", md))
    if blank_links:
        issues.append({"severity": "minor", "type": "blank_links",
                       "detail": f"{blank_links} links with empty/anchor-only targets"})
        score -= min(0.1, blank_links * 0.01)

    return {"score": max(0.0, score), "issues": issues}


# --------------------------------------------------------------------------- #
# Structure expectation derived from PDF text (independent of Marker)
# --------------------------------------------------------------------------- #
def expected_structure_from_pdf(pdf_pages: list) -> dict:
    """Cheap heuristics on raw PDF text to know what SHOULD exist."""
    full = "\n".join(pdf_pages)
    # bullet/numbered list lines
    list_lines = len(re.findall(r"^\s*(?:[-*•]|\d+[.)\t])\s+", full, flags=re.MULTILINE))
    # heading-ish lines: ALLCAPS or "A.", "C2." style or Title Case short lines
    heading_lines = len(re.findall(r"^\s*[A-Z][A-Z0-9 ./&'-]{4,}\s*$", full, flags=re.MULTILINE))
    heading_lines += len(re.findall(r"^\s*[A-G]\d?\.\s", full, flags=re.MULTILINE))
    return {"expected_list_lines": list_lines, "expected_headings": heading_lines}


# --------------------------------------------------------------------------- #
# Core verification
# --------------------------------------------------------------------------- #
def verify(bundle: dict, cfg: dict) -> dict:
    md = bundle["markdown"] or ""
    md_text = strip_markdown(md)
    meta = bundle["marker_metadata"] or {}
    block_counts = bundle["md_block_counts"] or {"total": {}, "per_page": {}}
    pdf_text = bundle["pdf_text_by_page"] or {"pages": []}
    pdf_tables = bundle["pdf_tables_by_page"] or {"pages": []}
    marker_json = bundle.get("marker_json")   # NEW

    pdf_pages = pdf_text.get("pages", [])
    n_pages_pdf = len(pdf_pages)
    n_pages_meta = len(meta.get("page_stats", []) or [])

    findings = []          # all severity-tagged issues
    section_flags = []     # per-page "which section is inaccurate"

    # ---- 1) TEXT RECALL (overall + per page) ------------------------------ #
    overall_text_recall = recall("\n".join(pdf_pages), md_text)
    md_pages = split_markdown_by_page(md, max(n_pages_pdf, 1))
    page_recalls = []
    for i, pg in enumerate(pdf_pages):
        r = recall(pg, md_pages[i] if i < len(md_pages) else "")
        page_recalls.append(round(r, 4))
        if r < cfg["page_recall_critical"] and len(tokens(pg)) > 20:
            sev = "critical"
        elif r < cfg["page_recall_major"] and len(tokens(pg)) > 20:
            sev = "major"
        else:
            sev = None
        if sev:
            # snippet of likely-missing content: tokens in pdf, absent in md page
            cand = tokens(md_pages[i] if i < len(md_pages) else "")
            missing = [t for t in tokens(pg) if cand.get(t, 0) == 0]
            section_flags.append({
                "page": i + 1, "dimension": "text", "severity": sev,
                "recall": round(r, 3),
                "detail": f"Page {i+1} text recall {r:.0%} is below threshold.",
                "sample_missing_terms": missing[:15],
            })
            findings.append(section_flags[-1])
    text_score = overall_text_recall

    # ---- 2) STRUCTURE MATCH ---------------------------------------------- #
    totals = block_counts.get("total", {})
    md_tables_parsed = parse_md_tables(md)
    exp = expected_structure_from_pdf(pdf_pages)

    n_headings = totals.get("SectionHeader", 0)
    n_lists = totals.get("ListItem", 0) + totals.get("ListGroup", 0)
    n_tables_json = totals.get("Table", 0) + totals.get("TableGroup", 0)
    n_figs = totals.get("Figure", 0) + totals.get("Picture", 0) + totals.get("FigureGroup", 0)

    def ratio(actual, expected):
        if expected <= 0:
            return 1.0
        return min(1.0, actual / expected)

    head_ratio = ratio(n_headings, exp["expected_headings"])
    list_ratio = ratio(n_lists, exp["expected_list_lines"])
    structure_score = round(0.5 * head_ratio + 0.5 * list_ratio, 4)
    if head_ratio < 0.4 and exp["expected_headings"] > 5:
        findings.append({"dimension": "structure", "severity": "major",
                         "detail": f"Only {n_headings} headings detected vs ~{exp['expected_headings']} expected from PDF."})
    if list_ratio < 0.3 and exp["expected_list_lines"] > 10:
        findings.append({"dimension": "structure", "severity": "major",
                         "detail": f"Only {n_lists} list items vs ~{exp['expected_list_lines']} list-like lines in PDF (bullets likely flattened)."})

    # ---- 3) TABLE FIDELITY (content, not just counts) -------------------- #
    pdf_tbl_engine = pdf_tables.get("engine", "none")
    pdf_tbl_pages = pdf_tables.get("pages", [])
    total_pdf_tables = sum(len(p) for p in pdf_tbl_pages)
    total_md_tables = len(md_tables_parsed) or n_tables_json
    table_debug = None

    if pdf_tbl_engine == "none" or not pdf_tbl_pages:
        table_score = None
        findings.append({
            "dimension": "table", "severity": "major",
            "detail": f"PDF table ground-truth unavailable (engine={pdf_tbl_engine!r}); "
                      f"table fidelity NOT verified. Markdown has {total_md_tables} table(s). "
                      f"Install pdfplumber and regenerate the bundle.",
        })
    elif total_pdf_tables == 0:
        table_score = 1.0  # genuinely no tables in the PDF
    else:
        table_score, table_findings, table_debug = table_content_fidelity(
            pdf_tbl_pages, md, cfg)
        findings.extend(table_findings)
        section_flags.extend([f for f in table_findings if "page" in f])
        if table_score < cfg["min_table"] / 100.0:
            findings.append({"dimension": "table", "severity": "major",
                             "detail": f"Table content score {table_score:.0%} below minimum for table-heavy doc."})

    # ---- 3b) TICK-MARK CONSISTENCY --------------------------------------- #
    md_tables_cells = parse_md_tables_with_cells(md)
    tick_findings = check_tick_columns(
        md_tables_cells,
        tick_col_ratio=cfg.get("tick_col_ratio", 0.5),
        min_ticks=cfg.get("tick_min_per_col", 3),
    )
    if tick_findings:
        findings.extend(tick_findings)
        section_flags.extend(tick_findings)
        if table_score is not None:
            table_score = round(
                table_score * max(0.5, 1.0 - cfg.get("tick_penalty_each", 0.03)
                                  * len(tick_findings)), 4)

    # ---- 3c) CELL-BOUNDARY BLEED (NEW) ----------------------------------- #
    # Detects character-bleed artifacts such as "Non-HSBC ... F" | "ree"
    # instead of "Non-HSBC ..." | "Free".
    bleed_findings = check_cell_boundary_bleed(md_tables_cells)
    if bleed_findings:
        findings.extend(bleed_findings)
        section_flags.extend(bleed_findings)
        if table_score is not None:
            table_score = round(
                table_score * max(0.5, 1.0 - cfg.get("bleed_penalty_each", 0.04)
                                  * len(bleed_findings)), 4)

    # ---- 3d) PER-ROW RECALL (NEW) ---------------------------------------- #
    # Catches token-aggregation masking: a word present elsewhere in the doc
    # hides a missing word in a specific corrupted row.
    if pdf_tbl_pages and total_pdf_tables > 0:
        per_row_findings = check_per_row_recall(
            pdf_tbl_pages, md_tables_cells,
            low_recall_threshold=cfg.get("per_row_recall_threshold", 0.5),
        )
        if per_row_findings:
            findings.extend(per_row_findings)
            section_flags.extend(per_row_findings)
            if table_score is not None:
                table_score = round(
                    table_score * max(0.5, 1.0 - cfg.get("per_row_penalty_each", 0.02)
                                      * len(per_row_findings)), 4)

    # ---- 3e) MARKER JSON CELL FIDELITY (NEW) ----------------------------- #
    # Use Marker's own structured output as a second ground-truth to catch
    # bleed artifacts that survive the pdfplumber token-recall check.
    if marker_json:
        marker_cell_findings = check_marker_cell_fidelity(
            marker_json, md,
            bleed_threshold=cfg.get("marker_cell_bleed_threshold", 0.85),
        )
        if marker_cell_findings:
            findings.extend(marker_cell_findings)
            section_flags.extend(marker_cell_findings)
            if table_score is not None:
                table_score = round(
                    table_score * max(0.5, 1.0 - cfg.get("marker_cell_penalty_each", 0.03)
                                      * len(marker_cell_findings)), 4)

    # ---- 4) PAGE CONSISTENCY --------------------------------------------- #
    page_consistency_ok = True
    if n_pages_meta and n_pages_pdf and n_pages_meta != n_pages_pdf:
        page_consistency_ok = False
        findings.append({"dimension": "page", "severity": "critical",
                         "detail": f"Marker reports {n_pages_meta} pages but PDF text extraction found {n_pages_pdf}."})
    for i, r in enumerate(page_recalls):
        if r == 0.0 and len(tokens(pdf_pages[i])) > 20:
            findings.append({"page": i + 1, "dimension": "page", "severity": "critical",
                             "detail": f"Page {i+1} appears entirely missing from markdown (0% recall)."})

    # ---- 5) STABILITY ---------------------------------------------------- #
    if bundle["stability_run2"] is not None:
        a = strip_markdown(md)
        b = strip_markdown(bundle["stability_run2"])
        sm = difflib.SequenceMatcher(None, a, b)
        stability_score = round(sm.ratio(), 4)
        if stability_score < 0.97:
            findings.append({"dimension": "stability", "severity": "minor",
                             "detail": f"Two conversion runs differ ({stability_score:.1%} similar) — nondeterministic output."})
    else:
        stability_score = None  # not measured

    # ---- 6) CLEANLINESS -------------------------------------------------- #
    clean = cleanliness_signals(md)
    cleanliness_score = clean["score"]
    findings.extend(clean["issues"])

    # ---- AGGREGATE OVERALL SCORE ----------------------------------------- #
    weights = cfg["weights"]
    components = {
        "text": text_score,
        "structure": structure_score,
        "cleanliness": cleanliness_score,
    }
    if table_score is not None:
        components["table"] = table_score
    if stability_score is not None:
        components["stability"] = stability_score
    used_w = {k: weights[k] for k in components}
    wsum = sum(used_w.values())
    overall = sum(components[k] * used_w[k] for k in components) / wsum * 100.0

    n_critical = sum(1 for f in findings if f.get("severity") == "critical")
    n_major = sum(1 for f in findings if f.get("severity") == "major")
    n_minor = sum(1 for f in findings if f.get("severity") == "minor")

    # ---- ROUTING --------------------------------------------------------- #
    if n_critical > 0 or overall < cfg["manual_review_below"]:
        route = "MANUAL_REVIEW"
    elif (text_score < cfg["retry_text_below"] or
          (total_pdf_tables and table_score is not None
           and table_score < cfg["retry_table_below"])):
        route = "RETRY_OCR_LLM"
    elif overall >= cfg["min_overall"] and n_major == 0:
        route = "PASS"
    else:
        route = "MANUAL_REVIEW"

    return {
        "document": bundle["manifest"]["source_pdf"]["base_name"],
        "overall_score": round(overall, 1),
        "route": route,
        "scores": {k: round(v * 100, 1) for k, v in components.items()},
        "page_count": {"pdf": n_pages_pdf, "marker_meta": n_pages_meta,
                       "consistent": page_consistency_ok},
        "tables": {"pdf_fragments": total_pdf_tables, "markdown": total_md_tables,
                   "engine": pdf_tbl_engine, "content_check": table_debug},
        "severity_counts": {"critical": n_critical, "major": n_major, "minor": n_minor},
        "per_page_text_recall": page_recalls,
        "inaccurate_sections": section_flags,   # <-- WHICH sections are off
        "findings": findings,
        "thresholds": {k: cfg[k] for k in
                       ("min_overall", "min_table", "manual_review_below",
                        "retry_text_below", "retry_table_below")},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_CFG = {
    "weights": {"text": 0.40, "structure": 0.20, "table": 0.25,
                "cleanliness": 0.10, "stability": 0.05},
    "min_overall": 85.0,          # overall pass threshold
    "min_table": 70.0,            # min table score for table-heavy docs
    "manual_review_below": 70.0,  # below this -> always manual review
    "retry_text_below": 0.80,     # weak text -> retry with OCR/LLM
    "retry_table_below": 0.50,    # weak tables -> retry
    "page_recall_critical": 0.35, # per-page recall below -> critical
    "page_recall_major": 0.65,    # per-page recall below -> major
    # --- table CONTENT thresholds (cell text + numbers) --- #
    "table_cell_critical": 0.50,    # cell-text recall below -> critical
    "table_cell_major": 0.80,       # cell-text recall below -> major
    "table_numeric_critical": 0.60, # numeric recall below -> critical
    "table_numeric_major": 0.85,    # numeric recall below -> major
    # --- tick-mark consistency (corrupted-checkmark detection) --- #
    "tick_col_ratio": 0.50,         # column counts as tick-column if ticks dominate
    "tick_min_per_col": 3,          # need >=3 valid ticks to treat col as tick-column
    "tick_penalty_each": 0.03,      # table_score penalty per suspect cell (capped)
    # --- cell-boundary bleed (NEW) --- #
    "bleed_penalty_each": 0.04,     # table_score penalty per bleed finding
    # --- per-row recall (NEW) --- #
    "per_row_recall_threshold": 0.50,  # flag rows below this token recall
    "per_row_penalty_each": 0.02,      # table_score penalty per low-recall row
    # --- marker JSON cell fidelity (NEW) --- #
    "marker_cell_bleed_threshold": 0.85,  # row-level recall below this triggers cell diff
    "marker_cell_penalty_each": 0.03,     # table_score penalty per bleed cell found
}


def main():
    ap = argparse.ArgumentParser(description="Verify a PDF->Markdown conversion bundle.")
    ap.add_argument("--bundle", required=True, help="Path to verification/manifest.json")
    ap.add_argument("--report", default=None, help="Where to write JSON report")
    ap.add_argument("--min-overall", type=float, default=DEFAULT_CFG["min_overall"])
    ap.add_argument("--min-table", type=float, default=DEFAULT_CFG["min_table"])
    ap.add_argument("--manual-below", type=float, default=DEFAULT_CFG["manual_review_below"])
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg["min_overall"] = args.min_overall
    cfg["min_table"] = args.min_table
    cfg["manual_review_below"] = args.manual_below

    bundle = load_bundle(args.bundle)
    report = verify(bundle, cfg)

    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(args.bundle)), "verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- human-readable summary ---- #
    print(f"Document : {report['document']}")
    print(f"Overall  : {report['overall_score']}/100   ->  ROUTE: {report['route']}")
    print(f"Scores   : {report['scores']}")
    print(f"Pages    : PDF={report['page_count']['pdf']} "
          f"Marker={report['page_count']['marker_meta']} "
          f"consistent={report['page_count']['consistent']}")
    tb = report['tables']
    print(f"Tables   : PDF_fragments={tb['pdf_fragments']} MD={tb['markdown']} engine={tb['engine']}")
    if tb.get('content_check'):
        cc = tb['content_check']
        print(f"           cell_text_recall={cc['cell_text_recall']:.0%} "
              f"numeric_recall={cc['numeric_recall']:.0%} "
              f"(pdf_numbers={cc['pdf_numbers']}, missing={cc['missing_numbers']})")
    print(f"Severity : {report['severity_counts']}")
    if report["inaccurate_sections"]:
        print("\nInaccurate sections (route to review):")
        for s in report["inaccurate_sections"]:
            loc = f"p{s.get('page')}" if s.get("page") else "-"
            print(f"  [{s['severity'].upper():8}] {loc:4} {s['dimension']:9} {s['detail']}")
    print(f"\nFull report -> {report_path}")


if __name__ == "__main__":
    main()
