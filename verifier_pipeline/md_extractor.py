"""
md_extractor.py — Detect and parse GitHub-style Markdown tables.

Public API
----------
extract_markdown_tables(text)  -> list[dict]   (start_line, end_line, raw_text, rows)
parse_table(raw_text)          -> dict          (header, separator, body_rows, col_counts, all_cells)
"""
from __future__ import annotations

import re

# ── compiled patterns ─────────────────────────────────────────────────────────
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_SEP_RE       = re.compile(r"^\s*\|[\s\-:]+(\|[\s\-:]+)+\|\s*$")


# ── extraction ────────────────────────────────────────────────────────────────

def extract_markdown_tables(text: str) -> list[dict]:
    """
    Detect GitHub-style Markdown tables preserving original 1-based line numbers.

    Returns a list of dicts, each with:
        start_line  int   1-based line number of first table row
        end_line    int   1-based line number of last table row (inclusive)
        raw_text    str   raw Markdown block
        rows        list[str]
    """
    lines  = text.split("\n")
    tables: list[dict] = []
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            has_sep = any(
                _SEP_RE.match(lines[k]) for k in range(i, min(i + 3, j))
            )
            if has_sep and (j - i) >= 2:
                tables.append({
                    "start_line": i + 1,   # 1-based
                    "end_line":   j,
                    "raw_text":   "\n".join(lines[i:j]),
                    "rows":       [lines[k] for k in range(i, j)],
                })
                i = j
                continue
        i += 1
    return tables


# ── parsing ───────────────────────────────────────────────────────────────────

def _split_row(row: str) -> list[str]:
    """Split a Markdown pipe-delimited row; strip surrounding pipes and whitespace."""
    parts = row.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def parse_table(raw_text: str) -> dict:
    """
    Parse a raw Markdown table block into structured components.

    Returns dict with:
        header      list[str]
        separator   list[str]
        body_rows   list[list[str]]
        col_counts  list[int]   per row (header + body)
        all_cells   list[str]  flat
    """
    lines = [ln for ln in raw_text.split("\n") if _TABLE_ROW_RE.match(ln)]
    if not lines:
        return dict(header=[], separator=[], body_rows=[], col_counts=[], all_cells=[])

    header_row  = _split_row(lines[0])
    sep_idx     = next((i for i, ln in enumerate(lines) if _SEP_RE.match(ln)), 1)
    sep_row     = _split_row(lines[sep_idx])
    body_lines  = [lines[i] for i in range(len(lines)) if i != sep_idx and i > 0]
    body_rows   = [_split_row(ln) for ln in body_lines]

    col_counts  = [len(header_row)] + [len(r) for r in body_rows]
    all_cells   = list(header_row) + [c for row in body_rows for c in row]

    return dict(
        header    = header_row,
        separator = sep_row,
        body_rows = body_rows,
        col_counts= col_counts,
        all_cells = all_cells,
    )
