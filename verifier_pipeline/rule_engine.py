"""
rule_engine.py — Apply derived rules to a single parsed Markdown table.

Public API
----------
apply_rules_to_table(table_data, parsed, rules)
    -> (triggered_rules, errors, confidence, verdict)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from models import Rule, ErrorInstance, TriggeredRule


# ── compiled patterns ─────────────────────────────────────────────────────────
_NOISE_TICKS      = re.compile(r"(?<!\w)[,\.•\*\~\\\/](?!\w)|<b>\s*[Vv√~]\s*</b>")
_TRAILING_ARTIFACT= re.compile(r"[✓✗✔\w\$\d]\s*[_\-\.]$")
_NUMERIC          = re.compile(r"(HK\$|RMB|USD|US\$|[\$£€¥])\s*[\d,]+|^\d[\d,\.]+$|^\d+%")
_PROSE_SENTENCE   = re.compile(r"^[A-Z][a-z].{30,}[\.a-z]$")
_MULTI_VALUE      = re.compile(
    r"(Waived\s+Waived|HK\$\d+\s+HK\$\d+|Nil\s+Nil|[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)"
)
_OCR_NOISE        = re.compile(r"[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\s+[a-zA-Z]{3,}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _cell_boundary_bleed(row: list[str]) -> list[str]:
    """Return evidence strings for split-word cell pairs within a single row."""
    evidence = []
    for i in range(len(row) - 1):
        a, b = row[i], row[i + 1]
        if (a and b
                and not re.search(r"[\s\.,;:\!\?)\]\'\"✓✗]$", a)
                and re.match(r"^[a-z]", b)
                and len(a) <= 6 and len(b) <= 6):
            evidence.append(f"'{a}' | '{b}'")
    return evidence


# ── main engine ───────────────────────────────────────────────────────────────

def apply_rules_to_table(
    table_data: dict,
    parsed: dict,
    rules: list[Rule],
) -> tuple[list[TriggeredRule], list[ErrorInstance], float, str]:
    """
    Apply all rules to a parsed Markdown table.

    Parameters
    ----------
    table_data : dict   output of extract_markdown_tables (contains raw_text, rows)
    parsed     : dict   output of parse_table (header, body_rows, col_counts, …)
    rules      : list   derived Rule objects

    Returns
    -------
    triggered_rules, errors, confidence, verdict
    """
    triggered: list[TriggeredRule]  = []
    errors:    list[ErrorInstance]  = []
    rule_map                        = {r.rule_id: r for r in rules}

    header     = parsed["header"]
    body_rows  = parsed["body_rows"]
    col_counts = parsed["col_counts"]

    modal_count   = Counter(col_counts).most_common(1)[0][0] if col_counts else 0
    header_count  = len(header)

    def _hit(rule_id: str, evidence: str, error_code: str,
             message: str, severity: str = "medium") -> None:
        r = rule_map.get(rule_id)
        if not r:
            return
        triggered.append(TriggeredRule(
            rule_id          = rule_id,
            rule_name        = r.rule_name,
            confidence_weight= r.confidence_weight,
            matched_evidence = evidence[:200],
        ))
        errors.append(ErrorInstance(
            code     = error_code,
            message  = message,
            evidence = evidence[:200],
            severity = severity,
        ))

    # R01 — Cell boundary bleed
    for row in body_rows:
        for ev in _cell_boundary_bleed(row):
            _hit("R01", ev, "CELL_BOUNDARY_BLEED",
                 "A word appears to be split across adjacent cells.", "high")

    # R02 — Row width mismatch
    for idx, row in enumerate(body_rows):
        if len(row) != modal_count and modal_count > 0:
            ev = f"Row {idx+1}: {len(row)} cols vs expected {modal_count}"
            _hit("R02", ev, "ROW_WIDTH_MISMATCH",
                 f"Body row has {len(row)} columns; table modal is {modal_count}.", "medium")

    # R03 — Merged cell collapse
    for row in body_rows:
        for cell in row:
            if _MULTI_VALUE.search(cell):
                _hit("R03", cell[:120], "MERGED_CELL_COLLAPSE",
                     "Multiple distinct values are concatenated inside a single cell.", "high")
                break

    # R04 — Corrupted symbol
    for row in body_rows:
        for cell in row:
            if _NOISE_TICKS.search(cell) and len(cell.strip()) <= 12:
                _hit("R04", repr(cell), "CORRUPTED_SYMBOL",
                     "Cell contains a noise character where a symbol (✓/✗) is expected.", "high")
                break

    # R05 — Trailing artifact
    for row in body_rows:
        for cell in row:
            if _TRAILING_ARTIFACT.search(cell):
                _hit("R05", repr(cell), "TRAILING_ARTIFACT",
                     "Cell has a trailing artifact (underscore, stray dash/period).", "low")
                break

    # R06 — Row split
    for i in range(len(body_rows) - 1):
        a_label = body_rows[i][0]   if body_rows[i]   else ""
        b_label = body_rows[i+1][0] if body_rows[i+1] else ""
        if (a_label and b_label
                and len(a_label.split()) <= 4
                and b_label[0].islower()):
            ev = f"Row {i+1} label='{a_label}' | Row {i+2} label='{b_label}'"
            _hit("R06", ev, "CELL_ROW_SPLIT",
                 "Two consecutive rows appear to be a single logical row split by the extractor.",
                 "high")

    # R07 — Phantom / OCR noise
    for row in body_rows:
        for cell in row:
            if _OCR_NOISE.search(cell) and not re.search(r"[\$\d]", cell):
                _hit("R07", cell[:120], "PHANTOM_CONTENT",
                     "Cell contains OCR noise text absent from the source document.", "high")
                break

    # R09 — Missing numeric in numeric column
    if body_rows and header:
        col_numeric_ratio: list[float] = []
        for col_i in range(len(header)):
            vals  = [row[col_i] for row in body_rows
                     if col_i < len(row) and row[col_i].strip()]
            ratio = (sum(1 for v in vals if _NUMERIC.search(v)) / len(vals)) if vals else 0.0
            col_numeric_ratio.append(ratio)

        for row_idx, row in enumerate(body_rows):
            for col_i, cell in enumerate(row):
                if (col_i < len(col_numeric_ratio)
                        and col_numeric_ratio[col_i] >= 0.5
                        and not cell.strip()):
                    hdr = header[col_i] if col_i < len(header) else "?"
                    ev  = f"Row {row_idx+1}, col {col_i+1} (header='{hdr}'): empty"
                    _hit("R09", ev, "MISSING_NUMERIC",
                         "Empty cell in a column that should contain a numeric/currency value.",
                         "medium")
                    break

    # R10 — Caption absorbed into table
    if header:
        first_h = " ".join(header)
        if _PROSE_SENTENCE.match(first_h) or len(first_h) > 80:
            _hit("R10", first_h[:100], "CAPTION_IN_TABLE",
                 "The header row appears to be a prose caption rather than column labels.",
                 "medium")
    if body_rows:
        first_cell = body_rows[0][0] if body_rows[0] else ""
        if _PROSE_SENTENCE.match(first_cell) and len(first_cell) > 60:
            _hit("R10", first_cell[:100], "CAPTION_IN_TABLE",
                 "First body row appears to be a prose caption absorbed into the table.",
                 "medium")

    # R11 — Extra columns
    if header_count > 8:
        empty_header_cols = sum(1 for h in header if not h.strip())
        if empty_header_cols > header_count // 3:
            ev = f"{empty_header_cols}/{header_count} header cells are empty"
            _hit("R11", ev, "EXTRA_COLUMNS",
                 "Table has an unusually large number of empty header columns.", "medium")

    # R12 — Truncated text
    for row in body_rows:
        for cell in row:
            if (len(cell) > 60
                    and re.search(r"\b(and|or|the|of|to|on|in|for|with|a|an)\s*$",
                                  cell, re.I)):
                _hit("R12", cell[-40:], "TRUNCATED_TEXT",
                     "Cell text ends abruptly on a common word, suggesting truncation.",
                     "medium")
                break

    # ── aggregate confidence ──────────────────────────────────────────────────
    if triggered:
        product    = 1.0
        for t in triggered:
            product *= (1.0 - t.confidence_weight)
        confidence = round(min(1.0 - product, 0.99), 3)
        verdict    = "INCORRECT"
    else:
        confidence = 0.05
        verdict    = "CORRECT"

    return triggered, errors, confidence, verdict
