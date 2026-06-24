#!/usr/bin/env python3
"""
PDF-to-Markdown Table Verifier — v2.0
Supports partial labels: CORRECT | PARTIAL_CORRECT | PARTIAL_INCORRECT | INCORRECT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

try:
    from rapidfuzz import fuzz  # type: ignore
    def _fuzzy(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b) / 100.0
except ImportError:
    import difflib
    def _fuzzy(a: str, b: str) -> float:  # type: ignore
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT LEVELS (ordered by severity)
# ═══════════════════════════════════════════════════════════════════════════════

VERDICT_CORRECT           = "CORRECT"
VERDICT_PARTIAL_CORRECT   = "PARTIAL_CORRECT"
VERDICT_PARTIAL_INCORRECT = "PARTIAL_INCORRECT"
VERDICT_INCORRECT         = "INCORRECT"

# Correct-row ratio thresholds that define each verdict tier
THRESHOLD_CORRECT           = 1.0   # == 1.0  → CORRECT
THRESHOLD_PARTIAL_CORRECT   = 0.80  # >= 0.80 → PARTIAL_CORRECT
THRESHOLD_PARTIAL_INCORRECT = 0.40  # >= 0.40 → PARTIAL_INCORRECT
                                     # <  0.40 → INCORRECT


def _verdict_from_ratio(correct_ratio: float) -> str:
    """Map a correct-row ratio to the four-level verdict."""
    if correct_ratio >= THRESHOLD_CORRECT:
        return VERDICT_CORRECT
    if correct_ratio >= THRESHOLD_PARTIAL_CORRECT:
        return VERDICT_PARTIAL_CORRECT
    if correct_ratio >= THRESHOLD_PARTIAL_INCORRECT:
        return VERDICT_PARTIAL_INCORRECT
    return VERDICT_INCORRECT


def _verdict_from_errors(triggered_rules: list, error_codes: set) -> tuple[str, float]:
    """
    When we have no explicit correct_ratio (pure heuristic run),
    derive the verdict from which rules fired and how severe they are.

    Returns (verdict, estimated_correct_ratio).
    """
    if not triggered_rules:
        return VERDICT_CORRECT, 1.0

    # Structural errors that indicate the table is fundamentally broken
    structural_errors = {
        "TABLE_MERGE", "MISSING_COLUMNS", "EXTRA_COLUMNS",
        "MERGED_CELL_COLLAPSE", "SECTION_TRUNCATED",
    }
    # Moderate errors – some data lost
    moderate_errors = {
        "MISSING_ROW", "CELL_ROW_SPLIT", "CORRUPTED_SYMBOL",
        "WRONG_VALUE", "PHANTOM_CONTENT", "CAPTION_IN_TABLE",
    }
    # Minor / cosmetic errors
    minor_errors = {
        "CELL_BOUNDARY_BLEED", "TRAILING_ARTIFACT", "FOOTNOTE_ERROR",
        "WRONG_SYMBOL", "TRUNCATED_TEXT",
    }

    has_structural = bool(error_codes & structural_errors)
    has_moderate   = bool(error_codes & moderate_errors)
    has_minor      = bool(error_codes & minor_errors)

    if has_structural:
        # Estimate ratio based on how many structural codes fired
        n_structural = len(error_codes & structural_errors)
        est_ratio = max(0.0, 0.35 - 0.05 * n_structural)
        return _verdict_from_ratio(est_ratio), round(est_ratio, 2)

    if has_moderate:
        n_moderate = len(error_codes & moderate_errors)
        est_ratio = max(0.0, 0.70 - 0.10 * n_moderate)
        return _verdict_from_ratio(est_ratio), round(est_ratio, 2)

    if has_minor:
        n_minor = len(error_codes & minor_errors)
        est_ratio = max(0.0, 0.90 - 0.05 * n_minor)
        return _verdict_from_ratio(est_ratio), round(est_ratio, 2)

    return VERDICT_CORRECT, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rule:
    rule_id: str
    rule_name: str
    description: str
    why_it_indicates_an_error: str
    confidence_weight: float
    partial_impact: str   # "minor" | "moderate" | "structural"
    examples_from_dataset: list[str] = field(default_factory=list)


@dataclass
class ErrorInstance:
    code: str
    message: str
    evidence: str
    severity: str   # "high" | "medium" | "low"


@dataclass
class TriggeredRule:
    rule_id: str
    rule_name: str
    confidence_weight: float
    matched_evidence: str


@dataclass
class SourceRange:
    start_line: int
    end_line: int


@dataclass
class PartialDetail:
    """Mirrors the partial_detail block in table_dataset.json."""
    correct_rows: list
    incorrect_rows: list
    correct_ratio: float
    note: str = ""


@dataclass
class TableReport:
    table_id: str
    markdown_source_range: SourceRange
    page: Optional[int]
    section: Optional[str]
    pdf_table_index: Optional[int]
    pdf_match_confidence: float
    verdict: str       # CORRECT | PARTIAL_CORRECT | PARTIAL_INCORRECT | INCORRECT
    confidence: float
    estimated_correct_ratio: float
    partial_detail: Optional[PartialDetail]
    triggered_rules: list[TriggeredRule]
    errors: list[ErrorInstance]
    table_excerpt: str
    suggested_fix: str
    notes: list[str]


@dataclass
class VerificationReport:
    document_name: str
    verifier_version: str
    generated_at: str
    rules: list[Rule]
    tables: list[TableReport]
    summary: dict


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

def load_examples(path: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns (correct, partial_correct, partial_incorrect, incorrect) grouped by
    the four verdict tiers, derived from label + partial_detail.correct_ratio.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    correct, partial_correct, partial_incorrect, incorrect = [], [], [], []
    for ex in raw:
        label = ex.get("label", "").upper()
        pd = ex.get("partial_detail", {}) or {}
        ratio = pd.get("correct_ratio", 1.0 if label == "CORRECT" else 0.0)

        if label == "CORRECT":
            correct.append(ex)
        else:
            verdict = _verdict_from_ratio(ratio)
            if verdict == VERDICT_PARTIAL_CORRECT:
                partial_correct.append(ex)
            elif verdict == VERDICT_PARTIAL_INCORRECT:
                partial_incorrect.append(ex)
            else:
                incorrect.append(ex)

    return correct, partial_correct, partial_incorrect, incorrect


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARKDOWN TABLE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_SEP_RE       = re.compile(r"^\s*\|[\s\-:]+(\|[\s\-:]+)+\|\s*$")


def extract_markdown_tables(text: str) -> list[dict]:
    lines = text.split("\n")
    tables, i = [], 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            has_sep = any(_SEP_RE.match(lines[k]) for k in range(i, min(i + 3, j)))
            if has_sep and (j - i) >= 2:
                raw = "\n".join(lines[i:j])
                tables.append({
                    "start_line": i + 1,
                    "end_line": j,
                    "raw_text": raw,
                    "rows": [lines[k] for k in range(i, j)],
                })
                i = j
                continue
        i += 1
    return tables


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABLE PARSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _split_row(row: str) -> list[str]:
    parts = row.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def parse_table(raw_text: str) -> dict:
    lines = [l for l in raw_text.split("\n") if _TABLE_ROW_RE.match(l)]
    if not lines:
        return dict(header=[], separator=[], body_rows=[], col_counts=[], all_cells=[])
    header_row = _split_row(lines[0])
    sep_idx = next((i for i, l in enumerate(lines) if _SEP_RE.match(l)), 1)
    separator_row = _split_row(lines[sep_idx])
    body_lines = [lines[i] for i in range(len(lines)) if i != sep_idx and i > 0]
    body_rows = [_split_row(l) for l in body_lines]
    col_counts = [len(header_row)] + [len(r) for r in body_rows]
    all_cells = list(header_row) + [c for row in body_rows for c in row]
    return dict(
        header=header_row,
        separator=separator_row,
        body_rows=body_rows,
        col_counts=col_counts,
        all_cells=all_cells,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TAXONOMY & RULE DERIVATION
# ═══════════════════════════════════════════════════════════════════════════════

TAXONOMY: dict[str, tuple[str, str]] = {
    # (description, partial_impact)
    "CELL_BOUNDARY_BLEED":      ("A word or phrase is split across adjacent cells.", "minor"),
    "BROKEN_WORD_FRAGMENT":     ("A cell contains a non-word token that is part of a longer word.", "minor"),
    "ROW_WIDTH_MISMATCH":       ("A body row has a different column count than the header.", "moderate"),
    "HEADER_DATA_MISALIGNMENT": ("Header column count differs from separator or data rows.", "moderate"),
    "MISSING_NUMERIC":          ("A cell expected to hold a numeric/currency value is empty.", "moderate"),
    "MERGED_CELL_COLLAPSE":     ("Multiple distinct values are concatenated inside one cell.", "structural"),
    "TRUNCATED_TEXT":           ("A cell or section is cut off mid-sentence or mid-word.", "moderate"),
    "COLUMN_SHIFT":             ("Data values are shifted into the wrong column.", "moderate"),
    "MALFORMED_PERCENT":        ("A percentage or rate value is malformed or split.", "minor"),
    "INCONSISTENT_COLUMN_COUNT":("Different rows have inconsistent column counts.", "moderate"),
    "CORRUPTED_SYMBOL":         ("A tick, checkmark, or symbol is replaced with noise characters.", "moderate"),
    "PHANTOM_CONTENT":          ("A cell contains OCR noise absent from the source PDF.", "moderate"),
    "FOOTNOTE_ERROR":           ("Footnote markers or text are missing, reordered, or truncated.", "minor"),
    "CELL_ROW_SPLIT":           ("A single logical row is split into two or more rows.", "moderate"),
    "TRAILING_ARTIFACT":        ("A cell has trailing punctuation or symbol bleed.", "minor"),
    "EXTRA_COLUMNS":            ("The Markdown table has more columns than the source PDF.", "structural"),
    "CAPTION_IN_TABLE":         ("A section caption is absorbed as a table row.", "moderate"),
    # ── NEW in v2 ──────────────────────────────────────────────────────────────
    "WRONG_VALUE":              ("A cell holds a completely wrong value vs the PDF source.", "moderate"),
    "MISSING_ROW":              ("One or more entire rows are absent from the markdown table.", "moderate"),
    "WRONG_SYMBOL":             ("An annotation/footnote marker is corrupted (e.g. *-* for ±,*).", "minor"),
    "TABLE_MERGE":              ("Two logically separate PDF tables are collapsed into one table.", "structural"),
    "MISSING_COLUMNS":          ("Markdown table has fewer columns than the source PDF table.", "structural"),
    "SECTION_TRUNCATED":        ("Table ends abruptly; rows at the bottom are missing.", "structural"),
}


def derive_rules_from_examples(
    correct: list[dict],
    partial_correct: list[dict],
    partial_incorrect: list[dict],
    incorrect: list[dict],
) -> list[Rule]:
    all_incorrect = partial_correct + partial_incorrect + incorrect
    total_inc = max(len(all_incorrect), 1)
    total_cor = max(len(correct), 1)

    error_counts: Counter = Counter()
    code_to_examples: dict[str, list[str]] = defaultdict(list)
    for ex in all_incorrect:
        reason = ex.get("error_reason", "") or ""
        for code in TAXONOMY:
            if code in reason:
                error_counts[code] += 1
                code_to_examples[code].append(ex["table_id"])

    fp_counts: Counter = Counter()
    for ex in correct:
        md = ex.get("md_table", "")
        parsed = parse_table(md)
        counts = parsed["col_counts"]
        if counts and len(set(counts)) > 1:
            fp_counts["ROW_WIDTH_MISMATCH"] += 1

    def _weight(code: str, base: float) -> float:
        freq = error_counts.get(code, 0) / total_inc
        fp_rate = fp_counts.get(code, 0) / total_cor
        w = base * (0.6 + 0.4 * freq) * max(0.1, 1 - fp_rate)
        return round(min(max(w, 0.05), 0.99), 2)

    rules: list[Rule] = []

    def _mk(rid, name, desc, why, base, code):
        _, impact = TAXONOMY.get(code, ("", "moderate"))
        rules.append(Rule(
            rule_id=rid, rule_name=name,
            description=desc, why_it_indicates_an_error=why,
            confidence_weight=_weight(code, base),
            partial_impact=impact,
            examples_from_dataset=code_to_examples.get(code, []),
        ))

    _mk("R01", "Cell boundary bleed",
        "Detects tokens split across adjacent cells (half-word in each cell).",
        "PDF column boundaries inserted inside a text run split a word into two cells.",
        0.93, "CELL_BOUNDARY_BLEED")

    _mk("R02", "Row width mismatch",
        "Flags rows whose column count differs from the modal count of the table.",
        "Merged/split cells produce rows with too few or too many columns.",
        0.88, "ROW_WIDTH_MISMATCH")

    _mk("R03", "Merged cell collapse",
        "Detects cells with multiple distinct values concatenated (e.g. 'Waived Waived').",
        "Failed column-boundary detection collapses tier values into one cell.",
        0.90, "MERGED_CELL_COLLAPSE")

    _mk("R04", "Corrupted symbol",
        "Detects noise characters substituting for ✓/✗ (commas, bullets, raw HTML).",
        "OCR mis-reads graphical tick marks as nearby ASCII noise.",
        0.91, "CORRUPTED_SYMBOL")

    _mk("R05", "Trailing artifact",
        "Detects trailing underscore, dash, or stray punctuation after a valid value.",
        "Typographic rules or table borders bleed into adjacent cell text.",
        0.82, "TRAILING_ARTIFACT")

    _mk("R06", "Row split",
        "Two consecutive rows whose first cells appear to be one logical label split in two.",
        "Row-level extraction inserts spurious line breaks inside wrapped cells.",
        0.87, "CELL_ROW_SPLIT")

    _mk("R07", "Phantom / OCR noise content",
        "Flags cells with garbled, implausible text absent from the source (e.g. 'lakada ada').",
        "OCR artefacts from non-text PDF regions inject random characters.",
        0.85, "PHANTOM_CONTENT")

    _mk("R08", "Footnote error",
        "Detects footnote blocks that are reordered, missing their numeric marker, or truncated.",
        "Converters often mishandle footnote numbering and ordering.",
        0.78, "FOOTNOTE_ERROR")

    _mk("R09", "Missing numeric / currency value",
        "In a numeric-dominant column, flags unexpectedly empty cells.",
        "Column-shift or extraction failures leave numeric cells blank.",
        0.76, "MISSING_NUMERIC")

    _mk("R10", "Caption absorbed into table",
        "Detects when the first table row is a prose sentence rather than a column header.",
        "Imprecise bounding-box detection absorbs captions as table rows.",
        0.80, "CAPTION_IN_TABLE")

    _mk("R11", "Extra columns",
        "Flags tables with more columns than expected (many empty header cells).",
        "PDF extraction mistakes whitespace/alignment guides for column separators.",
        0.83, "EXTRA_COLUMNS")

    _mk("R12", "Truncated text",
        "Detects cells/footnotes ending abruptly on a common word (preposition, conjunction).",
        "Page-boundary or buffer issues silently truncate content.",
        0.79, "TRUNCATED_TEXT")

    # ── New rules (v2) ─────────────────────────────────────────────────────────

    _mk("R13", "Wrong value",
        "A cell holds a wrong value: wrong currency amount, wrong word, or inverted flag.",
        "OCR digit confusion, column shift, or footnote marker merging produces wrong values.",
        0.88, "WRONG_VALUE")

    _mk("R14", "Missing row",
        "Entire rows are absent from the markdown table compared to the PDF source.",
        "Extractor skips rows when it cannot resolve cell boundaries or page-edge content.",
        0.85, "MISSING_ROW")

    _mk("R15", "Wrong annotation / footnote symbol",
        "An annotation marker (e.g. ±,* → *-*) is corrupted or replaced with wrong character.",
        "Superscript and special marker symbols are frequently mis-read by OCR.",
        0.75, "WRONG_SYMBOL")

    _mk("R16", "Table merge (two tables collapsed into one)",
        "Two logically separate PDF tables appear merged into a single markdown table.",
        "When adjacent tables share a visual boundary the extractor treats them as one.",
        0.92, "TABLE_MERGE")

    _mk("R17", "Missing columns",
        "Markdown table has fewer columns than the PDF source (tier columns dropped).",
        "Column-detection failures cause entire charge-tier columns to vanish.",
        0.90, "MISSING_COLUMNS")

    _mk("R18", "Section truncated (rows missing from bottom)",
        "The table ends abruptly; rows present in the PDF are absent from the markdown tail.",
        "Page-break handling or extraction buffer limits cut the table short.",
        0.84, "SECTION_TRUNCATED")

    return rules


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RULE APPLICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_NOISE_TICKS      = re.compile(r"(?<!\w)[,\.•\*\~\\/](?!\w)|<b>\s*[Vv√~]\s*</b>")
_TRAILING_ARTIFACT= re.compile(r"[✓✗✔\w\$\d]\s*[_\-\.]$")
_NUMERIC          = re.compile(r"(HK\$|RMB|USD|US\$|[\$£€¥])\s*[\d,]+|^\d[\d,\.]+$|^\d+%")
_PROSE_SENTENCE   = re.compile(r"^[A-Z][a-z].{30,}[\.a-z]$")
_MULTI_VALUE      = re.compile(
    r"(Waived\s+Waived|HK\$\d+\s+HK\$\d+|Nil\s+Nil|[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)"
)
_OCR_NOISE        = re.compile(r"[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\s+[a-zA-Z]{3,}")
# Wrong-value heuristics
_BACKSLASH_IN_CELL= re.compile(r"^\\$")   # lone backslash
_COMMA_IN_CELL    = re.compile(r"^,$")       # lone comma
_GARBAGE_MARKER   = re.compile(r"^[\*\-\\,\.]{1,3}$")  # single punctuation only
# Table merge signal: bold section header row inside table body
_BOLD_SECTION_HDR = re.compile(r"\*\*\[?(C\d|Section|[A-Z]\d)\b")
# Missing columns signal: expected 4-5 charge-tier columns but only 1-2 present
_CHARGE_HEADER    = re.compile(r"Charge|Amount|Fee|Rate", re.I)
# Footnote marker detection: paragraph starting with a digit + space or superscript
_FOOTNOTE_START   = re.compile(r"^\s*(\d+|¹|²|³|⁴|⁵|⁶)[\s\.]")
# Wrong annotation symbol: asterisk-dash-asterisk pattern
_WRONG_ANNOT_SYM  = re.compile(r"\*[-–]\s*\*")
# Section truncation: last body row is empty or single char
_TRUNCATED_LAST   = re.compile(r"^\s*\|?\s*\(?truncat|abruptly|\.\.\.")


def _cell_boundary_bleed(row: list[str]) -> list[str]:
    evidence = []
    for i in range(len(row) - 1):
        a, b = row[i], row[i + 1]
        if (a and b and
                not re.search(r"[\s\.,;:\!\?)\]\'\"✓✗]$", a) and
                re.match(r"^[a-z]", b) and
                len(a) <= 6 and len(b) <= 6):
            evidence.append(f"'{a}' | '{b}'")
    return evidence


def apply_rules_to_table(
    table_data: dict,
    parsed: dict,
    rules: list[Rule],
) -> tuple[list[TriggeredRule], list[ErrorInstance], float, str, float]:
    """
    Apply all rules to a parsed table.

    Returns:
        triggered_rules, errors, confidence, verdict, estimated_correct_ratio
    """
    triggered: list[TriggeredRule] = []
    errors: list[ErrorInstance] = []
    rule_map = {r.rule_id: r for r in rules}

    raw_text  = table_data["raw_text"]
    body_rows = parsed["body_rows"]
    header    = parsed["header"]
    col_counts= parsed["col_counts"]

    header_count = len(header)
    modal_count  = Counter(col_counts).most_common(1)[0][0] if col_counts else 0

    def _hit(rule_id: str, evidence: str, code: str, message: str, severity: str = "medium"):
        r = rule_map.get(rule_id)
        if not r:
            return
        triggered.append(TriggeredRule(
            rule_id=rule_id,
            rule_name=r.rule_name,
            confidence_weight=r.confidence_weight,
            matched_evidence=evidence[:200],
        ))
        errors.append(ErrorInstance(
            code=code,
            message=message,
            evidence=evidence[:200],
            severity=severity,
        ))

    # ── R01: Cell boundary bleed ──────────────────────────────────────────────
    for row in body_rows:
        for ev in _cell_boundary_bleed(row):
            _hit("R01", ev, "CELL_BOUNDARY_BLEED",
                 "A word appears split across adjacent cells.", "high")

    # ── R02: Row width mismatch ───────────────────────────────────────────────
    for idx, row in enumerate(body_rows):
        if len(row) != modal_count and modal_count > 0:
            _hit("R02", f"Row {idx+1}: {len(row)} cols vs expected {modal_count}",
                 "ROW_WIDTH_MISMATCH",
                 f"Body row has {len(row)} columns; table modal is {modal_count}.",
                 "medium")

    # ── R03: Merged cell collapse ─────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if _MULTI_VALUE.search(cell):
                _hit("R03", cell[:120], "MERGED_CELL_COLLAPSE",
                     "Multiple distinct values concatenated in a single cell.", "high")
                break

    # ── R04: Corrupted symbol ─────────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if _NOISE_TICKS.search(cell) and len(cell.strip()) <= 12:
                _hit("R04", repr(cell), "CORRUPTED_SYMBOL",
                     "Cell contains noise character where ✓/✗ is expected.", "high")
                break

    # ── R05: Trailing artifact ────────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if _TRAILING_ARTIFACT.search(cell):
                _hit("R05", repr(cell), "TRAILING_ARTIFACT",
                     "Cell has a trailing artifact (underscore, stray dash/period).", "low")
                break

    # ── R06: Row split ────────────────────────────────────────────────────────
    for i in range(len(body_rows) - 1):
        a_label = body_rows[i][0]   if body_rows[i]   else ""
        b_label = body_rows[i+1][0] if body_rows[i+1] else ""
        if (a_label and b_label and
                len(a_label.split()) <= 4 and
                b_label and b_label[0].islower()):
            _hit("R06",
                 f"Row {i+1} label='{a_label}' | Row {i+2} label='{b_label}'",
                 "CELL_ROW_SPLIT",
                 "Two consecutive rows appear to be one logical row split by the extractor.",
                 "high")

    # ── R07: Phantom / OCR noise ──────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if _OCR_NOISE.search(cell) and not re.search(r"[\$\d]", cell):
                _hit("R07", cell[:120], "PHANTOM_CONTENT",
                     "Cell contains OCR noise text absent from the source document.", "high")
                break

    # ── R08: Footnote error ───────────────────────────────────────────────────
    # Detect footnote blocks below the table in raw_text
    below_table = raw_text.split("\n")
    ft_lines = [l for l in below_table if not _TABLE_ROW_RE.match(l) and l.strip()]
    if ft_lines:
        found_numbered = [l for l in ft_lines if _FOOTNOTE_START.match(l)]
        found_unnumbered = [l for l in ft_lines if l.strip() and not _FOOTNOTE_START.match(l)
                            and len(l.strip()) > 30]
        if found_unnumbered and not found_numbered:
            _hit("R08", found_unnumbered[0][:80], "FOOTNOTE_ERROR",
                 "Footnote paragraph present but missing its numeric marker.", "medium")

    # ── R09: Missing numeric in numeric column ────────────────────────────────
    if body_rows and header:
        col_numeric_ratio: list[float] = []
        for col_i in range(len(header)):
            vals = [row[col_i] for row in body_rows
                    if col_i < len(row) and row[col_i].strip()]
            ratio = (sum(1 for v in vals if _NUMERIC.search(v)) / len(vals)) if vals else 0.0
            col_numeric_ratio.append(ratio)
        for row_idx, row in enumerate(body_rows):
            for col_i, cell in enumerate(row):
                if (col_i < len(col_numeric_ratio) and
                        col_numeric_ratio[col_i] >= 0.5 and
                        not cell.strip()):
                    _hit("R09",
                         f"Row {row_idx+1}, col {col_i+1} "
                         f"(header='{header[col_i] if col_i < len(header) else '?'}'): empty",
                         "MISSING_NUMERIC",
                         "Empty cell in a column that should contain a numeric/currency value.",
                         "medium")
                    break

    # ── R10: Caption absorbed ─────────────────────────────────────────────────
    if header:
        first_h = " ".join(header)
        if _PROSE_SENTENCE.match(first_h) or len(first_h) > 80:
            _hit("R10", first_h[:100], "CAPTION_IN_TABLE",
                 "Header row appears to be a prose caption rather than column labels.", "medium")
    if body_rows:
        first_cell = body_rows[0][0] if body_rows[0] else ""
        if _PROSE_SENTENCE.match(first_cell) and len(first_cell) > 60:
            _hit("R10", first_cell[:100], "CAPTION_IN_TABLE",
                 "First body row appears to be a prose caption absorbed into the table.", "medium")

    # ── R11: Extra columns ────────────────────────────────────────────────────
    if header_count > 8:
        empty_header_cols = sum(1 for h in header if not h.strip())
        if empty_header_cols > header_count // 3:
            _hit("R11", f"{empty_header_cols}/{header_count} header cells are empty",
                 "EXTRA_COLUMNS",
                 "Table has an unusually large number of empty header columns.", "medium")

    # ── R12: Truncated text ───────────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if (len(cell) > 60 and
                    re.search(r"\b(and|or|the|of|to|on|in|for|with|a|an)\s*$", cell, re.I)):
                _hit("R12", cell[-40:], "TRUNCATED_TEXT",
                     "Cell text ends abruptly on a common word, suggesting truncation.", "medium")
                break

    # ── R13: Wrong value (new) ────────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            stripped = cell.strip()
            if _BACKSLASH_IN_CELL.match(stripped):
                _hit("R13", repr(cell), "WRONG_VALUE",
                     "Cell contains lone backslash where a monetary/tick value is expected.", "high")
                break
            if _COMMA_IN_CELL.match(stripped):
                _hit("R13", repr(cell), "WRONG_VALUE",
                     "Cell contains lone comma where a monetary/tick value is expected.", "high")
                break
            if _GARBAGE_MARKER.match(stripped) and stripped not in ("", "N/A", "Nil", "✓", "✗"):
                _hit("R13", repr(cell), "WRONG_VALUE",
                     "Cell contains a garbage/symbol token where a real value is expected.", "high")
                break

    # ── R14: Missing rows — proxy heuristic (new) ─────────────────────────────
    # If the last 2 body rows are all-empty or single-char, flag potential truncation
    if len(body_rows) >= 2:
        last_row = body_rows[-1]
        all_empty_or_tiny = all(len(c.strip()) <= 1 for c in last_row)
        if all_empty_or_tiny:
            _hit("R14",
                 f"Last row: {last_row}",
                 "MISSING_ROW",
                 "Last table row is empty/degenerate — rows may be missing from the bottom.",
                 "medium")

    # ── R15: Wrong annotation symbol (new) ────────────────────────────────────
    for row in body_rows:
        full_row = " | ".join(row)
        if _WRONG_ANNOT_SYM.search(full_row):
            _hit("R15", full_row[:120], "WRONG_SYMBOL",
                 "Annotation marker is corrupted (asterisk-dash-asterisk pattern detected).",
                 "low")
            break

    # ── R16: Table merge detection (new) ──────────────────────────────────────
    bold_headers_in_body = sum(
        1 for row in body_rows
        if row and _BOLD_SECTION_HDR.search(row[0])
    )
    if bold_headers_in_body >= 2:
        _hit("R16",
             f"{bold_headers_in_body} bold section-header rows found inside table body",
             "TABLE_MERGE",
             "Multiple section headers inside one table body suggest two tables were merged.",
             "high")

    # ── R17: Missing columns (new) ────────────────────────────────────────────
    charge_headers = sum(1 for h in header if _CHARGE_HEADER.search(h))
    if charge_headers == 1 and len(header) <= 2:
        # Classic sign: "Item | Charge" but PDF has 5 columns
        _hit("R17",
             f"Table has only {len(header)} columns but Charge/Fee header detected",
             "MISSING_COLUMNS",
             "Table appears to be missing customer-tier columns (only 1-2 columns present).",
             "high")

    # ── R18: Section truncated (new) ─────────────────────────────────────────
    # If table has fewer body rows than typical AND last row is degenerate
    if body_rows and len(body_rows) <= 3:
        last_row_text = " ".join(body_rows[-1])
        if _TRUNCATED_LAST.search(last_row_text.lower()):
            _hit("R18", last_row_text[:100], "SECTION_TRUNCATED",
                 "Table body ends with a truncation note — rows missing from the bottom.", "high")

    # ── Aggregate confidence and verdict ──────────────────────────────────────
    error_codes = {e.code for e in errors}
    if triggered:
        product = 1.0
        for t in triggered:
            product *= (1.0 - t.confidence_weight)
        confidence = round(min(1.0 - product, 0.99), 3)
        verdict, est_ratio = _verdict_from_errors(triggered, error_codes)
    else:
        confidence = 0.05
        verdict    = VERDICT_CORRECT
        est_ratio  = 1.0

    return triggered, errors, confidence, verdict, est_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PDF MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PDFTableRef:
    page: int
    section: str
    pdf_table_index: int
    header_text: str
    page_text: str


def extract_pdf_refs(pdf_path: Path) -> list[PDFTableRef]:
    if not _HAS_PDFPLUMBER:
        return []
    refs: list[PDFTableRef] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                section = ""
                for line in page_text.split("\n")[:5]:
                    if re.match(r"^[A-Z][0-9\.\s]", line.strip()) or line.isupper():
                        section = line.strip()[:80]
                        break
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables, start=1):
                    header_text = " ".join(
                        cell or "" for cell in (table[0] if table else [])
                    ).strip()
                    refs.append(PDFTableRef(
                        page=page_num,
                        section=section or f"Page {page_num}",
                        pdf_table_index=t_idx,
                        header_text=header_text,
                        page_text=page_text[:400],
                    ))
    except Exception as e:
        print(f"[warn] PDF extraction failed: {e}", file=sys.stderr)
    return refs


def match_table_to_pdf(
    parsed: dict,
    pdf_refs: list[PDFTableRef],
) -> tuple[Optional[int], Optional[str], Optional[int], float]:
    if not pdf_refs:
        return None, None, None, 0.0
    md_header = " ".join(parsed.get("header", []))
    md_sample = md_header + " " + " ".join(
        c for row in (parsed.get("body_rows") or [])[:3] for c in row
    )
    best_score, best_ref = 0.0, None
    for ref in pdf_refs:
        score = _fuzzy(md_sample, ref.header_text + " " + ref.page_text)
        if score > best_score:
            best_score, best_ref = score, ref
    if best_ref:
        return best_ref.page, best_ref.section, best_ref.pdf_table_index, round(best_score, 3)
    return None, None, None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MARKDOWN ANNOTATION — extended for partial labels
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_markdown(
    original_md: str,
    tables_meta: list[dict],
    reports: list[TableReport],
) -> str:
    """
    Wrap each detected table with VERIFIER comment tags that now include
    the four-level verdict and estimated correct ratio.
    """
    lines = original_md.split("\n")
    replacements: list[tuple[int, int, str]] = []

    for meta, report in zip(tables_meta, reports):
        start = meta["start_line"] - 1
        end   = meta["end_line"]

        errors_str = ",".join(e.code for e in report.errors) if report.errors else ""
        verifier_open = (
            f"<!-- VERIFIER: status={report.verdict} | "
            f"confidence={report.confidence:.2f} | "
            f"correct_ratio={report.estimated_correct_ratio:.2f} | "
            f"errors={errors_str} -->"
        )
        verifier_close = "<!-- /VERIFIER -->"

        page_str    = str(report.page)            if report.page            else "unknown"
        section_str = report.section              or "unknown"
        pdf_idx_str = str(report.pdf_table_index) if report.pdf_table_index else "unknown"
        pdf_ref = (
            f'\n<!-- PDF_REF: page={page_str} | '
            f'section="{section_str}" | '
            f'pdf_table_index={pdf_idx_str} -->'
        )

        # Add a human-readable partial label note when relevant
        if report.verdict in (VERDICT_PARTIAL_CORRECT, VERDICT_PARTIAL_INCORRECT):
            pd = report.partial_detail
            pd_note = ""
            if pd:
                pd_note = (
                    f"\n<!-- PARTIAL_DETAIL: "
                    f"correct_rows={len(pd.correct_rows)} "
                    f"incorrect_rows={len(pd.incorrect_rows)} "
                    f"ratio={pd.correct_ratio:.2f}"
                    + (f" | {pd.note}" if pd.note else "")
                    + " -->"
                )
            verifier_close = f"<!-- /VERIFIER -->{pd_note}"

        table_block = "\n".join(lines[start:end])
        annotated = f"{verifier_open}\n{table_block}\n{verifier_close}{pdf_ref}"
        replacements.append((start, end, annotated))

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_verification_report(
    pdf_name: str,
    rules: list[Rule],
    table_reports: list[TableReport],
) -> dict:
    counts = Counter(t.verdict for t in table_reports)
    uncertain = sum(
        1 for t in table_reports
        if t.pdf_match_confidence < 0.35 and t.page is not None
    )
    return {
        "document_name": pdf_name,
        "verifier_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rules": [asdict(r) for r in rules],
        "tables": [
            {
                "table_id": t.table_id,
                "markdown_source_range": asdict(t.markdown_source_range),
                "page": t.page,
                "section": t.section,
                "pdf_table_index": t.pdf_table_index,
                "pdf_match_confidence": t.pdf_match_confidence,
                "verdict": t.verdict,
                "confidence": t.confidence,
                "estimated_correct_ratio": t.estimated_correct_ratio,
                "partial_detail": asdict(t.partial_detail) if t.partial_detail else None,
                "triggered_rules": [asdict(r) for r in t.triggered_rules],
                "errors": [asdict(e) for e in t.errors],
                "table_excerpt": t.table_excerpt,
                "suggested_fix": t.suggested_fix,
                "notes": t.notes,
            }
            for t in table_reports
        ],
        "summary": {
            "total_tables":            len(table_reports),
            "correct_tables":          counts.get(VERDICT_CORRECT, 0),
            "partial_correct_tables":  counts.get(VERDICT_PARTIAL_CORRECT, 0),
            "partial_incorrect_tables":counts.get(VERDICT_PARTIAL_INCORRECT, 0),
            "incorrect_tables":        counts.get(VERDICT_INCORRECT, 0),
            "uncertain_pdf_matches":   uncertain,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _suggested_fix(errors: list[ErrorInstance]) -> str:
    codes = {e.code for e in errors}
    if "TABLE_MERGE"         in codes:
        return "Split the merged markdown table into separate tables matching the PDF structure."
    if "MISSING_COLUMNS"     in codes:
        return "Re-extract the table ensuring all customer-tier columns are preserved."
    if "CELL_BOUNDARY_BLEED" in codes or "CELL_ROW_SPLIT" in codes:
        return "Re-check PDF cell segmentation; reconstruct the row from the original PDF bounding box."
    if "MERGED_CELL_COLLAPSE"in codes:
        return "Verify the source PDF column count; split concatenated values into separate cells."
    if "CORRUPTED_SYMBOL"    in codes:
        return "Replace noise characters with the correct Unicode symbol (✓ U+2713) using the PDF source."
    if "WRONG_VALUE"         in codes:
        return "Cross-check every cell value against the original PDF page; pay attention to currency amounts."
    if "MISSING_ROW"         in codes or "SECTION_TRUNCATED" in codes:
        return "Re-extract the table from the PDF ensuring all rows including bottom rows are captured."
    if "FOOTNOTE_ERROR"      in codes:
        return "Re-extract footnotes preserving their numeric markers and original order."
    if "ROW_WIDTH_MISMATCH"  in codes:
        return "Align column counts by re-examining the PDF table structure and separator rows."
    if "PHANTOM_CONTENT"     in codes:
        return "Remove OCR artefact text; compare cell content against the PDF raster rendering."
    if not errors:
        return ""
    return "Review the flagged cells against the original PDF table."


def main(
    examples_path: Path,
    pdf_path: Path,
    markdown_path: Path,
    output_report_path: Path,
    output_markdown_path: Path,
) -> None:
    print("[1/6] Loading labeled examples …")
    correct_ex, partial_correct_ex, partial_incorrect_ex, incorrect_ex = load_examples(examples_path)
    print(
        f"      {len(correct_ex)} correct | "
        f"{len(partial_correct_ex)} partial-correct | "
        f"{len(partial_incorrect_ex)} partial-incorrect | "
        f"{len(incorrect_ex)} incorrect"
    )

    print("[2/6] Deriving rules from examples …")
    rules = derive_rules_from_examples(
        correct_ex, partial_correct_ex, partial_incorrect_ex, incorrect_ex
    )
    print(f"      {len(rules)} rules derived")

    print("[3/6] Extracting Markdown tables …")
    md_text = markdown_path.read_text(encoding="utf-8")
    tables_meta = extract_markdown_tables(md_text)
    print(f"      {len(tables_meta)} tables found")

    print("[4/6] Extracting PDF table references …")
    pdf_refs = extract_pdf_refs(pdf_path)
    print(
        f"      {len(pdf_refs)} PDF table locations extracted"
        + (" (pdfplumber not installed — skipping)" if not _HAS_PDFPLUMBER else "")
    )

    print("[5/6] Applying rules and matching to PDF …")
    table_reports: list[TableReport] = []
    for idx, meta in enumerate(tables_meta):
        t_id   = f"table_{idx+1:03d}"
        parsed = parse_table(meta["raw_text"])
        triggered, errors, confidence, verdict, est_ratio = apply_rules_to_table(
            meta, parsed, rules
        )
        page, section, pdf_idx, match_conf = match_table_to_pdf(parsed, pdf_refs)

        partial_detail = None
        if verdict in (VERDICT_PARTIAL_CORRECT, VERDICT_PARTIAL_INCORRECT):
            partial_detail = PartialDetail(
                correct_rows=[],
                incorrect_rows=[],
                correct_ratio=est_ratio,
                note="Estimated by heuristic rule engine (no ground-truth row labels available here)",
            )

        excerpt = (parsed["header"] and "| " + " | ".join(parsed["header"][:4]) + " |") or meta["raw_text"][:80]
        notes: list[str] = []
        if not pdf_refs:
            notes.append("PDF matching skipped: pdfplumber not available. Install with: pip install pdfplumber")
        if match_conf < 0.35 and pdf_refs:
            notes.append(f"Low PDF match confidence ({match_conf:.2f}) — manual verification recommended.")

        report = TableReport(
            table_id=t_id,
            markdown_source_range=SourceRange(meta["start_line"], meta["end_line"]),
            page=page,
            section=section,
            pdf_table_index=pdf_idx,
            pdf_match_confidence=match_conf,
            verdict=verdict,
            confidence=confidence,
            estimated_correct_ratio=est_ratio,
            partial_detail=partial_detail,
            triggered_rules=triggered,
            errors=errors,
            table_excerpt=str(excerpt)[:120],
            suggested_fix=_suggested_fix(errors),
            notes=notes,
        )
        table_reports.append(report)

        verdict_icon = {"CORRECT": "✓", "PARTIAL_CORRECT": "◑",
                        "PARTIAL_INCORRECT": "◐", "INCORRECT": "✗"}.get(verdict, "?")
        print(
            f"      [{verdict_icon}] {t_id}: {verdict} "
            f"(conf={confidence:.2f}, ratio={est_ratio:.2f}, errors={len(errors)}, page={page})"
        )

    print("[6/6] Writing outputs …")
    report_dict = build_verification_report(pdf_path.name, rules, table_reports)
    output_report_path.write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      Report   → {output_report_path}")

    annotated = annotate_markdown(md_text, tables_meta, table_reports)
    output_markdown_path.write_text(annotated, encoding="utf-8")
    print(f"      Markdown → {output_markdown_path}")

    s = report_dict["summary"]
    print(
        f"\n  Summary: {s['total_tables']} tables | "
        f"{s['correct_tables']} correct | "
        f"{s['partial_correct_tables']} partial-correct | "
        f"{s['partial_incorrect_tables']} partial-incorrect | "
        f"{s['incorrect_tables']} incorrect | "
        f"{s['uncertain_pdf_matches']} uncertain PDF matches"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PDF-to-Markdown Table Verifier v2 (partial labels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--examples",        type=Path, default=Path("table_dataset.json"))
    p.add_argument("--pdf",             type=Path, default=Path("source.pdf"))
    p.add_argument("--markdown",        type=Path, default=Path("converted.md"))
    p.add_argument("--output-report",   type=Path, default=Path("verification_report.json"))
    p.add_argument("--output-markdown", type=Path, default=Path("annotated_converted.md"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    for attr, flag in [("examples", "--examples"), ("pdf", "--pdf"), ("markdown", "--markdown")]:
        p = getattr(args, attr)
        if not p.exists():
            print(f"[error] {flag}: file not found: {p}", file=sys.stderr)
            sys.exit(1)
    main(
        examples_path=args.examples,
        pdf_path=args.pdf,
        markdown_path=args.markdown,
        output_report_path=args.output_report,
        output_markdown_path=args.output_markdown,
    )
