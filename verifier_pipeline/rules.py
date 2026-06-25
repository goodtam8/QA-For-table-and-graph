"""
rules.py — Error taxonomy and rule derivation from labeled examples.

Public API
----------
TAXONOMY            dict[str, str]      error-code → short description
derive_rules_from_examples(correct, incorrect) -> list[Rule]
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from models import Rule
from md_extractor import parse_table


# ── error taxonomy ────────────────────────────────────────────────────────────

TAXONOMY: dict[str, str] = {
    "CELL_BOUNDARY_BLEED":      "A word or phrase is split unnaturally across adjacent cells.",
    "BROKEN_WORD_FRAGMENT":     "A cell contains a non-word token that appears to be part of a longer word.",
    "ROW_WIDTH_MISMATCH":       "A body row contains a different number of columns than the header.",
    "HEADER_DATA_MISALIGNMENT": "The header column count is inconsistent with separator or data rows.",
    "MISSING_NUMERIC":          "A cell expected to contain a numeric or currency value is empty.",
    "MERGED_CELL_COLLAPSE":     "Multiple distinct values are concatenated inside a single cell.",
    "TRUNCATED_TEXT":           "A cell or section appears to be cut off mid-sentence or mid-word.",
    "COLUMN_SHIFT":             "Data values are shifted into the wrong column.",
    "MALFORMED_PERCENT":        "A percentage or rate value is malformed or split.",
    "INCONSISTENT_COLUMN_COUNT":"Different rows have inconsistent column counts throughout the table.",
    "CORRUPTED_SYMBOL":         "A tick, checkmark, or special symbol is replaced with noise characters.",
    "PHANTOM_CONTENT":          "A cell contains OCR noise or text absent from the source PDF.",
    "FOOTNOTE_ERROR":           "Footnote markers or text are missing, reordered, or truncated.",
    "CELL_ROW_SPLIT":           "A single logical row is erroneously split into two or more rows.",
    "TRAILING_ARTIFACT":        "A cell contains trailing punctuation or symbol bleed not in the source.",
    "EXTRA_COLUMNS":            "The Markdown table has more columns than the source PDF table.",
    "CAPTION_IN_TABLE":         "A section caption or heading is absorbed as a table row.",
}


# ── weight helper (private) ───────────────────────────────────────────────────

def _compute_weight(
    code: str,
    base: float,
    error_counts: Counter,
    fp_counts: Counter,
    total_inc: int,
    total_cor: int,
) -> float:
    freq    = error_counts.get(code, 0) / total_inc
    fp_rate = fp_counts.get(code, 0) / total_cor
    w       = base * (0.6 + 0.4 * freq) * max(0.1, 1 - fp_rate)
    return round(min(max(w, 0.05), 0.99), 2)


# ── public API ────────────────────────────────────────────────────────────────

def derive_rules_from_examples(
    correct: list[dict],
    incorrect: list[dict],
) -> list[Rule]:
    """
    Derive generalizable rules by observing patterns in labeled examples.
    Confidence weights are computed from precision/frequency signals.
    """
    error_counts:    Counter               = Counter()
    code_to_examples: dict[str, list[str]] = defaultdict(list)

    for ex in incorrect:
        reason = ex.get("error_reason", "") or ""
        for code in TAXONOMY:
            if code in reason:
                error_counts[code] += 1
                code_to_examples[code].append(ex["table_id"])

    total_inc = max(len(incorrect), 1)
    total_cor = max(len(correct), 1)

    # False-positive counts from correct examples
    fp_counts: Counter = Counter()
    for ex in correct:
        parsed = parse_table(ex.get("md_table", ""))
        counts = parsed["col_counts"]
        if counts and len(set(counts)) > 1:
            fp_counts["ROW_WIDTH_MISMATCH"] += 1

    def W(code: str, base: float) -> float:
        return _compute_weight(code, base, error_counts, fp_counts, total_inc, total_cor)

    eg = code_to_examples.get

    return [
        Rule("R01", "Cell boundary bleed",
             "Detects tokens that appear to be halves of a single word split across adjacent cells. "
             "Signals: a cell ends mid-word and the next cell begins with a lower-case continuation.",
             "PDF column boundaries are sometimes inserted inside a continuous text run.",
             W("CELL_BOUNDARY_BLEED", 0.93), eg("CELL_BOUNDARY_BLEED", [])),

        Rule("R02", "Row width mismatch",
             "Counts column separators per row and flags rows whose width differs from the modal count.",
             "Merged or split cells produce rows with fewer or more columns than the header.",
             W("ROW_WIDTH_MISMATCH", 0.88), eg("ROW_WIDTH_MISMATCH", [])),

        Rule("R03", "Merged cell collapse",
             "Detects cells containing multiple distinct values concatenated without a separator.",
             "When the PDF extractor fails to maintain column boundaries, values collapse into one cell.",
             W("MERGED_CELL_COLLAPSE", 0.90), eg("MERGED_CELL_COLLAPSE", [])),

        Rule("R04", "Corrupted symbol",
             "Detects cells containing noise characters substituting for symbols (✓, ✗, etc.).",
             "OCR and PDF extractors frequently mis-read graphical tick marks.",
             W("CORRUPTED_SYMBOL", 0.91), eg("CORRUPTED_SYMBOL", [])),

        Rule("R05", "Trailing artifact",
             "Detects cells with trailing underscore, dash, or stray punctuation after a valid value.",
             "PDF extraction can bleed typographic rules or underlines into cell text.",
             W("TRAILING_ARTIFACT", 0.82), eg("TRAILING_ARTIFACT", [])),

        Rule("R06", "Row split (single logical row → multiple rows)",
             "Detects consecutive rows where the first cell looks like a continuation of the preceding row.",
             "Row-level PDF extraction errors insert spurious line breaks inside wrapped cells.",
             W("CELL_ROW_SPLIT", 0.87), eg("CELL_ROW_SPLIT", [])),

        Rule("R07", "Phantom / OCR noise content",
             "Flags cells containing garbled or implausible text bearing no resemblance to surrounding content.",
             "OCR artefacts from non-text PDF regions can inject random character sequences.",
             W("PHANTOM_CONTENT", 0.85), eg("PHANTOM_CONTENT", [])),

        Rule("R08", "Footnote error",
             "Detects footnote blocks that are reordered, missing their numeric marker, or truncated.",
             "PDF-to-Markdown converters often mishandle footnote numbering and ordering.",
             W("FOOTNOTE_ERROR", 0.78), eg("FOOTNOTE_ERROR", [])),

        Rule("R09", "Missing numeric / currency value",
             "In a column whose header or majority of cells contain numeric/currency patterns, "
             "flags cells that are unexpectedly empty.",
             "Column-shift or extraction failures can leave numeric cells blank.",
             W("MISSING_NUMERIC", 0.76), eg("MISSING_NUMERIC", [])),

        Rule("R10", "Caption absorbed into table",
             "Detects when the first row of a Markdown table appears to be a prose sentence or caption.",
             "PDF extractors with imprecise bounding-box detection absorb captions as table rows.",
             W("CAPTION_IN_TABLE", 0.80), eg("CAPTION_IN_TABLE", [])),

        Rule("R11", "Extra columns (column count inflation)",
             "Flags tables whose column count is higher than expected.",
             "PDF extraction can mistake whitespace for column separators.",
             W("EXTRA_COLUMNS", 0.83), eg("EXTRA_COLUMNS", [])),

        Rule("R12", "Truncated text",
             "Detects cells or footnote text that end abruptly mid-word or mid-sentence.",
             "Text extraction page-boundary or buffer issues can silently truncate content.",
             W("TRUNCATED_TEXT", 0.79), eg("TRUNCATED_TEXT", [])),
    ]
