#!/usr/bin/env python3
"""
PDF-to-Markdown Table Verifier
A production-ready, generalisable QA tool for detecting conversion errors in Markdown tables.

Updated version:
- Removed R01, R04, R05, R06, R08, R10, R12 from rule derivation and rule application.
- Added INCORRECT confidence threshold: only mark a table INCORRECT when confidence >= 0.55.
- Added REVIEW verdict for triggered-but-below-threshold tables.
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

# ── Optional dependencies (graceful degrades) ─────────────────────────────────
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


INCORRECT_CONFIDENCE_THRESHOLD = 0.55

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
class TableReport:
    table_id: str
    markdown_source_range: SourceRange
    page: Optional[int]
    section: Optional[str]
    pdf_table_index: Optional[int]
    pdf_match_confidence: float
    verdict: str   # "CORRECT" | "REVIEW" | "INCORRECT"
    confidence: float
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
    summary: dict[str, int]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

def load_examples(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (correct_examples, incorrect_examples)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    correct = [ex for ex in raw if ex.get("label", "").upper() == "CORRECT"]
    incorrect = [ex for ex in raw if ex.get("label", "").upper() == "INCORRECT"]
    return correct, incorrect

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARKDOWN TABLE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_SEP_RE = re.compile(r"^\s*\|[\s\-:]+(\|[\s\-:]+)+\|\s*$")

def extract_markdown_tables(text: str) -> list[dict]:
    """
    Detect GitHub-style Markdown tables preserving original line numbers.
    Returns list of dicts with keys: start_line, end_line, raw_text, rows.
    """
    lines = text.split("\n")
    tables = []
    i = 0
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
    """Split a Markdown row on | boundaries, strip whitespace."""
    parts = row.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.strip() for p in parts]


def parse_table(raw_text: str) -> dict:
    """
    Returns:
        header: list[str]
        separator: list[str]
        body_rows: list[list[str]]
        col_counts: list[int]   (per row, including header)
        all_cells: list[str]    (flat)
    """
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
# 4. RULE DERIVATION FROM LABELED EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

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
    "INCONSISTENT_COLUMN_COUNT": "Different rows have inconsistent column counts throughout the table.",
    "CORRUPTED_SYMBOL":         "A tick, checkmark, or special symbol is replaced with noise characters.",
    "PHANTOM_CONTENT":          "A cell contains OCR noise or text absent from the source PDF.",
    "FOOTNOTE_ERROR":           "Footnote markers or text are missing, reordered, or truncated.",
    "CELL_ROW_SPLIT":           "A single logical row is erroneously split into two or more rows.",
    "TRAILING_ARTIFACT":        "A cell contains trailing punctuation or symbol bleed not in the source.",
    "EXTRA_COLUMNS":            "The Markdown table has more columns than the source PDF table.",
    "CAPTION_IN_TABLE":         "A section caption or heading is absorbed as a table row.",
}


def derive_rules_from_examples(correct: list[dict], incorrect: list[dict]) -> list[Rule]:
    """
    Derive generalisable rules by observing patterns in labeled examples.
    Confidence weights are computed from precision/frequency signals.
    """
    rules: list[Rule] = []

    error_counts: Counter = Counter()
    code_to_examples: dict[str, list[str]] = defaultdict(list)
    for ex in incorrect:
        reason = ex.get("error_reason", "") or ""
        for code in TAXONOMY:
            if code in reason:
                error_counts[code] += 1
                code_to_examples[code].append(ex["table_id"])

    total_inc = max(len(incorrect), 1)
    total_cor = max(len(correct), 1)

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

    rules.append(Rule(
        rule_id="R02",
        rule_name="Row width mismatch",
        description=(
            "Counts column separators per row and flags rows whose width differs "
            "from the modal column count of the table."
        ),
        why_it_indicates_an_error=(
            "Merged or split cells in the PDF extraction stage produce rows with "
            "fewer or more columns than the header, indicating column boundary errors."
        ),
        confidence_weight=_weight("ROW_WIDTH_MISMATCH", 0.88),
        examples_from_dataset=code_to_examples.get("ROW_WIDTH_MISMATCH", []),
    ))

    rules.append(Rule(
        rule_id="R03",
        rule_name="Merged cell collapse",
        description=(
            "Detects cells containing multiple distinct values concatenated without "
            "a separator (e.g., 'Waived Waived' or 'HK$150 HK$125 HK$100')."
        ),
        why_it_indicates_an_error=(
            "When the PDF extractor fails to maintain column boundaries, values from "
            "multiple tier columns collapse into a single cell."
        ),
        confidence_weight=_weight("MERGED_CELL_COLLAPSE", 0.90),
        examples_from_dataset=code_to_examples.get("MERGED_CELL_COLLAPSE", []),
    ))

    rules.append(Rule(
        rule_id="R07",
        rule_name="Phantom / OCR noise content",
        description=(
            "Flags cells containing garbled or implausible text that bears no "
            "resemblance to surrounding content (e.g., 'lakada ada fara', ')A/ : 1')."
        ),
        why_it_indicates_an_error=(
            "OCR artefacts from non-text PDF regions (logos, rules, watermarks) can "
            "inject random character sequences into extracted table cells."
        ),
        confidence_weight=_weight("PHANTOM_CONTENT", 0.85),
        examples_from_dataset=code_to_examples.get("PHANTOM_CONTENT", []),
    ))

    rules.append(Rule(
        rule_id="R09",
        rule_name="Missing numeric / currency value",
        description=(
            "In a column whose header or majority of cells contain numeric/currency "
            "patterns (HK$, RMB, %, digits), flags cells that are unexpectedly empty."
        ),
        why_it_indicates_an_error=(
            "Column-shift or extraction failures can leave numeric cells blank while "
            "their values land in the wrong column."
        ),
        confidence_weight=_weight("MISSING_NUMERIC", 0.76),
        examples_from_dataset=code_to_examples.get("MISSING_NUMERIC", []),
    ))

    rules.append(Rule(
        rule_id="R11",
        rule_name="Extra columns (column count inflation)",
        description=(
            "Flags tables whose column count is higher than expected, suggesting the "
            "extractor inserted phantom column boundaries."
        ),
        why_it_indicates_an_error=(
            "PDF extraction can mistake whitespace or alignment guides for column "
            "separators, generating extra empty columns."
        ),
        confidence_weight=_weight("EXTRA_COLUMNS", 0.83),
        examples_from_dataset=code_to_examples.get("EXTRA_COLUMNS", []),
    ))

    return rules

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RULE APPLICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_NUMERIC = re.compile(r"(HK\$|RMB|USD|US\$|[\$£€¥])\s*[\d,]+|^\d[\d,\.]+$|^\d+%")
_MULTI_VALUE = re.compile(
    r"(Waived\s+Waived|HK\$\d+\s+HK\$\d+|Nil\s+Nil|[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)"
)
_OCR_NOISE = re.compile(r"[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\s+[a-zA-Z]{3,}\s+[a-zA-Z]{3,}")


def apply_rules_to_table(
    table_data: dict,
    parsed: dict,
    rules: list[Rule],
) -> tuple[list[TriggeredRule], list[ErrorInstance], float, str]:
    """
    Apply active rules to a parsed table.
    Returns: triggered_rules, errors, confidence, verdict
    """
    triggered: list[TriggeredRule] = []
    errors: list[ErrorInstance] = []
    rule_map = {r.rule_id: r for r in rules}

    header = parsed["header"]
    body_rows = parsed["body_rows"]
    col_counts = parsed["col_counts"]

    header_count = len(header)
    modal_count = Counter(col_counts).most_common(1)[0][0] if col_counts else 0

    def _hit(rule_id: str, evidence: str, error_code: str, message: str, severity: str = "medium"):
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
            code=error_code,
            message=message,
            evidence=evidence[:200],
            severity=severity,
        ))

    # ── R02: Row width mismatch ───────────────────────────────────────────────
    for idx, row in enumerate(body_rows):
        if len(row) != modal_count and modal_count > 0:
            ev = f"Row {idx+1}: {len(row)} cols vs expected {modal_count}"
            _hit(
                "R02", ev,
                "ROW_WIDTH_MISMATCH",
                f"Body row has {len(row)} columns; table modal is {modal_count}.",
                "high"
            )

    # ── R03: Merged cell collapse ─────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if _MULTI_VALUE.search(cell):
                _hit(
                    "R03", cell[:120],
                    "MERGED_CELL_COLLAPSE",
                    "Multiple distinct values are concatenated inside a single cell.",
                    "high"
                )
                break

    # ── R07: Phantom / OCR noise ──────────────────────────────────────────────
    for row in body_rows:
        for cell in row:
            if _OCR_NOISE.search(cell) and not re.search(r"[\$\d]", cell):
                _hit(
                    "R07", cell[:120],
                    "PHANTOM_CONTENT",
                    "Cell contains OCR noise text absent from the source document.",
                    "high"
                )
                break

    # ── R09: Missing numeric in numeric column ────────────────────────────────
    if body_rows and header:
        col_numeric_ratio: list[float] = []
        for col_i in range(len(header)):
            vals = [row[col_i] for row in body_rows if col_i < len(row) and row[col_i].strip()]
            ratio = sum(1 for v in vals if _NUMERIC.search(v)) / len(vals) if vals else 0.0
            col_numeric_ratio.append(ratio)

        for row_idx, row in enumerate(body_rows):
            for col_i in range(len(header)):
                cell = row[col_i] if col_i < len(row) else ""
                if col_numeric_ratio[col_i] >= 0.5 and not cell.strip():
                    ev = f"Row {row_idx+1}, col {col_i+1} (header='{header[col_i]}'): empty"
                    _hit(
                        "R09", ev,
                        "MISSING_NUMERIC",
                        "Empty cell in a column that should contain a numeric/currency value.",
                        "medium"
                    )
                    break

    # ── R11: Extra columns ────────────────────────────────────────────────────
    if header_count > 8:
        empty_header_cols = sum(1 for h in header if not h.strip())
        if empty_header_cols > header_count // 3:
            ev = f"{empty_header_cols}/{header_count} header cells are empty"
            _hit(
                "R11", ev,
                "EXTRA_COLUMNS",
                "Table has an unusually large number of empty header columns.",
                "medium"
            )

    # ─── Aggregate confidence & thresholded verdict ───────────────────────────
    if triggered:
        product = 1.0
        for t in triggered:
            product *= (1.0 - t.confidence_weight)
        confidence = round(min(1.0 - product, 0.99), 3)
        verdict = "INCORRECT" if confidence >= INCORRECT_CONFIDENCE_THRESHOLD else "REVIEW"
    else:
        confidence = 0.05
        verdict = "CORRECT"

    return triggered, errors, confidence, verdict

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
    """Extract table locations from the PDF using pdfplumber."""
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
                    header_text = " ".join(cell or "" for cell in (table[0] if table else [])).strip()
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


def match_table_to_pdf(parsed: dict, pdf_refs: list[PDFTableRef]) -> tuple[Optional[int], Optional[str], Optional[int], float]:
    """
    Fuzzy-match the Markdown table header/content to the closest PDF table.
    Returns: page, section, pdf_table_index, match_confidence
    """
    if not pdf_refs:
        return None, None, None, 0.0

    md_header = " ".join(parsed.get("header", []))
    md_sample = md_header + " " + " ".join(c for row in (parsed.get("body_rows") or [])[:3] for c in row)

    best_score = 0.0
    best_ref: Optional[PDFTableRef] = None
    for ref in pdf_refs:
        score = _fuzzy(md_sample, ref.header_text + " " + ref.page_text)
        if score > best_score:
            best_score = score
            best_ref = ref

    if best_ref:
        return best_ref.page, best_ref.section, best_ref.pdf_table_index, round(best_score, 3)
    return None, None, None, 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# 7. MARKDOWN ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_markdown(original_md: str, tables_meta: list[dict], reports: list[TableReport]) -> str:
    """
    Wrap each detected table with VERIFIER and PDF_REF comment tags.
    tables_meta[i] contains: start_line, end_line, raw_text
    reports[i] is the TableReport for that table.
    """
    lines = original_md.split("\n")
    replacements: list[tuple[int, int, str]] = []

    for meta, report in zip(tables_meta, reports):
        start = meta["start_line"] - 1
        end = meta["end_line"]

        errors_str = ",".join(e.code for e in report.errors) if report.errors else ""
        verifier_open = (
            f"<!-- VERIFIER: status={report.verdict} | "
            f"confidence={report.confidence:.2f} | "
            f"errors={errors_str} -->"
        )
        verifier_close = "<!-- /VERIFIER -->"

        page_str = str(report.page) if report.page else "unknown"
        section_str = report.section or "unknown"
        pdf_idx_str = str(report.pdf_table_index) if report.pdf_table_index else "unknown"
        pdf_ref = (
            f'\n<!-- PDF_REF: page={page_str} | '
            f'section="{section_str}" | '
            f'pdf_table_index={pdf_idx_str} -->'
        )

        table_block = "\n".join(lines[start:end])
        annotated = f"{verifier_open}\n{table_block}\n{verifier_close}{pdf_ref}"
        replacements.append((start, end, annotated))

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]

    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_verification_report(pdf_name: str, rules: list[Rule], table_reports: list[TableReport]) -> dict:
    correct_count = sum(1 for t in table_reports if t.verdict == "CORRECT")
    incorrect_count = sum(1 for t in table_reports if t.verdict == "INCORRECT")
    review_count = sum(1 for t in table_reports if t.verdict == "REVIEW")
    uncertain = sum(1 for t in table_reports if t.pdf_match_confidence < 0.35 and t.page is not None)
    return {
        "document_name": pdf_name,
        "verifier_version": "1.1",
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
                "triggered_rules": [asdict(r) for r in t.triggered_rules],
                "errors": [asdict(e) for e in t.errors],
                "table_excerpt": t.table_excerpt,
                "suggested_fix": t.suggested_fix,
                "notes": t.notes,
            }
            for t in table_reports
        ],
        "summary": {
            "total_tables": len(table_reports),
            "correct_tables": correct_count,
            "review_tables": review_count,
            "incorrect_tables": incorrect_count,
            "uncertain_pdf_matches": uncertain,
        },
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _suggested_fix(errors: list[ErrorInstance]) -> str:
    codes = {e.code for e in errors}
    if "MERGED_CELL_COLLAPSE" in codes:
        return "Verify that the source PDF has the correct number of columns; split concatenated values."
    if "ROW_WIDTH_MISMATCH" in codes:
        return "Align column counts by re-examining the PDF table structure and separator rows."
    if "PHANTOM_CONTENT" in codes:
        return "Remove OCR artefact text; compare cell content against the PDF raster rendering."
    if "MISSING_NUMERIC" in codes:
        return "Check for column shift or dropped numeric values and restore them from the source PDF."
    if "EXTRA_COLUMNS" in codes:
        return "Remove phantom columns introduced by whitespace or separator mis-detection."
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
    correct_ex, incorrect_ex = load_examples(examples_path)
    print(f"      {len(correct_ex)} correct, {len(incorrect_ex)} incorrect examples")

    print("[2/6] Deriving rules from examples …")
    rules = derive_rules_from_examples(correct_ex, incorrect_ex)
    print(f"      {len(rules)} rules derived")

    print("[3/6] Extracting Markdown tables …")
    md_text = markdown_path.read_text(encoding="utf-8")
    tables_meta = extract_markdown_tables(md_text)
    print(f"      {len(tables_meta)} tables found")

    print("[4/6] Extracting PDF table references …")
    pdf_refs = extract_pdf_refs(pdf_path)
    print(f"      {len(pdf_refs)} PDF table locations extracted"
          + (" (pdfplumber not installed — skipping PDF matching)" if not _HAS_PDFPLUMBER else ""))

    print("[5/6] Applying rules and matching to PDF …")
    table_reports: list[TableReport] = []
    for idx, meta in enumerate(tables_meta):
        t_id = f"table_{idx+1:03d}"
        parsed = parse_table(meta["raw_text"])
        triggered, errors, confidence, verdict = apply_rules_to_table(meta, parsed, rules)
        page, section, pdf_idx, match_conf = match_table_to_pdf(parsed, pdf_refs)

        excerpt = (parsed["header"] and "| " + " | ".join(parsed["header"][:4]) + " |") or meta["raw_text"][:80]
        notes = []
        if not pdf_refs:
            notes.append("PDF matching skipped: pdfplumber not available. Install with: pip install pdfplumber")
        if match_conf < 0.35 and pdf_refs:
            notes.append(f"Low PDF match confidence ({match_conf:.2f}) — manual verification recommended.")
        if verdict == "REVIEW":
            notes.append(
                f"Triggered rules found, but combined confidence {confidence:.2f} is below the incorrect threshold {INCORRECT_CONFIDENCE_THRESHOLD:.2f}."
            )

        report = TableReport(
            table_id=t_id,
            markdown_source_range=SourceRange(meta["start_line"], meta["end_line"]),
            page=page,
            section=section,
            pdf_table_index=pdf_idx,
            pdf_match_confidence=match_conf,
            verdict=verdict,
            confidence=confidence,
            triggered_rules=triggered,
            errors=errors,
            table_excerpt=str(excerpt)[:120],
            suggested_fix=_suggested_fix(errors),
            notes=notes,
        )
        table_reports.append(report)
        status_icon = "✗" if verdict == "INCORRECT" else ("~" if verdict == "REVIEW" else "✓")
        print(f"      [{status_icon}] {t_id}: {verdict} (conf={confidence:.2f}, errors={len(errors)}, page={page})")

    print("[6/6] Writing outputs …")
    report_dict = build_verification_report(pdf_path.name, rules, table_reports)
    output_report_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      Report  → {output_report_path}")

    annotated = annotate_markdown(md_text, tables_meta, table_reports)
    output_markdown_path.write_text(annotated, encoding="utf-8")
    print(f"      Markdown → {output_markdown_path}")

    summary = report_dict["summary"]
    print(
        f"\n  Summary: {summary['total_tables']} tables | "
        f"{summary['correct_tables']} correct | "
        f"{summary['review_tables']} review | "
        f"{summary['incorrect_tables']} incorrect | "
        f"{summary['uncertain_pdf_matches']} uncertain PDF matches"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PDF-to-Markdown Table Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--examples",         type=Path, default=Path("examples.json"))
    p.add_argument("--pdf",              type=Path, default=Path("source.pdf"))
    p.add_argument("--markdown",         type=Path, default=Path("converted.md"))
    p.add_argument("--output-report",    type=Path, default=Path("verification_report.json"))
    p.add_argument("--output-markdown",  type=Path, default=Path("annotated_converted.md"))
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
        examples_path=args.examples,
        pdf_path=args.pdf,
        markdown_path=args.markdown,
        output_report_path=args.output_report,
        output_markdown_path=args.output_markdown,
    )
