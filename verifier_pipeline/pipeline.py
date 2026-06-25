"""
pipeline.py — Main verification pipeline.

Orchestrates: load examples → derive rules → extract MD tables →
              extract PDF refs → apply rules → match PDF → write outputs.

Public API
----------
suggested_fix(errors)  -> str
main(examples_path, pdf_path, markdown_path, output_report_path, output_markdown_path)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from models      import TableReport, SourceRange
from examples    import load_examples
from md_extractor import extract_markdown_tables, parse_table
from rules        import derive_rules_from_examples
from rule_engine  import apply_rules_to_table
from pdf_matcher  import extract_pdf_refs, match_table_to_pdf
from annotator    import annotate_markdown
from report_builder import build_verification_report


# ── fix suggestion ────────────────────────────────────────────────────────────

def suggested_fix(errors) -> str:
    """Return a human-readable fix hint based on the error codes present."""
    codes = {e.code for e in errors}
    if "CELL_BOUNDARY_BLEED" in codes or "CELL_ROW_SPLIT" in codes:
        return ("Re-check PDF cell segmentation; "
                "reconstruct the row from the original PDF bounding box.")
    if "MERGED_CELL_COLLAPSE" in codes:
        return ("Verify that the source PDF has the correct number of columns; "
                "split concatenated values.")
    if "CORRUPTED_SYMBOL" in codes:
        return ("Replace noise characters with the correct Unicode symbol "
                "(e.g., ✓ U+2713) using the PDF source.")
    if "ROW_WIDTH_MISMATCH" in codes:
        return ("Align column counts by re-examining the PDF table structure "
                "and separator rows.")
    if "FOOTNOTE_ERROR" in codes:
        return ("Re-extract footnotes preserving their numeric markers "
                "and original order.")
    if "PHANTOM_CONTENT" in codes:
        return ("Remove OCR artefact text; compare cell content against "
                "the PDF raster rendering.")
    if errors:
        return "Review the flagged cells against the original PDF table."
    return ""


# ── pipeline ──────────────────────────────────────────────────────────────────

def main(
    examples_path:        Path,
    pdf_path:             Path,
    markdown_path:        Path,
    output_report_path:   Path,
    output_markdown_path: Path,
) -> None:
    # 1 — labeled examples
    print("[1/6] Loading labeled examples …")
    correct_ex, incorrect_ex = load_examples(examples_path)
    print(f"      {len(correct_ex)} correct, {len(incorrect_ex)} incorrect examples")

    # 2 — rules
    print("[2/6] Deriving rules from examples …")
    rules = derive_rules_from_examples(correct_ex, incorrect_ex)
    print(f"      {len(rules)} rules derived")

    # 3 — Markdown tables
    print("[3/6] Extracting Markdown tables …")
    md_text     = markdown_path.read_text(encoding="utf-8")
    tables_meta = extract_markdown_tables(md_text)
    print(f"      {len(tables_meta)} tables found")

    # 4 — PDF refs
    print("[4/6] Extracting PDF table references …")
    pdf_refs = extract_pdf_refs(pdf_path)
    suffix   = "" if pdf_refs else " (pdfplumber not installed — skipping PDF matching)"
    print(f"      {len(pdf_refs)} PDF table locations extracted{suffix}")

    # 5 — apply rules
    print("[5/6] Applying rules and matching to PDF …")
    table_reports: list[TableReport] = []

    for idx, meta in enumerate(tables_meta):
        t_id   = f"table_{idx+1:03d}"
        parsed = parse_table(meta["raw_text"])

        triggered, errors, confidence, verdict = apply_rules_to_table(meta, parsed, rules)
        page, section, pdf_idx, match_conf     = match_table_to_pdf(parsed, pdf_refs)

        excerpt = (
            parsed["header"] and "| " + " | ".join(parsed["header"][:4]) + " |"
        ) or meta["raw_text"][:80]

        notes: list[str] = []
        if not pdf_refs:
            notes.append(
                "PDF matching skipped: pdfplumber not available. "
                "Install with: pip install pdfplumber"
            )
        if match_conf < 0.35 and pdf_refs:
            notes.append(
                f"Low PDF match confidence ({match_conf:.2f}) — "
                "manual verification recommended."
            )

        report = TableReport(
            table_id              = t_id,
            markdown_source_range = SourceRange(meta["start_line"], meta["end_line"]),
            page                  = page,
            section               = section,
            pdf_table_index       = pdf_idx,
            pdf_match_confidence  = match_conf,
            verdict               = verdict,
            confidence            = confidence,
            triggered_rules       = triggered,
            errors                = errors,
            table_excerpt         = str(excerpt)[:120],
            suggested_fix         = suggested_fix(errors),
            notes                 = notes,
        )
        table_reports.append(report)

        icon = "✗" if verdict == "INCORRECT" else "✓"
        print(f"      [{icon}] {t_id}: {verdict} "
              f"(conf={confidence:.2f}, errors={len(errors)}, page={page})")

    # 6 — write outputs
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
        f"{s['incorrect_tables']} incorrect | "
        f"{s['uncertain_pdf_matches']} uncertain PDF matches"
    )
