"""
report_builder.py — Assemble the final JSON-serialisable verification report dict.

Public API
----------
build_verification_report(pdf_name, rules, table_reports) -> dict
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from models import Rule, TableReport


def build_verification_report(
    pdf_name:      str,
    rules:         list[Rule],
    table_reports: list[TableReport],
) -> dict:
    """
    Combine rules and per-table reports into a single serialisable dict.

    Returns a dict with keys:
        document_name, verifier_version, generated_at,
        rules, tables, summary
    """
    correct_count   = sum(1 for t in table_reports if t.verdict == "CORRECT")
    incorrect_count = len(table_reports) - correct_count
    uncertain       = sum(
        1 for t in table_reports
        if t.pdf_match_confidence < 0.35 and t.page is not None
    )

    return {
        "document_name":    pdf_name,
        "verifier_version": "1.0",
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "rules":  [asdict(r) for r in rules],
        "tables": [
            {
                "table_id":             t.table_id,
                "markdown_source_range":asdict(t.markdown_source_range),
                "page":                 t.page,
                "section":              t.section,
                "pdf_table_index":      t.pdf_table_index,
                "pdf_match_confidence": t.pdf_match_confidence,
                "verdict":              t.verdict,
                "confidence":           t.confidence,
                "triggered_rules":      [asdict(r) for r in t.triggered_rules],
                "errors":               [asdict(e) for e in t.errors],
                "table_excerpt":        t.table_excerpt,
                "suggested_fix":        t.suggested_fix,
                "notes":                t.notes,
            }
            for t in table_reports
        ],
        "summary": {
            "total_tables":           len(table_reports),
            "correct_tables":         correct_count,
            "incorrect_tables":       incorrect_count,
            "uncertain_pdf_matches":  uncertain,
        },
    }
