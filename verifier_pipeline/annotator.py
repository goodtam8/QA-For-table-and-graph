"""
annotator.py — Wrap detected Markdown tables with VERIFIER and PDF_REF comment tags.

Public API
----------
annotate_markdown(original_md, tables_meta, reports) -> str
"""
from __future__ import annotations

from models import TableReport


def annotate_markdown(
    original_md:  str,
    tables_meta:  list[dict],
    reports:      list[TableReport],
) -> str:
    """
    Insert HTML comment annotations around each Markdown table.

    Each table block is wrapped with:
        <!-- VERIFIER: status=… | confidence=… | errors=… -->
        <original table rows>
        <!-- /VERIFIER -->
        <!-- PDF_REF: page=… | section="…" | pdf_table_index=… -->

    Parameters
    ----------
    original_md  : the full Markdown document text
    tables_meta  : list of dicts from extract_markdown_tables
    reports      : one TableReport per entry in tables_meta (same order)

    Returns
    -------
    Annotated Markdown string.
    """
    lines = original_md.split("\n")
    replacements: list[tuple[int, int, str]] = []

    for meta, report in zip(tables_meta, reports):
        start = meta["start_line"] - 1   # convert to 0-based
        end   = meta["end_line"]          # exclusive

        errors_str    = ",".join(e.code for e in report.errors) if report.errors else ""
        verifier_open = (
            f"<!-- VERIFIER: status={report.verdict} | "
            f"confidence={report.confidence:.2f} | "
            f"errors={errors_str} -->"
        )
        verifier_close = "<!-- /VERIFIER -->"

        page_str    = str(report.page)             if report.page             else "unknown"
        section_str = report.section               if report.section          else "unknown"
        pdf_idx_str = str(report.pdf_table_index)  if report.pdf_table_index  else "unknown"
        pdf_ref     = (
            f'\n<!-- PDF_REF: page={page_str} | '
            f'section="{section_str}" | '
            f'pdf_table_index={pdf_idx_str} -->'
        )

        table_block = "\n".join(lines[start:end])
        annotated   = f"{verifier_open}\n{table_block}\n{verifier_close}{pdf_ref}"
        replacements.append((start, end, annotated))

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]

    return "\n".join(lines)
