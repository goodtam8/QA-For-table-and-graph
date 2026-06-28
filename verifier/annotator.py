"""
annotator.py — Injects VERIFY and PDF_REF comment tags into markdown documents.

This is the final step of the verification pipeline:
    1. Parse the markdown tables using MarkdownTableParser
    2. Run rules engine + confidence scorer against each table
    3. Inject comment tags BEFORE each table:
         <!-- VERIFY: CORRECT | confidence=0.94 -->
         <!-- VERIFY: INCORRECT | reason=cell_boundary_bleed | confidence=0.61 -->
         <!-- PDF_REF: page=5, section="Section C: Fees and Charges" -->
    4. Return the annotated markdown string

Usage:
    annotator = MarkdownAnnotator(
        rules_path="verification_rules.json",
        pdf_tables_by_page=pdf_tables_data,
        page_images_dir="hsbc_output/verification/page_images",
    )
    annotated = annotator.annotate(markdown_text, pdf_page_of_table)
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field

from .table_parser import MarkdownTableParser, ParsedTable
from .rules_engine import RulesEngine
from .confidence_scorer import ConfidenceScorer, Verdict


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class TableVerification:
    table_index: int
    verdict: Verdict
    md_page: int | None
    section: str | None
    raw_md_table: str  # original markdown table text
    annotated_md_table: str  # table with tags injected

    def verify_tag(self) -> str:
        return self.verdict.to_verify_tag()

    def pdf_ref_tag(self) -> str:
        parts = []
        if self.md_page:
            parts.append(f'page={self.md_page}')
        if self.section:
            escaped_section = self.section.replace('"', '\\"')
            parts.append(f'section="{escaped_section}"')
        return f"<!-- PDF_REF: {', '.join(parts)} -->"

    def to_dict(self) -> dict:
        return {
            "table_index": self.table_index,
            "md_page": self.md_page,
            "section": self.section,
            "verdict": self.verdict.to_dict(),
        }


# ----------------------------------------------------------------------
# PDF data helpers
# ----------------------------------------------------------------------

def load_pdf_tables_by_page(pdf_tables_path: str) -> dict:
    """Load pdf_tables_by_page.json and return the pages dict."""
    if not os.path.exists(pdf_tables_path):
        return {"pages": []}
    with open(pdf_tables_path, encoding="utf-8") as f:
        return json.load(f)


def get_pdf_table_for_md_table(
    md_table: ParsedTable,
    pdf_tables_by_page: dict,
    pdf_page_hint: int | None = None,
) -> tuple[list[list[str]] | None, dict[str, str] | None]:
    """
    Given a parsed MD table with a page hint, find the corresponding PDF table
    from pdf_tables_by_page.json.

    Returns (pdf_table_cells, pdf_footnotes) or (None, None) if not found.
    """
    pages = pdf_tables_by_page.get("pages", [])
    target_page = pdf_page_hint - 1 if pdf_page_hint else None

    if target_page is not None:
        if 0 <= target_page < len(pages):
            page_data = pages[target_page]
            tables = page_data if isinstance(page_data, list) else page_data.get("tables", [])
            if tables:
                tbl = tables[0]  # first table on the page
                cells = tbl.get("cells", [])
                # Try to extract footnotes from table header/section
                footnotes = _extract_footnotes_from_pdf_table(cells)
                return cells, footnotes

    # Fallback: search all pages for matching column count
    best_match = None
    for page_idx, page_data in enumerate(pages):
        tables = page_data if isinstance(page_data, list) else page_data.get("tables", [])
        for tbl in tables:
            cells = tbl.get("cells", [])
            if len(cells) > 0 and len(cells[0]) == md_table.ncols:
                best_match = (cells, _extract_footnotes_from_pdf_table(cells))
                break
        if best_match:
            break

    return best_match if best_match else (None, None)


def _extract_footnotes_from_pdf_table(cells: list[list[str]]) -> dict[str, str]:
    """Extract footnote references from PDF table cells."""
    footnotes: dict[str, str] = {}
    footnote_pattern = re.compile(
        r"(?:¹|²|³|⁴|⁵|⁶|⁷|⁸|⁹|⁰)(?!\d)|"
        r"\b([1-9])\b(?=\s*[A-Z])"
    )
    mapping = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

    for row in cells:
        for cell in row:
            for m in re.finditer(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", cell):
                s = m.group(0).translate(mapping)
                footnotes[s] = cell[m.start():m.end() + 20].strip()

    return footnotes


# ----------------------------------------------------------------------
# Main annotator
# ----------------------------------------------------------------------

class MarkdownAnnotator:
    """
    Annotates markdown documents with VERIFY and PDF_REF comment tags.

    Parameters
    ----------
    rules_engine : RulesEngine
        The rules engine instance.
    confidence_scorer : ConfidenceScorer
        The confidence scorer instance.
    pdf_tables_by_page : dict | None
        Loaded pdf_tables_by_page.json data.
    page_images_dir : str | Path | None
        Directory containing page_XXX.png images (for PDF_REF page mapping).
    llm_client : openai client | None
        Optional LLM client for ambiguous table verification.
    """

    TABLE_BLOCK_RE = re.compile(
        r"""
        (?P<preamble>
            (?:
                <!--\s*[Pp]age\s*(\d+)\s*-->|
                ^#{1,6}\s+.+$|
                ^\s*<!--.*?-->\s*$
            )+\n*
        )?
        (?P<table>
            \|[^\n]*\|[\r\n]+
            (?:\|[\s\-:|]+\|[\r\n]+)?
            (?:\|[^\n]*\|[\r\n]*)*
        )
        """,
        re.VERBOSE | re.MULTILINE,
    )

    def __init__(
        self,
        rules_engine: RulesEngine,
        confidence_scorer: ConfidenceScorer,
        pdf_tables_by_page: dict | None = None,
        page_images_dir: str | Path | None = None,
        llm_client=None,
    ):
        self.rules_engine = rules_engine
        self.confidence_scorer = confidence_scorer
        self.pdf_tables_by_page = pdf_tables_by_page or {"pages": []}
        self.page_images_dir = Path(page_images_dir) if page_images_dir else None
        self.llm_client = llm_client
        self._verifications: list[TableVerification] = []

    @property
    def verifications(self) -> list[TableVerification]:
        return self._verifications

    def annotate(self, markdown_text: str, pdf_page_map: dict[int, int] | None = None) -> str:
        """
        Annotate all tables in the markdown text with VERIFY and PDF_REF tags.

        Parameters
        ----------
        markdown_text : str
            The raw markdown string (from Marker output).
        pdf_page_map : dict[int, int] | None
            Optional mapping: table_index → PDF page number.
            If not provided, page is inferred from page markers in the markdown.

        Returns
        -------
        str
            Annotated markdown string with comment tags injected before each table.
        """
        self._verifications = []
        text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")

        def replace_table_block(m: re.Match) -> str:
            preamble = m.group("preamble") or ""
            raw_table = m.group("table")

            # Extract page number from preamble
            page_marker = re.search(r"<!--\s*[Pp]age\s*(\d+)\s*-->", preamble)
            page_hint = int(page_marker.group(1)) if page_marker else None

            # Extract section from preamble
            section_lines = re.findall(r"^#{1,6}\s+(.+)$", preamble, re.MULTILINE)
            section = section_lines[-1].strip() if section_lines else None

            # Parse the table
            parser = MarkdownTableParser(raw_table)
            tables = parser.tables

            if not tables:
                return raw_table

            # Take the first (and usually only) table in this block
            table = tables[0]

            # Get PDF page hint from the map if provided
            actual_page = pdf_page_map.get(table.table_index, page_hint) if pdf_page_map else page_hint

            # Get PDF table data
            pdf_cells, pdf_footnotes = get_pdf_table_for_md_table(
                table, self.pdf_tables_by_page, actual_page
            )

            # Run rules engine
            result = self.rules_engine.check_table(
                table, pdf_cells, pdf_footnotes, self.llm_client
            )

            # Compute verdict
            if self.llm_client and 0.60 <= self.confidence_scorer.compute_verdict(result).confidence < 0.85:
                verdict = self.confidence_scorer.compute_with_llm_refinement(
                    result, table, pdf_cells, self.llm_client
                )
            else:
                verdict = self.confidence_scorer.compute_verdict(result)

            # Build verification object
            verification = TableVerification(
                table_index=len(self._verifications),
                verdict=verdict,
                md_page=actual_page,
                section=section,
                raw_md_table=raw_table,
                annotated_md_table=raw_table,
            )
            self._verifications.append(verification)

            # Build annotated output
            verify_tag = verification.verify_tag()
            pdf_ref_tag = verification.pdf_ref_tag()

            return f"{preamble}{verify_tag}\n{pdf_ref_tag}\n{raw_table}"

        annotated = self.TABLE_BLOCK_RE.sub(replace_table_block, text)

        # Update annotated_md_table in verifications with the new text
        # (This is a bit redundant since we already replaced inline)
        for i, v in enumerate(self._verifications):
            err_str = " | ".join(f"reason={r}" for r in v.verdict.all_errors) if v.verdict.all_errors else ""
            conf_str = f"confidence={v.verdict.confidence:.2f}"
            detail_str = err_str or conf_str
            v.annotated_md_table = (
                f"<!-- VERIFY: {v.verdict.label} | {detail_str} -->\n"
                f"<!-- PDF_REF: page={v.md_page}, section=\"{v.section or ''}\" -->\n"
                f"{v.raw_md_table}"
            )

        return annotated

    def annotate_with_llm(
        self,
        markdown_text: str,
        pdf_page_map: dict[int, int] | None = None,
    ) -> str:
        """
        Annotate using LLM verification for all tables.
        Falls back to heuristic rules if LLM is not available.
        """
        from .confidence_scorer import verify_with_llm
        from .table_parser import MarkdownTableParser

        self._verifications = []
        text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")

        def replace_table_block(m: re.Match) -> str:
            preamble = m.group("preamble") or ""
            raw_table = m.group("table")

            page_marker = re.search(r"<!--\s*[Pp]age\s*(\d+)\s*-->", preamble)
            page_hint = int(page_marker.group(1)) if page_marker else None

            section_lines = re.findall(r"^#{1,6}\s+(.+)$", preamble, re.MULTILINE)
            section = section_lines[-1].strip() if section_lines else None

            parser = MarkdownTableParser(raw_table)
            tables = parser.tables
            if not tables:
                return raw_table

            table = tables[0]
            actual_page = pdf_page_map.get(table.table_index, page_hint) if pdf_page_map else page_hint

            pdf_cells, pdf_footnotes = get_pdf_table_for_md_table(
                table, self.pdf_tables_by_page, actual_page
            )

            # Rule-based check
            result = self.rules_engine.check_table(table, pdf_cells, pdf_footnotes)
            heuristic_verdict = self.confidence_scorer.compute_verdict(result)

            # LLM check
            llm_result = verify_with_llm(table, pdf_cells, self.llm_client)

            # Combine: use LLM verdict if it differs significantly
            if llm_result.get("verdict") in ("CORRECT", "INCORRECT"):
                llm_conf = llm_result.get("confidence", heuristic_verdict.confidence)
                # If LLM is more confident, blend
                combined_conf = (heuristic_verdict.confidence + llm_conf) / 2
                if llm_result.get("verdict") == "INCORRECT" and heuristic_verdict.label == "CORRECT":
                    combined_verdict = "INCORRECT"
                elif llm_result.get("verdict") == "CORRECT" and heuristic_verdict.label == "INCORRECT":
                    combined_verdict = "CORRECT" if llm_conf > 0.85 else "INCORRECT"
                else:
                    combined_verdict = heuristic_verdict.label
            else:
                combined_verdict = heuristic_verdict.label
                combined_conf = heuristic_verdict.confidence

            from .confidence_scorer import Verdict as V
            final_verdict = V(
                label=combined_verdict,
                confidence=combined_conf,
                matched_rules=heuristic_verdict.matched_rules,
                primary_error=heuristic_verdict.primary_error,
                all_errors=heuristic_verdict.all_errors,
            )

            v = TableVerification(
                table_index=len(self._verifications),
                verdict=final_verdict,
                md_page=actual_page,
                section=section,
                raw_md_table=raw_table,
                annotated_md_table=raw_table,
            )
            self._verifications.append(v)

            verify_tag = v.verify_tag()
            pdf_ref_tag = v.pdf_ref_tag()
            return f"{preamble}{verify_tag}\n{pdf_ref_tag}\n{raw_table}"

        annotated = self.TABLE_BLOCK_RE.sub(replace_table_block, text)

        for i, v in enumerate(self._verifications):
            err_str = " | ".join(f"reason={r}" for r in v.verdict.all_errors) if v.verdict.all_errors else ""
            conf_str = f"confidence={v.verdict.confidence:.2f}"
            detail_str = err_str or conf_str
            v.annotated_md_table = (
                f"<!-- VERIFY: {v.verdict.label} | {detail_str} -->\n"
                f"<!-- PDF_REF: page={v.md_page}, section=\"{v.section or ''}\" -->\n"
                f"{v.raw_md_table}"
            )

        return annotated

    def get_verification_summary(self) -> dict:
        """Return a summary dict of all table verifications."""
        return {
            "total_tables": len(self._verifications),
            "correct": sum(1 for v in self._verifications if v.verdict.label == "CORRECT"),
            "incorrect": sum(1 for v in self._verifications if v.verdict.label == "INCORRECT"),
            "review": sum(1 for v in self._verifications if v.verdict.label == "REVIEW"),
            "tables": [v.to_dict() for v in self._verifications],
        }
