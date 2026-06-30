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

    Uses a multi-stage matching strategy:
    1. Page + column count match
    2. Semantic header similarity match across all pages
    3. Cell content similarity match for final disambiguation
    """
    pages = pdf_tables_by_page.get("pages", [])
    target_page = pdf_page_hint - 1 if pdf_page_hint else None

    md_headers = md_table.header_texts()
    md_col_count = len(md_headers)

    # Stage 1: Try exact page match with column count verification
    if target_page is not None:
        if 0 <= target_page < len(pages):
            page_data = pages[target_page]
            tables = page_data if isinstance(page_data, list) else page_data.get("tables", [])
            if tables:
                for tbl in tables:
                    cells = tbl.get("cells", [])
                    if cells and len(cells[0]) == md_col_count:
                        # Verify header semantic similarity
                        pdf_headers = cells[0]
                        if _headers_semantically_match(pdf_headers, md_headers, md_col_count):
                            footnotes = _extract_footnotes_from_pdf_table(cells)
                            return cells, footnotes

    # Stage 2: Search all pages for table with matching column count + semantic similarity
    best_match = None
    best_score = 0.0

    for page_idx, page_data in enumerate(pages):
        tables = page_data if isinstance(page_data, list) else page_data.get("tables", [])
        for tbl in tables:
            cells = tbl.get("cells", [])
            if not cells:
                continue

            pdf_col_count = len(cells[0])

            # Check column count match first
            if pdf_col_count != md_col_count:
                continue

            pdf_headers = cells[0]
            similarity = _compute_header_similarity(pdf_headers, md_headers)

            if similarity >= 0.5:  # At least 50% semantic match
                # Stage 3: Verify with cell content similarity
                content_score = _compute_content_similarity(cells, md_table)
                combined_score = (similarity * 0.6) + (content_score * 0.4)

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (cells, _extract_footnotes_from_pdf_table(cells))

    if best_match and best_score >= 0.3:
        return best_match

    # Stage 4: Last resort - return None (don't match wrong table)
    # This prevents false positives from mismatched tables
    return None, None


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for semantic comparison."""
    import re
    t = text.lower().strip()
    t = re.sub(r"<[^>]+>", "", t)  # Remove HTML tags
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)  # Remove bold markers
    t = re.sub(r"\s+", " ", t)  # Normalize whitespace
    t = t.replace("\u00a0", " ")  # Non-breaking space
    t = t.replace("\n", " ").replace("\r", " ")
    return t


def _headers_semantically_match(
    pdf_headers: list[str],
    md_headers: list[str],
    min_cols: int,
) -> bool:
    """Check if PDF and MD headers are semantically similar."""
    if len(pdf_headers) != len(md_headers):
        return False

    matches = 0
    for pdf_h, md_h in zip(pdf_headers[:min_cols], md_headers[:min_cols]):
        pdf_norm = _normalize_for_comparison(pdf_h)
        md_norm = _normalize_for_comparison(md_h)

        # Check if normalized texts share significant terms
        pdf_terms = set(pdf_norm.split())
        md_terms = set(md_norm.split())

        # Remove very short terms and common words
        stop_words = {"the", "a", "an", "and", "or", "of", "for", "in", "on", "at"}
        pdf_terms -= stop_words
        md_terms -= stop_words

        if not pdf_terms or not md_terms:
            matches += 1  # Empty headers match
            continue

        overlap = len(pdf_terms & md_terms)
        union = len(pdf_terms | md_terms)
        jaccard = overlap / union if union > 0 else 0

        if jaccard >= 0.4 or pdf_norm in md_norm or md_norm in pdf_norm:
            matches += 1

    return matches >= min_cols * 0.7  # 70% of columns should match


def _compute_header_similarity(
    pdf_headers: list[str],
    md_headers: list[str],
) -> float:
    """Compute 0.0-1.0 similarity score between header lists."""
    if len(pdf_headers) != len(md_headers):
        return 0.0

    total_similarity = 0.0
    for pdf_h, md_h in zip(pdf_headers, md_headers):
        pdf_norm = _normalize_for_comparison(pdf_h)
        md_norm = _normalize_for_comparison(md_h)

        if pdf_norm == md_norm:
            total_similarity += 1.0
        elif pdf_norm in md_norm or md_norm in pdf_norm:
            total_similarity += 0.8
        else:
            pdf_terms = set(pdf_norm.split())
            md_terms = set(md_norm.split())
            stop_words = {"the", "a", "an", "and", "or", "of", "for", "in", "on", "at"}
            pdf_terms -= stop_words
            md_terms -= stop_words

            if pdf_terms and md_terms:
                overlap = len(pdf_terms & md_terms)
                union = len(pdf_terms | md_terms)
                total_similarity += overlap / union if union > 0 else 0
            else:
                total_similarity += 0.5

    return total_similarity / len(pdf_headers) if pdf_headers else 0.0


def _compute_content_similarity(
    pdf_cells: list[list[str]],
    md_table: ParsedTable,
) -> float:
    """Compute content similarity between PDF table and MD table."""
    if not pdf_cells or len(pdf_cells) < 2:
        return 0.0

    pdf_data_rows = pdf_cells[1:]  # Skip header
    md_rows = md_table.rows

    if not pdf_data_rows or not md_rows:
        return 0.0

    # Sample rows for efficiency
    sample_size = min(5, len(pdf_data_rows), len(md_rows))
    matches = 0
    total_checks = sample_size * min(len(pdf_data_rows[0]), len(md_rows[0]) if md_rows else 0)

    for i in range(sample_size):
        pdf_row_idx = min(i, len(pdf_data_rows) - 1)
        md_row_idx = min(i, len(md_rows) - 1)

        pdf_row = pdf_data_rows[pdf_row_idx]
        md_row = md_rows[md_row_idx]

        for j in range(min(len(pdf_row), len(md_row))):
            pdf_norm = _normalize_for_comparison(pdf_row[j])
            md_norm = _normalize_for_comparison(md_row[j].text)

            # Check for exact match or substring match
            if pdf_norm == md_norm:
                matches += 2
            elif pdf_norm in md_norm or md_norm in pdf_norm:
                matches += 1.5
            elif set(pdf_norm.split()) & set(md_norm.split()):
                matches += 1

    return matches / max(total_checks, 1)


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

    # ── Page-source helpers ─────────────────────────────────────────────────────

    def _load_table_bboxes(self) -> list[dict]:
        """Load table_bboxes.json if it exists, return [] otherwise."""
        # Always use a path relative to the script location.
        bbox_path = Path(__file__).parent.parent / "hsbc_output" / "verification" / "table_bboxes.json"
        if bbox_path.exists():
            with open(bbox_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _find_page_marker_backward(self, text: str, table_start: int) -> int | None:
        """
        Search backward from table_start for the last <!-- Page N --> marker.
        Uses m.start('table') (actual table start) not m.start() (preamble start),
        so next-page-marker-in-preamble does NOT leak into the backward search.
        """
        before = text[:table_start]
        matches = list(re.finditer(r"<!--\s*[Pp]age\s*(\d+)\s*-->", before))
        return int(matches[-1].group(1)) if matches else None

    def _pdf_page_from_bboxes(self, bboxes: list[dict], table_index: int) -> int | None:
        """
        Return 1-based PDF page number for a table_index using table_bboxes.json.
        table_bboxes entries are in markdown table order. page_index is 0-based in
        the JSON, so we add 1 to get the PDF page number.
        """
        if 0 <= table_index < len(bboxes):
            pdf_idx = bboxes[table_index].get("page_index")
            if pdf_idx is not None:
                return int(pdf_idx) + 1
        return None

    # ── Main annotator ─────────────────────────────────────────────────────────

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

        # Load table_bboxes.json for authoritative PDF page mapping.
        # Entries are in markdown table order; page_index is 0-based.
        table_bboxes = self._load_table_bboxes()

        # Block counter for table_bboxes alignment (table.table_index from
        # MarkdownTableParser is always 0 when parsing isolated table text,
        # so we need to count blocks in document order ourselves).
        block_idx = [0]

        def replace_table_block(m: re.Match) -> str:
            preamble = m.group("preamble") or ""
            raw_table = m.group("table")
            # m.start("table") points to the actual table start (after preamble),
            # whereas m.start() points to the preamble start — which may include
            # the next page marker that leaked in via greedy preamble matching.
            # Use m.start("table") for backward page search to avoid that bleed.
            table_actual_start = m.start("table")

            # Forward-looking: page marker inside this table's own preamble
            page_marker = re.search(r"<!--\s*[Pp]age\s*(\d+)\s*-->", preamble)
            page_hint = int(page_marker.group(1)) if page_marker else None

            # Backward search from the actual table position (not preamble start)
            if page_hint is None:
                page_hint = self._find_page_marker_backward(text, table_actual_start)

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
            actual_page = pdf_page_map.get(block_idx[0], page_hint) if pdf_page_map else page_hint

            # Fall back to table_bboxes.json when no page marker was found
            if actual_page is None:
                actual_page = self._pdf_page_from_bboxes(table_bboxes, block_idx[0])

            block_idx[0] += 1

            # Get PDF table data
            pdf_cells, pdf_footnotes = get_pdf_table_for_md_table(
                table, self.pdf_tables_by_page, actual_page
            )

            # If no PDF match found but we have substantial table content, use LLM for verification
            if pdf_cells is None and self.llm_client:
                from .confidence_scorer import verify_with_llm
                llm_result = verify_with_llm(table, None, self.llm_client)
                llm_verdict_label = llm_result.get("verdict", "UNKNOWN")
                llm_confidence = llm_result.get("confidence", 0.5)

                if llm_verdict_label == "CORRECT":
                    verdict = Verdict(
                        label="CORRECT",
                        confidence=min(0.85, llm_confidence + 0.1),
                        matched_rules=[],
                        primary_error=None,
                        all_errors=[],
                    )
                elif llm_verdict_label == "INCORRECT":
                    verdict = Verdict(
                        label="REVIEW",
                        confidence=llm_confidence * 0.7,
                        matched_rules=[],
                        primary_error="llm_assessment",
                        all_errors=["llm_assessment"],
                    )
                else:
                    # Unknown verdict - default to REVIEW
                    verdict = Verdict(
                        label="REVIEW",
                        confidence=0.50,
                        matched_rules=[],
                        primary_error="uncertain",
                        all_errors=["llm_uncertain"],
                    )
            else:
                # Run rules engine
                result = self.rules_engine.check_table(
                    table, pdf_cells, pdf_footnotes, self.llm_client
                )

                # Compute verdict
                if self.llm_client and 0.50 <= self.confidence_scorer.compute_verdict(result).confidence < 0.85:
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
        table_bboxes = self._load_table_bboxes()
        block_idx = [0]

        def replace_table_block(m: re.Match) -> str:
            preamble = m.group("preamble") or ""
            raw_table = m.group("table")
            table_actual_start = m.start("table")

            page_marker = re.search(r"<!--\s*[Pp]age\s*(\d+)\s*-->", preamble)
            page_hint = int(page_marker.group(1)) if page_marker else None

            if page_hint is None:
                page_hint = self._find_page_marker_backward(text, table_actual_start)

            section_lines = re.findall(r"^#{1,6}\s+(.+)$", preamble, re.MULTILINE)
            section = section_lines[-1].strip() if section_lines else None

            parser = MarkdownTableParser(raw_table)
            tables = parser.tables
            if not tables:
                return raw_table

            table = tables[0]
            actual_page = pdf_page_map.get(block_idx[0], page_hint) if pdf_page_map else page_hint

            # Fall back to table_bboxes.json when no page marker was found
            if actual_page is None:
                actual_page = self._pdf_page_from_bboxes(table_bboxes, block_idx[0])

            block_idx[0] += 1

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
