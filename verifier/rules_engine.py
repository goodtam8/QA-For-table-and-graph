"""
rules_engine.py — Loads verification rules and runs pattern checks against parsed tables.

Each rule is data-driven (loaded from verification_rules.json) and returns:
    RuleMatch(nm_rule_id, matched, severity, confidence_delta, reason, details)
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass, field
from typing import Optional

from .table_parser import ParsedTable, Cell


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class RuleMatch:
    rule_id: str
    rule_name: str
    matched: bool
    severity: str                      # "critical" | "major" | "minor"
    severity_weight: float
    confidence_delta: float            # how much this match reduces confidence
    reason: str                       # human-readable description
    details: list[str] = field(default_factory=list)  # specific error instances

    @property
    def is_error(self) -> bool:
        return self.matched

    def to_tag_fragment(self) -> str:
        """Format the rule match as a tag fragment for the annotation."""
        if self.matched:
            return f"reason={self.rule_id}"
        return ""


@dataclass
class RuleCheckResult:
    table_index: int
    matches: list[RuleMatch] = field(default_factory=list)

    @property
    def matched_rules(self) -> list[RuleMatch]:
        return [m for m in self.matches if m.matched]

    @property
    def error_reasons(self) -> list[str]:
        return [m.rule_id for m in self.matched_rules]

    @property
    def has_errors(self) -> bool:
        return len(self.matched_rules) > 0


# ----------------------------------------------------------------------
# Rule implementations
# ----------------------------------------------------------------------

class _FootnoteMismatchRule:
    """Rule: footnote_mismatch — footnote superscripts must match PDF anchors exactly."""

    rule_id = "footnote_mismatch"
    rule_name = "Footnote Number and Anchor Consistency"
    severity = "major"
    severity_weight = 0.25

    @staticmethod
    def check(
        table: ParsedTable,
        pdf_table_cells: Optional[list[list[str]]] = None,
        pdf_footnotes: Optional[dict[str, str]] = None,
    ) -> RuleMatch:
        if pdf_table_cells is None or not pdf_table_cells:
            return RuleMatch(
                rule_id="footnote_mismatch",
                rule_name="Footnote Number and Anchor Consistency",
                matched=False,
                severity="major",
                severity_weight=0.25,
                confidence_delta=0.0,
                reason="No PDF table data available; footnote check skipped.",
            )

        details: list[str] = []

        # Extract superscripts from PDF cells (expected set)
        pdf_superscripts: set[str] = set()
        for row in pdf_table_cells:
            for cell in row:
                found = re.findall(r"(?:¹|²|³|⁴|⁵|⁶|⁷|⁸|⁹|⁰)", cell)
                for f in found:
                    mapping = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
                    pdf_superscripts.add(f.translate(mapping))

        # Check MD superscripts against PDF
        md_superscripts: set[str] = set()
        for cell in table.all_cells():
            for s in cell.footnote_superscripts:
                md_superscripts.add(s.strip())

        # If PDF has footnotes but MD has none → mismatch
        if pdf_superscripts and not md_superscripts:
            details.append(f"PDF has footnotes {sorted(pdf_superscripts)} but MD has none")
            return RuleMatch(
                rule_id="footnote_mismatch",
                rule_name="Footnote Number and Anchor Consistency",
                matched=True,
                severity="major",
                severity_weight=0.25,
                confidence_delta=0.25,
                reason="Footnote superscripts missing from markdown table",
                details=details,
            )

        # Check for superscript renumbering (PDF has superscripts that don't match MD pattern)
        missing = pdf_superscripts - md_superscripts
        extra = md_superscripts - pdf_superscripts

        if missing or extra:
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing in MD: {sorted(missing)}")
            if extra:
                msg_parts.append(f"Extra in MD: {sorted(extra)}")
            details.append("; ".join(msg_parts))

            # Check: are they renumbered (same count but different numbers)?
            if len(missing) == len(extra) and not missing:
                details.append("Superscripts appear to have been renumbered")

            return RuleMatch(
                rule_id="footnote_mismatch",
                rule_name="Footnote Number and Anchor Consistency",
                matched=True,
                severity="major",
                severity_weight=0.25,
                confidence_delta=0.25,
                reason=f"Footnote superscript mismatch: {'; '.join(msg_parts)}",
                details=details,
            )

        return RuleMatch(
            rule_id="footnote_mismatch",
            rule_name="Footnote Number and Anchor Consistency",
            matched=False,
            severity="major",
            severity_weight=0.25,
            confidence_delta=0.0,
            reason="Footnote superscripts match PDF anchors.",
        )


class _CellBoundaryBleedRule:
    """Rule: cell_boundary_bleed — detect words split across cell boundaries."""

    rule_id = "cell_boundary_bleed"
    rule_name = "Cell Boundary Bleed Detection"
    severity = "major"
    severity_weight = 0.15

    # Fragments that are known bleed indicators (from training data)
    SUSPICIOUS_FRAGMENTS = {
        "ed": 3,   # length threshold
        "ved": 3,
        "ing": 3,
        "fee": 3,
        "harge": 5,
        "ree": 3,
        "able": 4,
        "cable": 5,
    }

    # Guard: fragments that are currency/numeric noise — not real bleeds
    CURRENCY_PATTERNS = re.compile(
        r"^(?:HK\$?|RMB|US\$?|RMB|USD|HKD|SGD|THB|GBP|EUR|CHF|CAD|AUD|NZD)\s*\d",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_currency_noise(text: str) -> bool:
        return bool(_CellBoundaryBleedRule.CURRENCY_PATTERNS.match(text.strip()))

    @staticmethod
    def _is_suppression_guard(fragment: str, left_context: str, right_context: str) -> bool:
        """
        Apply false-positive suppression guards.
        Returns True if this bleed should be suppressed.
        """
        f = fragment.strip().lower()
        l = left_context.strip().lower()
        r = right_context.strip().lower()

        # Guard: fragment length <= 2 AND both adjacent cells are non-empty → line-wrap artifact
        if len(f) <= 2 and l and r:
            return True

        # Guard: fragment is purely numeric
        if f.isdigit():
            return True

        # Guard: fragment is a single punctuation
        if re.match(r"^[.,;:\\-]$", f):
            return True

        # Guard: fragment is currency noise
        if _CellBoundaryBleedRule._is_currency_noise(f):
            return True

        # Guard: fragment is part of a footnote marker
        if re.match(r"^[⁰¹²³⁴⁵⁶⁷⁸⁹]+$", fragment):
            return True

        # Guard: fragment looks like "ed" completing a word tail that already ends a sentence
        # e.g. "Waive" + "ed" → "Waved" is unlikely in a financial table
        # But "Wai" + "ved" → "Waved" is suspicious
        known_suspicious = {
            ("wai", "ved"): True,   # "Wai" + "ved" → real bleed
            ("mini", "mum"): True,   # "mini" + "mum" → real bleed
            ("appli", "cable"): True,  # "appli" + "cable" → real bleed
            ("no char", "ged"): True,  # "No char" + "ged" → real bleed
        }

        for (tail, head), real_bleed in known_suspicious.items():
            if l.endswith(tail) and f == head:
                return not real_bleed  # if it's a real bleed, don't suppress

        return False

    @classmethod
    def check(cls, table: ParsedTable) -> RuleMatch:
        details: list[str] = []
        bleeds_found = 0

        for row_idx, row in enumerate(table.rows):
            for col_idx in range(len(row) - 1):
                left_cell = row[col_idx]
                right_cell = row[col_idx + 1]

                left_tail = left_cell.text.strip()
                right_head = right_cell.text.strip()

                # Skip if either cell is empty
                if not left_tail or not right_head:
                    continue

                # Check if right cell starts with a suspicious fragment
                for fragment, min_len in cls.SUSPICIOUS_FRAGMENTS.items():
                    if right_head.lower().startswith(fragment) and len(right_head) <= min_len + 2:
                        if not cls._is_suppression_guard(right_head, left_tail, ""):
                            bleeds_found += 1
                            details.append(
                                f"Row {row_idx + 1}, Col {col_idx + 1}: "
                                f"Cell ends '{left_tail[-6:]}...', next cell starts with '{right_head[:8]}...' "
                                f"(fragment '{fragment}')"
                            )

        if bleeds_found > 0:
            return RuleMatch(
                rule_id="cell_boundary_bleed",
                rule_name="Cell Boundary Bleed Detection",
                matched=True,
                severity="major",
                severity_weight=0.15,
                confidence_delta=0.15,
                reason=f"Found {bleeds_found} potential cell boundary bleeds",
                details=details,
            )

        return RuleMatch(
            rule_id="cell_boundary_bleed",
            rule_name="Cell Boundary Bleed Detection",
            matched=False,
            severity="major",
            severity_weight=0.15,
            confidence_delta=0.0,
            reason="No cell boundary bleeds detected.",
        )


class _CellContentSubstitutionRule:
    """Rule: cell_content_substitution — cell values must not differ between PDF and MD."""

    rule_id = "cell_content_substitution"
    rule_name = "Cell Content Value Consistency"
    severity = "critical"
    severity_weight = 0.30

    # Pairs of values that should be treated as semantically equivalent (not flagged)
    SEMANTIC_EQUIVALENTS = {
        frozenset(["no charge", "waived", "free", "nil", "n/a", "not applicable"]): frozenset(["no charge", "waived", "free", "nil", "n/a", "not applicable"]),
    }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize cell text for comparison."""
        t = text.lower().strip()
        t = re.sub(r"<[^>]+>", "", t)
        t = re.sub(r"\s+", " ", t)
        t = t.replace("\u00a0", " ")  # non-breaking space
        t = re.sub(r"[£$€¥]", "", t)  # strip currency symbols for comparison
        t = re.sub(r"[()]", "", t)
        return t

    @staticmethod
    def _is_na(text: str) -> bool:
        n = _CellContentSubstitutionRule._normalize(text)
        return n in ("n/a", "not applicable", "na", "n a", "nil")

    @staticmethod
    def _is_waived(text: str) -> bool:
        n = _CellContentSubstitutionRule._normalize(text)
        return n in ("waived", "waive", "free", "no charge", "nil")

    @classmethod
    def check(
        cls,
        table: ParsedTable,
        pdf_table_cells: Optional[list[list[str]]] = None,
    ) -> RuleMatch:
        if pdf_table_cells is None or not pdf_table_cells:
            return RuleMatch(
                rule_id="cell_content_substitution",
                rule_name="Cell Content Value Consistency",
                matched=False,
                severity="critical",
                severity_weight=0.30,
                confidence_delta=0.0,
                reason="No PDF table data for cell-by-cell comparison.",
            )

        details: list[str] = []
        errors = 0

        # Align rows between PDF and MD
        # PDF rows start at index 1 (index 0 = header)
        pdf_data_rows = pdf_table_cells[1:] if len(pdf_table_cells) > 1 else []

        for row_idx, md_row in enumerate(table.rows):
            if row_idx >= len(pdf_data_rows):
                continue
            pdf_row = pdf_data_rows[row_idx]

            for col_idx, md_cell in enumerate(md_row):
                if col_idx >= len(pdf_row):
                    continue
                pdf_cell = pdf_row[col_idx]

                pdf_norm = cls._normalize(pdf_cell)
                md_norm = cls._normalize(md_cell.text)

                # Exact match
                if pdf_norm == md_norm:
                    continue

                # Guard: both are N/A → correct
                if cls._is_na(pdf_cell) and cls._is_na(md_cell.text):
                    continue

                # Guard: both are waived equivalents → correct
                if cls._is_waived(pdf_cell) and cls._is_waived(md_cell.text):
                    continue

                # Check: N/A in PDF but not in MD → error
                if cls._is_na(pdf_cell) and not cls._is_na(md_cell.text):
                    errors += 1
                    details.append(
                        f"Row {row_idx + 1}, Col {col_idx + 1}: PDF='{pdf_cell.strip()}' "
                        f"but MD='{md_cell.text.strip()}' (expected N/A)"
                    )
                    continue

                # Check: major value mismatch (not whitespace/formatting difference)
                # A difference of more than 20% in character count suggests content change
                if max(len(pdf_norm), len(md_norm)) > 0:
                    similarity = len(set(pdf_norm) & set(md_norm)) / max(len(set(pdf_norm)), len(set(md_norm)))
                    if similarity < 0.4:
                        errors += 1
                        details.append(
                            f"Row {row_idx + 1}, Col {col_idx + 1}: "
                            f"PDF='{pdf_cell.strip()[:40]}' vs MD='{md_cell.text.strip()[:40]}' "
                            f"(low similarity: {similarity:.0%})"
                        )

        if errors > 0:
            return RuleMatch(
                rule_id="cell_content_substitution",
                rule_name="Cell Content Value Consistency",
                matched=True,
                severity="critical",
                severity_weight=0.30,
                confidence_delta=0.30,
                reason=f"Found {errors} cell content mismatches",
                details=details,
            )

        return RuleMatch(
            rule_id="cell_content_substitution",
            rule_name="Cell Content Value Consistency",
            matched=False,
            severity="critical",
            severity_weight=0.30,
            confidence_delta=0.0,
            reason="All cell values match PDF source.",
        )


class _RowMergeErrorRule:
    """Rule: row_merge_error — MD must not collapse child rows into parent rows."""

    rule_id = "row_merge_error"
    rule_name = "Row Hierarchy and Merge Detection"
    severity = "major"
    severity_weight = 0.20

    BULLET_CHARS = {"•", "-", "*", "–", "—", "○", "‣", "·"}

    @staticmethod
    def _has_bullet(text: str) -> bool:
        t = text.strip()
        return len(t) > 0 and (t[0] in _RowMergeErrorRule.BULLET_CHARS or t.startswith("- ") or t.startswith("* "))

    @classmethod
    def check(
        cls,
        table: ParsedTable,
        pdf_table_cells: Optional[list[list[str]]] = None,
    ) -> RuleMatch:
        details: list[str] = []

        # If no PDF data, fall back to structural heuristics
        if pdf_table_cells is None or len(pdf_table_cells) <= 1:
            # Structural check: look for bullet patterns in MD that suggest hierarchy
            md_bullet_rows = [i for i, row in enumerate(table.rows) if cls._has_bullet(row[0].text) if row]
            if len(md_bullet_rows) > 0:
                return RuleMatch(
                    rule_id="row_merge_error",
                    rule_name="Row Hierarchy and Merge Detection",
                    matched=False,  # bullets are present → no merge happened
                    severity="major",
                    severity_weight=0.20,
                    confidence_delta=0.0,
                    reason="Row hierarchy appears intact (bullet patterns present).",
                )
            return RuleMatch(
                rule_id="row_merge_error",
                rule_name="Row Hierarchy and Merge Detection",
                matched=False,
                severity="major",
                severity_weight=0.20,
                confidence_delta=0.0,
                reason="No PDF data for row merge comparison; no obvious structural issues detected.",
            )

        pdf_row_count = len(pdf_table_cells) - 1  # exclude header
        md_row_count = len(table.rows)

        # Check: MD has significantly fewer rows than PDF → merge likely
        if md_row_count < pdf_row_count * 0.8:
            missing = pdf_row_count - md_row_count
            details.append(
                f"MD has {md_row_count} rows vs PDF {pdf_row_count} rows "
                f"({missing} rows missing — possible merge)"
            )

            # Try to identify which rows are missing by looking for bullet patterns in PDF
            pdf_has_bullets = any(
                cls._has_bullet(row[0]) if row else False
                for row in pdf_table_cells[1:]
            )
            if pdf_has_bullets:
                details.append("PDF has bullet sub-items that may have been merged into parent rows")

            return RuleMatch(
                rule_id="row_merge_error",
                rule_name="Row Hierarchy and Merge Detection",
                matched=True,
                severity="major",
                severity_weight=0.20,
                confidence_delta=0.20,
                reason=f"MD row count ({md_row_count}) significantly lower than PDF ({pdf_row_count})",
                details=details,
            )

        return RuleMatch(
            rule_id="row_merge_error",
            rule_name="Row Hierarchy and Merge Detection",
            matched=False,
            severity="major",
            severity_weight=0.20,
            confidence_delta=0.0,
            reason=f"Row count matches (MD: {md_row_count}, PDF: {pdf_row_count}).",
        )


class _ColumnStructureErrorRule:
    """Rule: column_structure_error — column headers and counts must match between PDF and MD."""

    rule_id = "column_structure_error"
    rule_name = "Column Header and Structure Consistency"
    severity = "major"
    severity_weight = 0.25

    KEY_TERMS = {"customer", "account", "premier", "jade", "one", "personal", "integrated", "item", "service", "annual", "fee"}

    @staticmethod
    def _header_terms_match(pdf_header: str, md_header: str) -> bool:
        pdf_terms = set(re.findall(r"[a-z]{3,}", pdf_header.lower()))
        md_terms = set(re.findall(r"[a-z]{3,}", md_header.lower()))
        overlap = pdf_terms & md_terms
        return len(overlap) >= min(2, len(pdf_terms))

    @classmethod
    def check(
        cls,
        table: ParsedTable,
        pdf_table_cells: Optional[list[list[str]]] = None,
    ) -> RuleMatch:
        if pdf_table_cells is None or not pdf_table_cells:
            return RuleMatch(
                rule_id="column_structure_error",
                rule_name="Column Header and Structure Consistency",
                matched=False,
                severity="major",
                severity_weight=0.25,
                confidence_delta=0.0,
                reason="No PDF table data for column structure check.",
            )

        details: list[str] = []
        errors = 0

        pdf_headers = pdf_table_cells[0] if pdf_table_cells else []
        md_headers = table.header_texts()

        pdf_col_count = len(pdf_headers)
        md_col_count = len(md_headers)

        # Check column count
        if md_col_count != pdf_col_count:
            details.append(
                f"Column count mismatch: PDF has {pdf_col_count}, MD has {md_col_count}"
            )
            errors += 1

        # Check header alignment
        min_cols = min(pdf_col_count, md_col_count)
        for col_idx in range(min_cols):
            pdf_h = pdf_headers[col_idx].strip()
            md_h = md_headers[col_idx].strip()
            if pdf_h and md_h and not cls._header_terms_match(pdf_h, md_h):
                details.append(
                    f"Column {col_idx + 1}: PDF header='{pdf_h[:30]}' vs MD header='{md_h[:30]}' "
                    f"(key terms mismatch)"
                )
                errors += 1

        if errors > 0:
            return RuleMatch(
                rule_id="column_structure_error",
                rule_name="Column Header and Structure Consistency",
                matched=True,
                severity="major",
                severity_weight=0.25,
                confidence_delta=0.25,
                reason=f"Column structure mismatch ({errors} issues found)",
                details=details,
            )

        return RuleMatch(
            rule_id="column_structure_error",
            rule_name="Column Header and Structure Consistency",
            matched=False,
            severity="major",
            severity_weight=0.25,
            confidence_delta=0.0,
            reason="Column structure matches PDF source.",
        )


class _FootnoteOrderErrorRule:
    """Rule: footnote_order_error — footnote superscript ordering must match PDF reading order."""

    rule_id = "footnote_order_error"
    rule_name = "Footnote Superscript Order Preservation"
    severity = "minor"
    severity_weight = 0.15

    # Unicode superscript to digit mapping
    SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

    @staticmethod
    def _extract_superscript_sequence(text: str) -> list[str]:
        """Extract ordered list of superscript numbers from text."""
        sequence: list[str] = []

        # <sup>...</sup>
        for m in re.finditer(r"<sup[^>]*>([^<]+)</sup>", text, re.IGNORECASE):
            for part in m.group(1).split(","):
                part = part.strip()
                if part:
                    sequence.append(part)

        # Unicode superscript digits
        for m in re.finditer(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", text):
            s = m.group(0).translate(_FootnoteOrderErrorRule.SUPERSCRIPT_MAP)
            for ch in s:
                sequence.append(ch)

        return sequence

    @staticmethod
    def _is_ascending(seq: list[str]) -> bool:
        try:
            nums = [int(x) for x in seq]
            return nums == sorted(nums)
        except ValueError:
            return False

    @classmethod
    def check(cls, table: ParsedTable) -> RuleMatch:
        details: list[str] = []

        # Collect superscript sequences per cell
        md_sequences: list[tuple[int, int, list[str]]] = []  # (row, col, sequence)
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row):
                seq = cls._extract_superscript_sequence(cell.raw)
                if len(seq) > 1:  # only check cells with multiple superscripts
                    md_sequences.append((row_idx, col_idx, seq))

        if not md_sequences:
            return RuleMatch(
                rule_id="footnote_order_error",
                rule_name="Footnote Superscript Order Preservation",
                matched=False,
                severity="minor",
                severity_weight=0.15,
                confidence_delta=0.0,
                reason="No multi-superscript cells found.",
            )

        # Check for normalization (ascending order when PDF might not have been)
        normalized_count = 0
        for row_idx, col_idx, seq in md_sequences:
            if cls._is_ascending(seq):
                normalized_count += 1
                details.append(
                    f"Row {row_idx + 1}, Col {col_idx + 1}: superscripts appear normalized "
                    f"to ascending order: {seq}"
                )

        if normalized_count > 0:
            return RuleMatch(
                rule_id="footnote_order_error",
                rule_name="Footnote Superscript Order Preservation",
                matched=True,
                severity="minor",
                severity_weight=0.15,
                confidence_delta=0.15,
                reason=f"Found {normalized_count} cells with normalized superscript ordering",
                details=details,
            )

        return RuleMatch(
            rule_id="footnote_order_error",
            rule_name="Footnote Superscript Order Preservation",
            matched=False,
            severity="minor",
            severity_weight=0.15,
            confidence_delta=0.0,
            reason="Footnote superscript ordering appears correct.",
        )


# ----------------------------------------------------------------------
# Rules engine
# ----------------------------------------------------------------------

class RulesEngine:
    """
    Loads verification rules from JSON and runs all rule checks against parsed tables.

    Usage:
        engine = RulesEngine(rules_path="verification_rules.json")
        result = engine.check_table(parsed_table, pdf_table_cells=pdf_cells, pdf_footnotes=footnotes)
    """

    def __init__(self, rules_path: str | None = None):
        if rules_path is None:
            # Default to verification_rules.json in the repo root
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = os.path.join(base, "verification_rules.json")

        with open(rules_path, encoding="utf-8") as f:
            config = json.load(f)

        self._rules_config = config
        self._thresholds = config.get("confidence_thresholds", {"correct": 0.80, "review": 0.60})

        # Map rule_id → rule class
        self._rule_classes: dict[str, type] = {
            "footnote_mismatch": _FootnoteMismatchRule,
            "cell_boundary_bleed": _CellBoundaryBleedRule,
            "cell_content_substitution": _CellContentSubstitutionRule,
            "row_merge_error": _RowMergeErrorRule,
            "column_structure_error": _ColumnStructureErrorRule,
            "footnote_order_error": _FootnoteOrderErrorRule,
        }

    @property
    def thresholds(self) -> dict:
        return self._thresholds

    def check_table(
        self,
        table: ParsedTable,
        pdf_table_cells: list[list[str]] | None = None,
        pdf_footnotes: dict[str, str] | None = None,
        llm_client=None,
    ) -> RuleCheckResult:
        """
        Run all applicable rules against a single parsed table.

        Parameters
        ----------
        table : ParsedTable
            The parsed markdown table.
        pdf_table_cells : list[list[str]] | None
            PDF table cells as 2D array (from pdf_tables_by_page.json).
            Index 0 = header row, subsequent rows = data rows.
        pdf_footnotes : dict[str, str] | None
            PDF footnote map: {"1": "footnote text", ...}
        llm_client : openai client | None
            Optional LLM client for ambiguous rule checks.

        Returns
        -------
        RuleCheckResult
        """
        result = RuleCheckResult(table_index=table.table_index)
        rules_config = {r["rule_id"]: r for r in self._rules_config.get("rules", [])}

        # Run each rule
        for rule_id, rule_class in self._rule_classes.items():
            cfg = rules_config.get(rule_id, {})

            try:
                if rule_id == "footnote_mismatch":
                    match = rule_class.check(table, pdf_table_cells, pdf_footnotes)
                elif rule_id == "cell_boundary_bleed":
                    match = rule_class.check(table)
                elif rule_id == "cell_content_substitution":
                    match = rule_class.check(table, pdf_table_cells)
                elif rule_id == "row_merge_error":
                    match = rule_class.check(table, pdf_table_cells)
                elif rule_id == "column_structure_error":
                    match = rule_class.check(table, pdf_table_cells)
                elif rule_id == "footnote_order_error":
                    match = rule_class.check(table)
                else:
                    continue

                # Override severity from config if provided
                if cfg.get("severity"):
                    match = RuleMatch(
                        rule_id=match.rule_id,
                        rule_name=match.rule_name,
                        matched=match.matched,
                        severity=cfg["severity"],
                        severity_weight=cfg.get("severity_weight", match.severity_weight),
                        confidence_delta=match.confidence_delta,
                        reason=match.reason,
                        details=match.details,
                    )

                result.matches.append(match)

            except Exception as e:
                # Rule check failed — log but don't crash
                result.matches.append(RuleMatch(
                    rule_id=rule_id,
                    rule_name=cfg.get("rule_name", rule_id),
                    matched=False,
                    severity=cfg.get("severity", "unknown"),
                    severity_weight=cfg.get("severity_weight", 0.0),
                    confidence_delta=0.0,
                    reason=f"Rule check failed with error: {e}",
                ))

        return result
