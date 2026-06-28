"""
confidence_scorer.py — Aggregates rule matches into a confidence score and verdict.

Confidence score formula:
    base_confidence = 1.0
    for each matched rule:
        confidence -= rule.severity_weight
    confidence = max(0.0, min(1.0, confidence))

Verdict:
    confidence >= 0.80  → CORRECT
    0.60 <= confidence < 0.80 → REVIEW (treat as INCORRECT with low confidence)
    confidence < 0.60  → INCORRECT

Also supports LLM-assisted confidence refinement for ambiguous cases.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from .rules_engine import RuleCheckResult, RuleMatch
from .table_parser import ParsedTable

load_dotenv()


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Verdict:
    label: str          # "CORRECT" | "INCORRECT" | "REVIEW"
    confidence: float   # 0.0 to 1.0
    matched_rules: list[RuleMatch] = field(default_factory=list)
    primary_error: str | None = None  # top error reason
    all_errors: list[str] = field(default_factory=list)

    def to_verify_tag(self) -> str:
        """Format as <!-- VERIFY: CORRECT | confidence=X.XX --> or equivalent."""
        if self.label == "CORRECT":
            return f"<!-- VERIFY: CORRECT | confidence={self.confidence:.2f} -->"
        elif self.label == "REVIEW":
            reason = self.primary_error or self.all_errors[0] if self.all_errors else "uncertain"
            return f"<!-- VERIFY: REVIEW | reason={reason} | confidence={self.confidence:.2f} -->"
        else:
            reason = self.primary_error or self.all_errors[0] if self.all_errors else "unknown"
            return (
                f"<!-- VERIFY: INCORRECT | reason={reason} | confidence={self.confidence:.2f} -->"
            )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "primary_error": self.primary_error,
            "all_errors": self.all_errors,
            "matched_rules": [
                {
                    "rule_id": m.rule_id,
                    "rule_name": m.rule_name,
                    "matched": m.matched,
                    "confidence_delta": m.confidence_delta,
                    "reason": m.reason,
                }
                for m in self.matched_rules
            ],
        }


# ----------------------------------------------------------------------
# Confidence scorer
# ----------------------------------------------------------------------

class ConfidenceScorer:
    """
    Aggregates rule matches from RulesEngine into a final confidence score and verdict.

    Usage:
        scorer = ConfidenceScorer()
        verdict = scorer.compute_verdict(rule_check_result)
    """

    def __init__(self, rules_path: str | None = None):
        if rules_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = os.path.join(base, "verification_rules.json")

        with open(rules_path, encoding="utf-8") as f:
            config = json.load(f)

        self._thresholds = config.get("confidence_thresholds", {
            "correct": 0.80,
            "review": 0.60,
        })

    @property
    def thresholds(self) -> dict:
        return self._thresholds

    def compute_verdict(self, result: RuleCheckResult) -> Verdict:
        """
        Compute the final confidence score and verdict from rule check results.

        Algorithm:
        1. Start with base confidence = 1.0
        2. Subtract severity_weight for each matched rule
        3. Clamp to [0.0, 1.0]
        4. Apply threshold to determine label
        """
        base = 1.0
        matched = [m for m in result.matches if m.matched]

        # Sort matched rules by severity weight descending
        matched_sorted = sorted(matched, key=lambda m: m.severity_weight, reverse=True)

        confidence = base
        deductions: list[tuple[str, float]] = []

        for rule in matched_sorted:
            deduction = rule.severity_weight
            confidence -= deduction
            deductions.append((rule.rule_id, deduction))

        confidence = max(0.0, min(1.0, confidence))

        # Determine label based on thresholds
        correct_threshold = self._thresholds.get("correct", 0.70)
        review_threshold = self._thresholds.get("review", 0.50)

        if confidence >= correct_threshold:
            label = "CORRECT"
        elif confidence >= review_threshold:
            label = "REVIEW"  # Changed from "INCORRECT" - borderline case needs review
        else:
            label = "INCORRECT"

        # Primary error: highest-severity matched rule
        primary_error = matched_sorted[0].rule_id if matched_sorted else None

        # All error reasons
        all_errors = [m.rule_id for m in matched_sorted]

        # If confidence is very high but there are errors, cap and downgrade
        # (A table can't be CORRECT if any rule matched)
        if matched and label == "CORRECT":
            # Re-evaluate: if any rule matched with significant weight, downgrade
            max_weight = matched_sorted[0].severity_weight if matched_sorted else 0
            if max_weight >= 0.20:
                # Don't force INCORRECT - use REVIEW instead
                if confidence >= review_threshold:
                    label = "REVIEW"
                else:
                    label = "INCORRECT"

        return Verdict(
            label=label,
            confidence=confidence,
            matched_rules=matched_sorted,
            primary_error=primary_error,
            all_errors=all_errors,
        )

    def compute_with_llm_refinement(
        self,
        result: RuleCheckResult,
        table: ParsedTable,
        pdf_table_cells: list[list[str]] | None,
        llm_client=None,
    ) -> Verdict:
        """
        Optionally refine confidence using LLM for ambiguous cases.

        Called when:
        - Confidence is in the review zone (0.60–0.80)
        - Multiple rules matched with similar severity
        - The scorer wants a second opinion on borderline cases
        """
        base_verdict = self.compute_verdict(result)

        if base_verdict.confidence >= 0.90 and not base_verdict.matched_rules:
            # Already high confidence and no errors → no need for LLM
            return base_verdict

        if llm_client is None:
            return base_verdict

        # LLM refinement prompt
        system_prompt = (
            "You are a table verification assistant. Given a markdown table and its PDF source "
            "table cells, determine if the markdown table is CORRECT or INCORRECT. "
            "Consider: footnote consistency, cell content accuracy, row/column structure, "
            "boundary bleeds, and footnote order. "
            "Reply with ONLY a JSON object: "
            '{"verdict": "CORRECT" | "INCORRECT", "confidence": 0.0-1.0, "reason": "brief explanation"}'
        )

        # Build a concise comparison for the LLM
        md_rows_preview = []
        for row in table.rows[:5]:  # First 5 rows only
            md_rows_preview.append(" | ".join(c.text[:30] for c in row))

        pdf_rows_preview = []
        if pdf_table_cells:
            for row in pdf_table_cells[1:6]:  # Skip header
                pdf_rows_preview.append(" | ".join(c[:30] for c in row))

        user_prompt = f"""Markdown table (first 5 rows):
{chr(10).join(md_rows_preview)}

PDF table (first 5 rows, skipping header):
{chr(10).join(pdf_rows_preview)}

Rules already flagged: {[m.rule_id for m in base_verdict.matched_rules]}
LLM confidence: {base_verdict.confidence:.2f}

Is this table CORRECT or INCORRECT?"""

        try:
            response = llm_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Parse LLM response
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                llm_result = json.loads(match.group(0))
                llm_verdict = llm_result.get("verdict", "UNKNOWN")
                llm_confidence = float(llm_result.get("confidence", base_verdict.confidence))

                # Blend LLM verdict with rule-based verdict
                if llm_verdict == "CORRECT" and base_verdict.label == "INCORRECT":
                    # LLM disagrees → raise confidence slightly but don't override
                    # unless LLM confidence is very high
                    if llm_confidence > 0.85:
                        return Verdict(
                            label="REVIEW",
                            confidence=(base_verdict.confidence + llm_confidence) / 2,
                            matched_rules=base_verdict.matched_rules,
                            primary_error=base_verdict.primary_error,
                            all_errors=base_verdict.all_errors,
                        )

                # If LLM says INCORRECT but base says CORRECT, use REVIEW
                if llm_verdict == "INCORRECT" and base_verdict.label == "CORRECT":
                    if llm_confidence > 0.80:
                        return Verdict(
                            label="REVIEW",
                            confidence=base_verdict.confidence * 0.5,
                            matched_rules=base_verdict.matched_rules,
                            primary_error=base_verdict.primary_error,
                            all_errors=["llm_disagreement"],
                        )

                return base_verdict

        except Exception:
            pass  # Fall back to rule-based verdict

        return base_verdict


# ----------------------------------------------------------------------
# LLM-assisted verification (for high-value / ambiguous tables)
# ----------------------------------------------------------------------

def verify_with_llm(
    table: ParsedTable,
    pdf_table_cells: list[list[str]] | None,
    llm_client,
) -> dict:
    """
    Perform an LLM-only verification for a table (used as a fallback
    when heuristic rules are insufficient).

    Returns a dict with: verdict, confidence, reason, details
    """
    if llm_client is None:
        return {"verdict": "UNKNOWN", "confidence": 0.0, "reason": "No LLM client available"}

    system_prompt = (
        "You are a document verification assistant specialized in bank tariff tables. "
        "Your task is to verify that a markdown table accurately represents the PDF source table. "
        "Check for: (1) cell content accuracy, (2) footnote consistency, "
        "(3) column/row structure, (4) boundary bleeds. "
        "Output a JSON object with:\n"
        '- "verdict": "CORRECT" or "INCORRECT"\n'
        '- "confidence": a float 0.0-1.0\n'
        '- "reason": brief reason for the verdict\n'
        '- "issues": list of specific issues found'
    )

    md_header = " | ".join(h.text[:30] for h in table.header)
    md_sep = " | ".join("-" * 20 for _ in table.header)
    md_preview = "\n".join(
        " | ".join(c.text[:40] for c in row)
        for row in table.rows[:10]
    )

    pdf_preview = ""
    if pdf_table_cells:
        pdf_preview = "\n".join(
            " | ".join(c[:40] for c in row)
            for row in pdf_table_cells[1:11]
        )

    user_prompt = f"""Markdown table (up to 10 rows):
```
| {md_header} |
| {md_sep} |
{md_preview}
```

PDF table (up to 10 rows):
```
{pdf_preview if pdf_table_cells else 'N/A'}
```

Section: {table.section or 'Unknown'}
Page: {table.md_page}

Is this table correct?"""

    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        return {"verdict": "ERROR", "confidence": 0.0, "reason": str(e)}

    return {"verdict": "UNKNOWN", "confidence": 0.0, "reason": "Could not parse LLM response"}
