"""
models.py — Shared data-model dataclasses for the PDF-to-Markdown Table Verifier.
All other modules import from here; no module-level side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    verdict: str            # "CORRECT" | "INCORRECT"
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


@dataclass
class PDFTableRef:
    page: int
    section: str
    pdf_table_index: int
    header_text: str
    page_text: str
