"""
verifier — General-purpose table verification package.

Architecture:
    table_parser.py   — Parses markdown tables into row/col cell structures.
    rules_engine.py   — Loads rules and runs pattern checks.
    confidence_scorer.py — Aggregates rule match results into a confidence score.
    annotator.py     — Injects VERIFY + PDF_REF comment tags into markdown.
"""

from .table_parser import MarkdownTableParser, ParsedTable
from .rules_engine import RulesEngine
from .confidence_scorer import ConfidenceScorer
from .annotator import MarkdownAnnotator

__all__ = [
    "MarkdownTableParser",
    "ParsedTable",
    "RulesEngine",
    "ConfidenceScorer",
    "MarkdownAnnotator",
]
