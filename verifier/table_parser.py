"""
table_parser.py — Markdown table extractor and parser.

Extracts all GitHub-Flavored Markdown (GFM) pipe tables from a markdown
document and exposes them as structured ParsedTable objects with:
    - header row
    - data rows
    - raw cell text (HTML stripped)
    - footnote markers per cell
    - boundary bleed candidates per row
"""

from __future__ import annotations

import re
import html
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Cell:
    raw: str
    text: str
    footnote_superscripts: list[str] = field(default_factory=list)
    is_header: bool = False


@dataclass
class ParsedTable:
    """One GFM table extracted from markdown."""
    table_index: int
    page_context: str | None          # section heading / page marker preceding this table
    md_page: int | None              # page number hint (parsed from page marker, if present)
    section: str | None              # section name
    header: list[Cell] = field(default_factory=list)
    rows: list[list[Cell]] = field(default_factory=list)

    @property
    def ncols(self) -> int:
        return len(self.header) if self.header else (self.rows[0] if self.rows else 0)

    @property
    def nrows(self) -> int:
        return len(self.rows)

    def all_cells(self) -> list[Cell]:
        """Flatten header + all data rows into a single list."""
        return self.header + [c for row in self.rows for c in row]

    def cell_at(self, row: int, col: int) -> Optional[Cell]:
        """0-indexed. row=0 is the header row."""
        cells = [self.header] + self.rows
        if 0 <= row < len(cells) and 0 <= col < len(cells[row]):
            return cells[row][col]
        return None

    def header_texts(self) -> list[str]:
        return [c.text for c in self.header]

    def row_texts(self, row_idx: int) -> list[str]:
        if 0 <= row_idx < len(self.rows):
            return [c.text for c in self.rows[row_idx]]
        return []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# Match GFM table rows: starts with |, optionally preceded by delimiter row
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_TABLE_DELIM_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$", re.MULTILINE)

# Split a row string into cell strings (strips |, whitespace, leading/trailing cell whitespace)
_CELL_SPLIT_RE = re.compile(r"\s*\|\s*")


def _strip_html(text: str) -> str:
    """Remove common HTML tags and decode HTML entities."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()


def _extract_superscripts(text: str) -> list[str]:
    """Extract footnote-like superscript markers from cell text.

    Matches:
        <sup>n</sup>   → "n"
        <sup>m,n</sup> → ["m", "n"]
        ¹ ² ³           → ["1", "2", "3"]  (unicode superscript digits)
        ^n^             → "n"
    Returns a list of string footnote numbers.
    """
    superscripts: list[str] = []

    # <sup>...</sup>  (may contain comma-separated numbers)
    for m in re.finditer(r"<sup[^>]*>([^<]+)</sup>", text, re.IGNORECASE):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                superscripts.append(part)

    # Unicode superscript digits: ⁰¹²³⁴⁵⁶⁷⁸⁹
    for m in re.finditer(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", text):
        s = m.group(0)
        # Convert unicode superscript to plain digit
        mapping = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        superscripts.append(s.translate(mapping))

    # Markdown ^n^ footnote syntax
    for m in re.finditer(r"\^(\d+)\^", text):
        superscripts.append(m.group(1))

    return superscripts


def _parse_row(row_str: str, is_header: bool = False) -> list[Cell]:
    """Parse a single pipe-separated row string into a list of Cell objects."""
    # Strip the leading/trailing pipes and split
    parts = _CELL_SPLIT_RE.split(row_str.strip().strip("|"))
    cells: list[Cell] = []
    for part in parts:
        raw = part.strip()
        text = _strip_html(raw)
        superscripts = _extract_superscripts(raw)
        cells.append(Cell(raw=raw, text=text, footnote_superscripts=superscripts, is_header=is_header))
    return cells


# ----------------------------------------------------------------------
# Boundary-bleed detection helpers
# ----------------------------------------------------------------------

# Fragments that look like word splits at cell boundaries
BLEED_FRAGMENT_RE = re.compile(
    r"\b(ed|ved|ing|sion|tion|ment|ance|ence|harge|fee|ree|able|cable|able|ited|ount|arge)\b",
    re.IGNORECASE,
)

# Short fragments at cell boundaries (<=4 chars that are not purely numeric/symbolic)
SHORT_FRAGMENT_RE = re.compile(r"(?<=[|])\s*([a-zA-Z]{1,4}?)\s*(?=\||\|$)", re.MULTILINE)


def _detect_boundary_bleeds_in_row(row_cells: list[Cell]) -> list[dict]:
    """
    Scan a row's cell boundaries for potential word-split bleeds.

    Returns a list of bleed reports:
        {
            "fragment": "ved",
            "position": "right_of_col_2",
            "adjacent_cell_tail": "Wai",
            "likely_word": "Waved",
            "confidence": "high|medium|low"
        }
    """
    reports: list[dict] = []

    for col_idx, cell in enumerate(row_cells):
        cell_tail = cell.text.strip()
        cell_tail_last3 = cell_tail[-3:] if len(cell_tail) >= 3 else ""

        # Check the fragment at the RIGHT boundary of this cell (next cell is the right neighbor)
        if col_idx + 1 < len(row_cells):
            next_cell = row_cells[col_idx + 1]
            next_head = next_cell.text.strip()[:3]

            # If current cell ends with partial word and next starts with partial word,
            # check if concatenating them forms a real word
            candidates = _BLEED_CANDIDATES(cell_tail, next_head)
            for candidate, fragment, position in candidates:
                reports.append({
                    "fragment": fragment,
                    "position": position,
                    "adjacent_cell_tail": cell_tail[-8:] if cell_tail else "",
                    "adjacent_cell_head": next_head,
                    "likely_word": candidate,
                    "confidence": "high" if len(fragment) >= 3 else "medium",
                })

        # Also check the fragment at the LEFT boundary (cell preceded by pipe boundary)
        if col_idx > 0 and col_idx - 1 < len(row_cells):
            prev_cell = row_cells[col_idx - 1]
            prev_tail = prev_cell.text.strip()

            # Look for short fragments (1-4 letters) at the boundary
            # If this cell starts with a partial word, check if it could be a bleed
            cell_head = cell.text.strip()[:3]

            # Check if prev cell ends with partial word and this cell starts with partial
            bleed_candidates = _BLEED_CANDIDATES(prev_tail, cell_head)
            for candidate, fragment, position in bleed_candidates:
                reports.append({
                    "fragment": fragment,
                    "position": f"left_of_col_{col_idx}",
                    "adjacent_cell_tail": prev_tail[-8:] if prev_tail else "",
                    "adjacent_cell_head": cell_head,
                    "likely_word": candidate,
                    "confidence": "high" if len(fragment) >= 3 else "medium",
                })

    return reports


# Known word tails and heads that suggest a bleed when combined
_WORD_COMBINATIONS = [
    ("Wai", "ved", "Waved"),
    ("Wai", "ved", "Waved"),
    ("No char", "ged", "No charged"),
    ("mini", "mum", "minimum"),
    ("appli", "cable", "applicable"),
    ("incl", "ude", "include"),
    ("incl", "uded", "included"),
    ("not appli", "cable", "not applicable"),
    ("Item C", "harge", "Item Charge"),
    ("Account F", "ree", "Account Free"),
    ("Account F", "ree", "Account Free"),
    ("Wa", "ved", "Waved"),
    ("excl", "usive", "exclusive"),
    ("HK$", "100", "HK$100"),  # currency bleed - suppress
    ("HK$", "20", "HK$20"),    # currency bleed - suppress
]


def _BLEED_CANDIDATES(tail: str, head: str) -> list[tuple[str, str, str]]:
    """
    Given a cell tail and next cell head, return possible bleed reconstructions.

    Returns: [(reconstructed_word, fragment, position_description), ...]
    position_description examples:
        "right_of_last_col"
        "left_of_col_N"
    """
    results: list[tuple[str, str, str]] = []

    if not tail or not head:
        return results

    tail_lower = tail.lower()
    head_lower = head.lower()

    # Known combinations from training data
    for known_tail, known_head, known_word in _WORD_COMBINATIONS:
        if tail_lower.endswith(known_tail) and head_lower.startswith(known_head):
            results.append((known_word, head, "right_boundary"))

    # General heuristic: if tail ends with partial word (>=3 chars) and head
    # starts with partial word (>=2 chars), candidate reconstruction exists
    if len(tail) >= 2 and len(head) >= 2:
        reconstructed = tail + head
        # Only flag if the fragment is non-trivial (>=3 chars) and not currency
        if len(head) >= 3 and not head[0].isdigit():
            results.append((reconstructed, head, "right_boundary"))

    return results


# ----------------------------------------------------------------------
# Main parser class
# ----------------------------------------------------------------------

class MarkdownTableParser:
    """
    Extracts all GFM tables from a markdown string and parses them into
    structured ParsedTable objects.

    Usage:
        parser = MarkdownTableParser(markdown_text)
        for table in parser.tables:
            print(f"Table {table.table_index}: {table.nrows} rows x {table.ncols} cols")
    """

    # Pattern to extract table blocks including preceding page markers
    # Matches: <!-- Page N --> ... or ## Section Name ...  followed by a GFM table
    TABLE_BLOCK_RE = re.compile(
        r"""
        (?:
            # --- optional preamble lines (page markers, section headings) ---
            (?:
                <!--\s*[Pp]age\s*(\d+)\s*-->|
                ^#{1,6}\s+.+$|
                ^\s*<!--.*?-->\s*$|
                ^\s*$$
            )+\n*
        )?
        # --- the table itself ---
        (?:
            \|[^\n]*\|[\r\n]+
            (?:\|[\s\-:|]+\|[\r\n]+)?
            (?:\|[^\n]*\|[\r\n]*)*
        )
        """,
        re.VERBOSE | re.MULTILINE,
    )

    def __init__(self, markdown: str):
        self._markdown = markdown
        self._tables: list[ParsedTable] = []
        self._parse()

    def _parse(self) -> None:
        """Internal: extract and parse all tables."""
        self._tables = []
        text = self._markdown

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into lines
        lines = text.split("\n")

        # Find all table blocks (consecutive lines that look like tables)
        i = 0
        table_index = 0
        in_table = False
        table_lines: list[str] = []
        preamble: list[tuple[str, str]] = []  # (type, value) pairs: ("page", "3") or ("section", "A1...")

        while i < len(lines):
            line = lines[i]

            # Page marker
            page_m = re.match(r"^\s*<!--\s*[Pp]age\s*(\d+)\s*-->\s*$", line)
            if page_m:
                preamble.append(("page", page_m.group(1)))
                i += 1
                continue

            # Section heading
            heading_m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_m:
                # A heading INSIDE a table block terminates it
                if in_table:
                    if table_lines:
                        table = self._build_table(table_lines, preamble, table_index)
                        self._tables.append(table)
                        table_index += 1
                    table_lines = []
                    preamble = []
                    in_table = False
                preamble.append(("section", heading_m.group(2).strip()))
                i += 1
                continue

            # Empty line
            if not line.strip():
                i += 1
                continue

            # Table separator/delimiter row (|---|)
            if self._is_delimiter(line):
                # Start of a new table
                if in_table and table_lines:
                    # Flush previous table
                    table = self._build_table(table_lines, preamble[:-1] if preamble else [], table_index)
                    self._tables.append(table)
                    table_index += 1
                    table_lines = []
                    preamble = []
                in_table = True
                table_lines.append(line)
                i += 1
                continue

            # Data/header row
            if self._is_table_row(line):
                if not in_table:
                    in_table = True
                table_lines.append(line)
                i += 1
                continue

            # Non-table line: flush current table if any
            if in_table and table_lines:
                table = self._build_table(table_lines, preamble, table_index)
                self._tables.append(table)
                table_index += 1
                table_lines = []
                preamble = []
                in_table = False
            else:
                # Reset preamble if we're not in a table
                preamble = []

            i += 1

        # Flush last table
        if table_lines:
            table = self._build_table(table_lines, preamble, table_index)
            self._tables.append(table)

    def _is_delimiter(self, line: str) -> bool:
        """Return True if line is a GFM table delimiter row."""
        stripped = line.strip()
        return bool(re.match(r"^\|[\s\-:|]+\|\s*$", stripped))

    def _is_table_row(self, line: str) -> bool:
        """Return True if line looks like a GFM table data/header row."""
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    def _build_table(self, raw_lines: list[str], preamble: list[tuple[str, str]], table_index: int) -> ParsedTable:
        """Build a ParsedTable from raw text lines and preamble."""
        # Determine page number and section from preamble
        page_num: int | None = None
        section: str | None = None
        for ptype, pval in preamble:
            if ptype == "page":
                try:
                    page_num = int(pval)
                except ValueError:
                    pass
            elif ptype == "section":
                if section is None:
                    section = pval
                else:
                    section += " / " + pval

        # Filter out delimiter rows; keep header + data rows
        data_lines = [ln for ln in raw_lines if not self._is_delimiter(ln)]

        if not data_lines:
            return ParsedTable(
                table_index=table_index,
                page_context=section,
                md_page=page_num,
                section=section,
                header=[],
                rows=[],
            )

        # Parse header (first row)
        header_cells = _parse_row(data_lines[0], is_header=True)
        # Parse data rows
        data_rows: list[list[Cell]] = []
        for line in data_lines[1:]:
            row = _parse_row(line)
            if row:  # skip empty rows
                data_rows.append(row)

        return ParsedTable(
            table_index=table_index,
            page_context=section,
            md_page=page_num,
            section=section,
            header=header_cells,
            rows=data_rows,
        )

    @property
    def tables(self) -> list[ParsedTable]:
        return self._tables

    def table_at(self, index: int) -> ParsedTable | None:
        return self._tables[index] if 0 <= index < len(self._tables) else None

    def table_for_question(self, question: str, context_tables: list[ParsedTable]) -> list[ParsedTable]:
        """
        Given a user question, return the most relevant tables from the provided list.
        Uses keyword matching against header texts and section names.
        """
        q_lower = question.lower()
        keywords = set(re.findall(r"[a-z]{3,}", q_lower))

        scored: list[tuple[int, ParsedTable]] = []
        for tbl in context_tables:
            score = 0
            # Section match
            if tbl.section and any(kw in tbl.section.lower() for kw in keywords):
                score += 5
            # Header text match
            for h in tbl.header_texts():
                if any(kw in h.lower() for kw in keywords):
                    score += 2
            # Row content match
            for row in tbl.rows:
                for cell in row:
                    if any(kw in cell.text.lower() for kw in keywords):
                        score += 1
            if score > 0:
                scored.append((score, tbl))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]
