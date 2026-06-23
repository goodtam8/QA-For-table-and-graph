#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter
from pathlib import Path

RULES = {
    "STRUCTURE_DRIFT": 0.95,
    "CELL_BOUNDARY_FAILURE": 0.92,
    "ROW_FRAGMENTATION": 0.88,
    "SUSPICIOUS_CELL_TOKEN": 0.84,
    "FOOTNOTE_INTEGRITY": 0.78,
    "SECTION_CONTAMINATION": 0.76,
}

VALUE_PATTERNS = [
    r"Waived",
    r"N/A",
    r"Nil",
    r"Free",
    r"No charge",
    r"HK\$ ?\d[\d,]*(?:\.\d+)?",
    r"US\$ ?\d[\d,]*(?:\.\d+)?",
    r"RMB ?\d[\d,]*(?:\.\d+)?",
    r"\d+%",
    r"✓[⁰¹²³⁴⁵⁶⁷⁸⁹]*",
]
MULTI_VALUE_RE = re.compile("|".join(f"(?:{p})" for p in VALUE_PATTERNS))
ALIGN_RE = re.compile(r"^:?-{3,}:?$")
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
CONTINUATION_STARTS = {
    "and", "or", "to", "for", "with", "via", "from", "of", "on", "in",
    "transfer", "transfers", "services", "service", "accounts", "account",
    "payment", "payments", "exchange", "instructions", "instruction",
    "transactions", "transaction", "banking", "charge", "charges"
}
SUSPICIOUS_EXACT = {
    ",", ".", "•", "\\", "~", "_", "-", "√", "V", "’", "‘"
}

def is_table_line(line: str) -> bool:
    s = line.rstrip("\n")
    if not s.strip():
        return False
    if s.strip().startswith("```"):
        return False
    return s.count("|") >= 2

def parse_row(line: str):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]

def is_alignment_row(cells):
    return all(ALIGN_RE.match(c or "---") for c in cells)

def looks_like_section_row(cells):
    if not cells:
        return False
    first = strip_md(cells)
    others = [strip_md(c) for c in cells[1:]]
    nonempty_others = [c for c in others if c]
    if len(first) > 70 and not nonempty_others:
        return True
    if re.search(r"\b(C\d+|G\d+|A\d+|B\.|C\.|D\.|E\.|F\.|G\.|H\.|I\.)\b", first) and not nonempty_others:
        return True
    if re.match(r"^\*{0,2}\[?[A-Z]\d", first) and not nonempty_others:
        return True
    return False

def strip_md(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()

def count_meaningful_values(text: str) -> int:
    return len(MULTI_VALUE_RE.findall(text))

def suspicious_token(cell: str) -> bool:
    raw = strip_md(cell)
    if not raw:
        return False
    if raw in SUSPICIOUS_EXACT:
        return True
    if re.fullmatch(r"[,.;:•~\\/_-]+", raw):
        return True
    if re.search(r"✓[_-]+", raw):
        return True
    if re.search(r"[√V~]", raw) and "HK$" not in raw and "US$" not in raw and "RMB" not in raw:
        return True
    if re.search(r"✓[0-9]+", raw):
        return True
    if re.search(r"[A-Za-z]{1,3}[0-9~]{1,}", raw) and not re.fullmatch(r"[A-Z]\d+\.", raw):
        return True
    if re.fullmatch(r"Wai|ved|ansaction|tr|em|RTGAGE|lakada.*|I4~", raw, re.IGNORECASE):
        return True
    return False

def fragment_pair(left: str, right: str) -> bool:
    a = strip_md(left)
    b = strip_md(right)
    if not a or not b:
        return False
    if " " in a or " " in b:
        return False
    if not a.isalpha() or not b.isalpha():
        return False
    if len(a) > 5 or len(b) > 8:
        return False
    joined = (a + b).lower()
    if len(joined) < 5:
        return False
    return True

def extract_tables(lines):
    tables = []
    i = 0
    while i < len(lines):
        if is_table_line(lines[i]):
            start = i
            block = []
            while i < len(lines) and is_table_line(lines[i]):
                block.append((i + 1, lines[i].rstrip("\n")))
                i += 1
            tables.append(block)
        else:
            i += 1
    return tables

def add(anomalies, line_no, snippet, rule, message):
    anomalies.append({
        "line_number": line_no,
        "exact_markdown_snippet": snippet,
        "triggered_rule": rule,
        "confidence_weight": RULES[rule],
        "error_message": message,
    })

def analyze_table(block, anomalies):
    parsed = []
    for line_no, line in block:
        cells = parse_row(line)
        parsed.append((line_no, line, cells))

    non_align = [(ln, raw, cells) for ln, raw, cells in parsed if not is_alignment_row(cells)]
    if not non_align:
        return

    counts = [len(cells) for _, _, cells in non_align]
    expected = Counter(counts).most_common(1)[0][0]

    # R1: inconsistent column counts
    for line_no, raw, cells in non_align:
        if len(cells) != expected:
            add(
                anomalies, line_no, raw, "STRUCTURE_DRIFT",
                f"Row has {len(cells)} cells but the dominant table width is {expected}; likely merged/split columns or table boundary drift."
            )

    # R2: multiple values collapsed into one cell
    for line_no, raw, cells in non_align:
        for cell in cells:
            raw_cell = strip_md(cell)
            if count_meaningful_values(raw_cell) >= 2 and expected > 2:
                add(
                    anomalies, line_no, raw, "CELL_BOUNDARY_FAILURE",
                    f"Cell '{raw_cell}' appears to contain multiple logical values that should likely be separated across columns or rows."
                )
                break

    # R2: split word/value across adjacent cells
    for line_no, raw, cells in non_align:
        for a, b in zip(cells, cells[1:]):
            if fragment_pair(a, b):
                add(
                    anomalies, line_no, raw, "CELL_BOUNDARY_FAILURE",
                    f"Adjacent cells '{strip_md(a)}' and '{strip_md(b)}' look like one logical token split across a cell boundary."
                )
                break

    # R3: suspicious OCR debris or malformed token
    for line_no, raw, cells in non_align:
        bad_cells = [strip_md(c) for c in cells if suspicious_token(c)]
        if bad_cells:
            add(
                anomalies, line_no, raw, "SUSPICIOUS_CELL_TOKEN",
                f"Suspicious OCR-like or malformed cell token(s) detected: {bad_cells[:4]}."
            )

    # R4: likely row fragmentation
    for idx in range(len(non_align) - 1):
        ln1, raw1, c1 = non_align[idx]
        ln2, raw2, c2 = non_align[idx + 1]
        if len(c1) != len(c2):
            continue
        first1 = strip_md(c1[0])
        first2 = strip_md(c2[0])
        if not first1 or not first2:
            continue

        empty_ratio_1 = sum(1 for x in c1[1:] if not strip_md(x)) / max(1, len(c1) - 1)
        empty_ratio_2 = sum(1 for x in c2[1:] if not strip_md(x)) / max(1, len(c2) - 1)
        first2_start = first2.split()[0].lower() if first2.split() else ""

        continuation = (
            first2[:1].islower() or
            first2_start in CONTINUATION_STARTS or
            (len(first1.split()) <= 4 and len(first2.split()) <= 4 and (empty_ratio_1 > 0.5 or empty_ratio_2 > 0.5))
        )

        if continuation:
            add(
                anomalies,
                ln1,
                raw1 + "\n" + raw2,
                "ROW_FRAGMENTATION",
                "Two adjacent rows look like one logical PDF row that was fragmented during Markdown conversion."
            )

    # R6: section or caption contamination inside table
    section_rows = [(ln, raw, cells) for ln, raw, cells in non_align if looks_like_section_row(cells)]
    if len(section_rows) >= 2:
        for line_no, raw, _ in section_rows:
            add(
                anomalies, line_no, raw, "SECTION_CONTAMINATION",
                "This row looks like a section caption or neighboring table header absorbed into the table body."
            )

def analyze_footnotes(lines, anomalies):
    numbered = []
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if re.match(r"^\d+\s+\S+", s):
            num = int(re.match(r"^(\d+)", s).group(1))
            numbered.append((i, num, s))

    # Missing numbering: long paragraph right after a numbered footnote block
    for i in range(len(lines)):
        s = lines[i].strip()
        if not s:
            continue
        prev_num = re.match(r"^\d+\s+\S+", lines[i - 1].strip()) if i > 0 else None
        if prev_num and not re.match(r"^\d+\s+\S+", s):
            if len(s) > 60 and not is_table_line(s):
                add(
                    anomalies, i + 1, lines[i].rstrip("\n"), "FOOTNOTE_INTEGRITY",
                    "Long footnote-like paragraph is missing a leading footnote number."
                )

    # Non-monotonic order
    for (line_no_a, num_a, _), (line_no_b, num_b, _) in zip(numbered, numbered[1:]):
        if num_b < num_a:
            add(
                anomalies, line_no_b, lines[line_no_b - 1].rstrip("\n"), "FOOTNOTE_INTEGRITY",
                f"Footnote numbering is out of order: {num_a} followed by {num_b}."
            )

    # Stray punctuation lines
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if re.fullmatch(r"[.]+", s):
            add(
                anomalies, i, line.rstrip("\n"), "FOOTNOTE_INTEGRITY",
                "Stray punctuation-only line suggests a trailing artifact or broken footnote text."
            )

    # Truncated footnote-like lines
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if re.match(r"^\d+\s+\S+", s):
            if len(s) > 40 and not re.search(r"[.!?]$|[:;]$", s):
                add(
                    anomalies, i, line.rstrip("\n"), "FOOTNOTE_INTEGRITY",
                    "Numbered footnote line appears truncated because it ends without normal sentence closure."
                )

def verify_markdown(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)
    anomalies = []

    tables = extract_tables(lines)
    for block in tables:
        analyze_table(block, anomalies)

    analyze_footnotes(lines, anomalies)

    seen = set()
    deduped = []
    for item in anomalies:
        key = (
            item["line_number"],
            item["exact_markdown_snippet"],
            item["triggered_rule"],
            item["error_message"]
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    report = {"anomalies": deduped}
    out_path = md_path.parent / "verification_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, report

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_markdown_table.py <input_markdown_file>")
        sys.exit(1)

    md_path = Path(sys.argv[1]).expanduser().resolve()
    if not md_path.exists():
        print(f"Input file not found: {md_path}")
        sys.exit(1)

    out_path, report = verify_markdown(md_path)
    print(f"Wrote {len(report['anomalies'])} anomalies to {out_path}")

if __name__ == "__main__":
    main()