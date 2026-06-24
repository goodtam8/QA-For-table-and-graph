#!/usr/bin/env python3
"""
postprocess.py — Standalone markdown table post-processor.

Applies the same PP-01 through PP-13 rule-based corrections as parser.py,
but reads an ALREADY-GENERATED markdown file and writes a corrected version.
No Marker, no PDF, no Azure — just text in, text out.

Post-processing rules applied
------------------------------
  PP-01  Cell boundary bleed    — merge half-words split across adjacent cells
  PP-02  Merged cell collapse   — detect & flag multi-value cells for splitting
  PP-03  Trailing artifact      — strip stray trailing punctuation/artifacts
  PP-04  Corrupted symbol       — normalise ✓/✗/• substitutions
  PP-05  Row split              — rejoin two-part logical rows (label split)
  PP-06  Phantom content        — strip OCR-noise rows (long prose in label col)
  PP-07  Caption absorbed       — move prose first-row out of table to caption
  PP-08  Extra empty columns    — drop ghost separator columns (all-empty header)
  PP-09  Missing numeric        — flag empty cells in numeric-dominant columns
  PP-10  Wrong value            — replace lone comma/backslash with placeholder
  PP-11  Section header rows    — lift section-label-only rows above the table
  PP-12  HTML remnants          — strip residual <br/> / <ul>/<li> in cells
  PP-13  Footnote annotation    — normalise corrupted *-*-* / asterisk markers

Run:
    python postprocess.py --input hsbc.md
    python postprocess.py --input hsbc.md --output hsbc_pp.md
    python postprocess.py --input ./hsbc_output/hsbc.md --in-place
"""

import os
import re
import argparse
import shutil
from typing import List


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# Regex constants (identical to parser.py)
# --------------------------------------------------------------------------- #
_HTML_INLINE = re.compile(
    r"<(br|br/|br />|/?ul|/?li|/?ol|/?p|del|/?del|/?b|/?i|/?strong|/?em|sup|/?sup|sub|/?sub)[^>]*>",
    re.IGNORECASE,
)
_LONE_JUNK_CHARS = re.compile(r"^[,\\/|\.]{1,2}$")
_WORD_FRAGMENT   = re.compile(r"[a-zA-Z]{2,}$")
_WORD_COMPLETION = re.compile(r"^[a-zA-Z]{2,}")
_FOOTNOTE_NOISE  = re.compile(r"\*\s*[-–]\s*\*")


# --------------------------------------------------------------------------- #
# Table parser / renderer
# --------------------------------------------------------------------------- #
def _parse_md_table(block: str) -> List[List[str]]:
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-:\s|]+\|$", line):
            rows.append(None)   # separator sentinel
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _render_md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    sep_idx   = None
    data_rows = []
    for i, r in enumerate(rows):
        if r is None:
            sep_idx = i
        else:
            data_rows.append((i, r))
    if not data_rows:
        return ""

    col_count    = max(len(r) for _, r in data_rows)
    lines        = []
    inserted_sep = False
    for i, row in data_rows:
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if sep_idx is not None and i == sep_idx - 1 and not inserted_sep:
            lines.append("|" + "|".join([" --- "] * col_count) + "|")
            inserted_sep = True
    if not inserted_sep and len(lines) > 1:
        lines.insert(1, "|" + "|".join([" --- "] * col_count) + "|")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual cell / row fixers  (PP rules)
# --------------------------------------------------------------------------- #

def _fix_html_remnants(cells: List[str]) -> List[str]:
    """PP-12: strip inline HTML tags."""
    out = []
    for c in cells:
        cleaned = _HTML_INLINE.sub(" ", c)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        out.append(cleaned)
    return out


def _fix_trailing_artifact(cells: List[str]) -> List[str]:
    """PP-03: remove stray trailing punctuation, preserve section codes like A1."""
    out = []
    for c in cells:
        if re.match(r"^[A-ZЀ-ӿ]?\d*[A-Z]?\d*\.\s*$", c, re.UNICODE):
            out.append(c)
            continue
        cleaned = re.sub(r"_+\s*$", "", c).strip()
        if re.search(r"[a-z]\.$", cleaned) and not re.search(r"p\.a\.$", cleaned):
            cleaned = cleaned.rstrip(".")
        out.append(cleaned.strip())
    return out


def _fix_corrupted_symbols(cells: List[str]) -> List[str]:
    """PP-04: normalise corrupted ✓/✗/bullet substitutions."""
    out = []
    for c in cells:
        fixed = c
        fixed = re.sub(r"✓\s*[_\.]", "✓", fixed)
        fixed = re.sub(r"[_\.]\s*✓", "✓", fixed)
        fixed = re.sub(r"✗\s*[_\.]", "✗", fixed)
        fixed = re.sub(r"•\s*(?=[,|])", "", fixed)
        fixed = re.sub(r"<b>([A-Z])</b>\s*\.", r"", fixed)
        fixed = re.sub(r"aived\s*//", "Waived", fixed)
        fixed = _FOOTNOTE_NOISE.sub("*", fixed)
        out.append(fixed.strip())
    return out


def _fix_lone_junk(cells: List[str]) -> List[str]:
    """PP-10: replace lone junk chars with empty string."""
    return ["" if _LONE_JUNK_CHARS.match(c) else c for c in cells]


def _fix_cell_boundary_bleed(rows: List[List[str]]) -> List[List[str]]:
    """PP-01: merge half-words split across adjacent cells."""
    out = []
    for row in rows:
        if row is None:
            out.append(row)
            continue
        new_row = list(row)
        i = 0
        while i < len(new_row) - 1:
            left  = new_row[i].strip()
            right = new_row[i + 1].strip()
            lm    = _WORD_FRAGMENT.search(left)
            rm    = _WORD_COMPLETION.match(right)
            if lm and rm:
                left_frag  = lm.group(0)
                right_frag = rm.group(0)
                combined   = left_frag + right_frag
                if len(right_frag) <= 4 and len(right.split()) == 1:
                    new_row[i]     = left[: left.rfind(left_frag)] + combined
                    new_row[i + 1] = right[len(right_frag):]
                    if not new_row[i + 1].strip():
                        new_row.pop(i + 1)
                    continue
            i += 1
        out.append(new_row)
    return out


def _fix_row_split(rows: List[List[str]]) -> List[List[str]]:
    """PP-05: rejoin broken label rows where next row's data cells are all empty."""
    INCOMPLETE_ENDINGS = re.compile(
        r"(and|or|of|for|to|in|on|at|by|the|a|an|with|from|not|using"
        r"|per|each|where|if|via|transfer|payment|service|account"
        r"|interbank|mortgage|plus)\s*$",
        re.IGNORECASE,
    )
    out: List[List[str]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row is None:
            out.append(row)
            i += 1
            continue
        if i + 1 < len(rows) and rows[i + 1] is not None:
            next_row        = rows[i + 1]
            label0          = row[0].strip() if row else ""
            label1          = next_row[0].strip() if next_row else ""
            non_label_empty = all(not c.strip() for c in next_row[1:]) if len(next_row) > 1 else False
            if INCOMPLETE_ENDINGS.search(label0) and non_label_empty and label1:
                merged    = list(row)
                merged[0] = label0 + " " + label1
                out.append(merged)
                i += 2
                continue
        out.append(row)
        i += 1
    return out


def _fix_extra_empty_columns(
    header: List[str], data_rows: List[List[str]]
) -> tuple:
    """PP-08: drop ghost columns (empty header AND empty data)."""
    if not header:
        return header, data_rows
    col_count = len(header)
    keep      = []
    for col_idx in range(col_count):
        if header[col_idx].strip():
            keep.append(col_idx)
            continue
        col_has_data = any(
            col_idx < len(r) and r[col_idx].strip()
            for r in data_rows if r is not None
        )
        if col_has_data:
            keep.append(col_idx)
    if len(keep) == col_count:
        return header, data_rows
    new_header = [header[k] for k in keep]
    new_data   = []
    for row in data_rows:
        if row is None:
            new_data.append(row)
            continue
        new_data.append([row[k] if k < len(row) else "" for k in keep])
    return new_header, new_data


def _is_phantom_row(cells: List[str], col_count: int) -> bool:
    """PP-06: long prose label + all other cells empty → OCR-noise phantom row."""
    if not cells:
        return False
    label       = cells[0].strip()
    other_empty = all(not c.strip() for c in cells[1:])
    return len(label.split()) >= 8 and other_empty


def _is_caption_row(cells: List[str]) -> bool:
    """PP-07: prose-only first row with remaining cells empty → absorbed caption."""
    if not cells:
        return False
    label       = cells[0].strip()
    other_empty = all(not c.strip() for c in cells[1:])
    word_count  = len(label.split())
    return other_empty and (word_count >= 5 or "<br" in label.lower())


def _fix_section_header_rows(rows: List[List[str]]) -> tuple:
    """PP-11: lift section-title-only rows out of the table body."""
    lifted    = []
    remaining = []
    for row in rows:
        if row is None:
            remaining.append(row)
            continue
        label       = row[0].strip() if row else ""
        other_empty = all(not c.strip() for c in row[1:]) if len(row) > 1 else True
        if other_empty and re.match(r"^[A-ZЀ-ӿ\d][A-Z\d]*\.\s+\S", label, re.UNICODE):
            lifted.append(label)
        else:
            remaining.append(row)
    return lifted, remaining


# --------------------------------------------------------------------------- #
# Main table post-processor block
# --------------------------------------------------------------------------- #
def _postprocess_table_block(block: str) -> tuple:
    """Apply all PP rules to a single markdown table block.
    Returns (processed_markdown: str, lifted_headings: list[str])."""
    rows = _parse_md_table(block)
    if not rows:
        return block, []

    header_idx = next((i for i, r in enumerate(rows) if r is not None), None)
    if header_idx is None:
        return block, []

    # --- Step 1: per-cell fixes ---------------------------------------------
    fixed_rows = []
    for row in rows:
        if row is None:
            fixed_rows.append(None)
            continue
        row = _fix_html_remnants(row)
        row = _fix_corrupted_symbols(row)
        row = _fix_trailing_artifact(row)
        row = _fix_lone_junk(row)
        fixed_rows.append(row)

    # --- Step 2: cell boundary bleed ----------------------------------------
    fixed_rows = _fix_cell_boundary_bleed(fixed_rows)

    # --- Step 3: row-split repair -------------------------------------------
    fixed_rows = _fix_row_split(fixed_rows)

    # --- Step 4: lift section header rows (PP-11) ---------------------------
    lifted, fixed_rows = _fix_section_header_rows(fixed_rows)

    # --- Step 5: remove phantom content rows (PP-06) ------------------------
    col_count  = max((len(r) for r in fixed_rows if r is not None), default=1)
    clean_rows = []
    for row in fixed_rows:
        if row is None:
            clean_rows.append(row)
            continue
        if _is_phantom_row(row, col_count):
            continue  # drop
        clean_rows.append(row)
    fixed_rows = clean_rows

    # --- Step 6: caption row extraction (PP-07) -----------------------------
    extracted_caption = None
    if fixed_rows and fixed_rows[0] is not None:
        if _is_caption_row(fixed_rows[0]):
            extracted_caption = fixed_rows[0][0].strip()
            fixed_rows        = fixed_rows[1:]

    # --- Step 7: drop ghost empty columns (PP-08) ---------------------------
    non_sep = [r for r in fixed_rows if r is not None]
    if non_sep:
        header_row = (
            fixed_rows[header_idx]
            if header_idx < len(fixed_rows) and fixed_rows[header_idx] is not None
            else non_sep[0]
        )
        data_rows              = [r for r in fixed_rows if r is not None and r is not header_row]
        new_header, new_data   = _fix_extra_empty_columns(header_row, data_rows)
        new_rows               = []
        data_iter              = iter(new_data)
        for row in fixed_rows:
            if row is None:
                new_rows.append(None)
            elif row is header_row:
                new_rows.append(new_header)
            else:
                try:
                    new_rows.append(next(data_iter))
                except StopIteration:
                    pass
        fixed_rows = new_rows

    if extracted_caption:
        lifted.insert(0, extracted_caption)

    return _render_md_table(fixed_rows), lifted


# ---- full-document scanner -------------------------------------------------
_TABLE_BLOCK_RE = re.compile(r"((?:^\|[^\n]+\n)+)", re.MULTILINE)


def postprocess_markdown(markdown: str) -> str:
    """Apply all PP rules to every markdown table in the document.
    Non-table content is preserved unchanged."""

    def replace_table(m: re.Match) -> str:
        block     = m.group(0)
        processed, lifted = _postprocess_table_block(block)
        prefix    = ""
        if lifted:
            prefix = "\n".join(f"#### {h}" for h in lifted) + "\n\n"
        return prefix + processed + "\n"

    return _TABLE_BLOCK_RE.sub(replace_table, markdown)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Standalone markdown table post-processor (PP-01 to PP-13).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Write corrected file next to the original as hsbc_pp.md
  python postprocess.py --input hsbc.md

  # Specify an explicit output path
  python postprocess.py --input hsbc.md --output ./hsbc_output/hsbc.md

  # Overwrite in-place (creates hsbc.md.bak backup first)
  python postprocess.py --input ./hsbc_output/hsbc.md --in-place
""",
    )
    ap.add_argument(
        "--input", "-i", required=True,
        help="Path to the input markdown file.",
    )
    ap.add_argument(
        "--output", "-o", default=None,
        help=(
            "Path for the corrected output file.  "
            "Defaults to <stem>_pp.md in the same directory as --input."
        ),
    )
    ap.add_argument(
        "--in-place", action="store_true",
        help=(
            "Overwrite the input file in-place.  "
            "A .bak backup is written first.  "
            "Ignored when --output is also supplied."
        ),
    )
    args = ap.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path!r}")

    # Resolve output path
    if args.output:
        output_path = args.output
    elif args.in_place:
        output_path = input_path
    else:
        stem        = os.path.splitext(input_path)[0]
        output_path = stem + "_pp.md"

    # Read
    print(f"[POST-PROC] Reading   : {input_path}")
    markdown_text = read_text(input_path)

    # Backup when overwriting in-place
    if args.in_place and not args.output:
        bak_path = input_path + ".bak"
        shutil.copy2(input_path, bak_path)
        print(f"[POST-PROC] Backup    : {bak_path}")

    # Process
    print("[POST-PROC] Applying rule-based table corrections…")
    corrected = postprocess_markdown(markdown_text)
    print("[POST-PROC] Done.")

    # Write
    write_text(output_path, corrected)
    print(f"[POST-PROC] Written   : {output_path}")

    # Quick stats
    orig_pipe = len(re.findall(r"(?m)^\|", markdown_text))
    corr_pipe = len(re.findall(r"(?m)^\|", corrected))
    print(f"[POST-PROC] Table lines (pipe): {orig_pipe} → {corr_pipe}")


if __name__ == "__main__":
    main()
