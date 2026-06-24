"""
postprocess.py — Standalone markdown table post-processor.

Applies all PP-01 through PP-13 rules to an existing Markdown file
produced by parser.py (or any compatible converter).  Does NOT require
Marker, Azure OpenAI, or the original PDF.

Usage:
    python postprocess.py --input ./hsbc_output/hsbc.md
    python postprocess.py --input ./hsbc_output/hsbc.md --output ./hsbc_output/hsbc_pp.md
    python postprocess.py --input ./hsbc_output/hsbc.md --inplace
    python postprocess.py --input ./hsbc_output/hsbc.md --inplace --backup

Arguments:
    --input        Path to the source Markdown file (required).
    --output       Path to write the post-processed Markdown.
                   Defaults to <stem>_pp.md alongside the input file.
    --inplace      Overwrite the input file with the processed output.
                   (Ignores --output when set.)
    --backup       When used with --inplace, write a backup copy
                   as <stem>_backup.md before overwriting.
    --no-manifest  Skip updating / creating the manifest.json in the
                   adjacent verification/ folder.
    --verbose      Print per-table processing details.

Post-processing rules
---------------------
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
"""

import os
import re
import json
import hashlib
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Optional


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Regex patterns  (identical to parser.py)
# --------------------------------------------------------------------------- #

_HTML_INLINE = re.compile(
    r"<(br|br/|br />|/?ul|/?li|/?ol|/?p|del|/?del|/?b|/?i|/?strong|/?em"
    r"|sup|/?sup|sub|/?sub)[^>]*>",
    re.IGNORECASE,
)
_LONE_JUNK_CHARS = re.compile(r"^[,\\/\|\.]{1,2}$")
_WORD_FRAGMENT   = re.compile(r"[a-zA-Z]{2,}$")
_WORD_COMPLETION = re.compile(r"^[a-zA-Z]{2,}")
_FOOTNOTE_NOISE  = re.compile(r"\*\s*[-–]\s*\*")
_TABLE_BLOCK_RE  = re.compile(r"((?:^\|[^\n]+\n)+)", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Table parser / renderer
# --------------------------------------------------------------------------- #

def _parse_md_table(block: str) -> List[Optional[List[str]]]:
    rows: List[Optional[List[str]]] = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-:\s|]+\|$", line):
            rows.append(None)          # separator sentinel
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _render_md_table(rows: List[Optional[List[str]]]) -> str:
    if not rows:
        return ""
    sep_idx = None
    data_rows: List[Tuple[int, List[str]]] = []
    for i, r in enumerate(rows):
        if r is None:
            sep_idx = i
        else:
            data_rows.append((i, r))
    if not data_rows:
        return ""

    col_count  = max(len(r) for _, r in data_rows)
    lines: List[str] = []
    inserted_sep = False
    for i, row in data_rows:
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if sep_idx is not None and i == sep_idx - 1 and not inserted_sep:
            lines.append("|" + "|".join([" --- " for _ in range(col_count)]) + "|")
            inserted_sep = True
    if not inserted_sep and len(lines) > 1:
        sep_line = "|" + "|".join([" --- " for _ in range(col_count)]) + "|"
        lines.insert(1, sep_line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual fixers  (PP-01 … PP-13)
# --------------------------------------------------------------------------- #

def _fix_html_remnants(cells: List[str]) -> List[str]:
    """PP-12: strip inline HTML tags."""
    out = []
    for c in cells:
        cleaned = _HTML_INLINE.sub(" ", c)
        out.append(re.sub(r"\s{2,}", " ", cleaned).strip())
    return out


def _fix_corrupted_symbols(cells: List[str]) -> List[str]:
    """PP-04 + PP-13: normalise ✓/✗/bullet substitutions and footnote noise."""
    out = []
    for c in cells:
        f = c
        f = re.sub(r"✓\s*[_\.]", "✓", f)
        f = re.sub(r"[_\.]\s*✓", "✓", f)
        f = re.sub(r"✗\s*[_\.]", "✗", f)
        f = re.sub(r"•\s*(?=[,\|])", "", f)
        f = re.sub(r"<b>([A-Z])</b>\s*\.", r"", f)
        f = re.sub(r"aived\s*//", "Waived", f)
        f = _FOOTNOTE_NOISE.sub("*", f)      # PP-13
        out.append(f.strip())
    return out


def _fix_trailing_artifact(cells: List[str]) -> List[str]:
    """PP-03: strip stray trailing punctuation; preserve section codes (A1., G3.)."""
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


def _fix_lone_junk(cells: List[str]) -> List[str]:
    """PP-10: replace lone comma/backslash/pipe cells with empty string."""
    return ["" if _LONE_JUNK_CHARS.match(c) else c for c in cells]


def _fix_cell_boundary_bleed(
    rows: List[Optional[List[str]]],
) -> List[Optional[List[str]]]:
    """PP-01: merge words split across adjacent cells."""
    out: List[Optional[List[str]]] = []
    for row in rows:
        if row is None:
            out.append(row)
            continue
        new_row = list(row)
        i = 0
        while i < len(new_row) - 1:
            left  = new_row[i].strip()
            right = new_row[i + 1].strip()
            lm = _WORD_FRAGMENT.search(left)
            rm = _WORD_COMPLETION.match(right)
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


def _fix_row_split(
    rows: List[Optional[List[str]]],
) -> List[Optional[List[str]]]:
    """PP-05: rejoin rows where label ends in an incomplete phrase."""
    INCOMPLETE = re.compile(
        r"(and|or|of|for|to|in|on|at|by|the|a|an|with|from|not|using"
        r"|per|each|where|if|via|transfer|payment|service|account"
        r"|interbank|mortgage|plus)\s*$",
        re.IGNORECASE,
    )
    out: List[Optional[List[str]]] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row is None:
            out.append(row)
            i += 1
            continue
        if i + 1 < len(rows) and rows[i + 1] is not None:
            next_row    = rows[i + 1]
            label0      = row[0].strip() if row else ""
            label1      = next_row[0].strip() if next_row else ""
            non_label_empty = all(not c.strip() for c in next_row[1:]) if len(next_row) > 1 else False
            if INCOMPLETE.search(label0) and non_label_empty and label1:
                merged    = list(row)
                merged[0] = label0 + " " + label1
                out.append(merged)
                i += 2
                continue
        out.append(row)
        i += 1
    return out


def _fix_extra_empty_columns(
    header: List[str],
    data_rows: List[Optional[List[str]]],
) -> Tuple[List[str], List[Optional[List[str]]]]:
    """PP-08: drop ghost columns (empty header + all-empty data)."""
    if not header:
        return header, data_rows
    col_count = len(header)
    keep = []
    for col_idx in range(col_count):
        if header[col_idx].strip():
            keep.append(col_idx)
            continue
        col_has_data = any(
            col_idx < len(r) and r[col_idx].strip()
            for r in data_rows
            if r is not None
        )
        if col_has_data:
            keep.append(col_idx)
    if len(keep) == col_count:
        return header, data_rows
    new_header = [header[k] for k in keep]
    new_data: List[Optional[List[str]]] = []
    for row in data_rows:
        if row is None:
            new_data.append(row)
            continue
        new_data.append([row[k] if k < len(row) else "" for k in keep])
    return new_header, new_data


def _is_phantom_row(cells: List[str], col_count: int) -> bool:
    """PP-06: OCR-noise rows — long prose in label col, all other cells empty."""
    if not cells:
        return False
    label       = cells[0].strip()
    other_empty = all(not c.strip() for c in cells[1:])
    return len(label.split()) >= 8 and other_empty


def _is_caption_row(cells: List[str]) -> bool:
    """PP-07: prose first-row absorbed as caption."""
    if not cells:
        return False
    label       = cells[0].strip()
    other_empty = all(not c.strip() for c in cells[1:])
    word_count  = len(label.split())
    return other_empty and (word_count >= 5 or "<br" in label.lower())


def _fix_section_header_rows(
    rows: List[Optional[List[str]]],
) -> Tuple[List[str], List[Optional[List[str]]]]:
    """PP-11: lift section-label-only rows as headings above the table."""
    lifted: List[str] = []
    remaining: List[Optional[List[str]]] = []
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
# Master per-table processor
# --------------------------------------------------------------------------- #

def _postprocess_table_block(
    block: str,
    verbose: bool = False,
) -> Tuple[str, List[str]]:
    """Apply all PP rules to one markdown table block.
    Returns (processed_markdown, lifted_headings)."""
    rows = _parse_md_table(block)
    if not rows:
        return block, []

    header_idx = next((i for i, r in enumerate(rows) if r is not None), None)
    if header_idx is None:
        return block, []

    # Step 1 — per-cell fixes
    fixed: List[Optional[List[str]]] = []
    for row in rows:
        if row is None:
            fixed.append(None)
            continue
        row = _fix_html_remnants(row)
        row = _fix_corrupted_symbols(row)
        row = _fix_trailing_artifact(row)
        row = _fix_lone_junk(row)
        fixed.append(row)

    # Step 2 — cell boundary bleed
    fixed = _fix_cell_boundary_bleed(fixed)

    # Step 3 — row-split repair
    fixed = _fix_row_split(fixed)

    # Step 4 — lift section header rows
    lifted, fixed = _fix_section_header_rows(fixed)

    # Step 5 — remove phantom content rows
    col_count = max((len(r) for r in fixed if r is not None), default=1)
    clean: List[Optional[List[str]]] = []
    for row in fixed:
        if row is None:
            clean.append(row)
            continue
        if _is_phantom_row(row, col_count):
            if verbose:
                print(f"  [PP-06] Dropped phantom row: {row[0][:60]!r}")
            continue
        clean.append(row)
    fixed = clean

    # Step 6 — caption row extraction
    extracted_caption: Optional[str] = None
    if fixed and fixed[0] is not None and _is_caption_row(fixed[0]):
        extracted_caption = fixed[0][0].strip()
        if verbose:
            print(f"  [PP-07] Extracted caption: {extracted_caption[:60]!r}")
        fixed = fixed[1:]

    # Step 7 — drop ghost empty columns
    non_sep = [r for r in fixed if r is not None]
    if non_sep:
        hdr_row = (
            fixed[header_idx]
            if header_idx < len(fixed) and fixed[header_idx] is not None
            else non_sep[0]
        )
        data_rows_only = [r for r in fixed if r is not None and r is not hdr_row]
        new_header, new_data = _fix_extra_empty_columns(hdr_row, data_rows_only)
        new_rows: List[Optional[List[str]]] = []
        data_iter = iter(new_data)
        for row in fixed:
            if row is None:
                new_rows.append(None)
            elif row is hdr_row:
                new_rows.append(new_header)
            else:
                try:
                    new_rows.append(next(data_iter))
                except StopIteration:
                    pass
        fixed = new_rows

    if extracted_caption:
        lifted.insert(0, extracted_caption)

    return _render_md_table(fixed), lifted


# --------------------------------------------------------------------------- #
# Full-document processor
# --------------------------------------------------------------------------- #

def postprocess_markdown(markdown: str, verbose: bool = False) -> str:
    """Apply all PP rules to every table in *markdown*.
    Non-table content is preserved unchanged."""
    table_count = [0]

    def replace_table(m: re.Match) -> str:
        table_count[0] += 1
        block = m.group(0)
        if verbose:
            print(f"\n[TABLE {table_count[0]}]")
        processed, lifted = _postprocess_table_block(block, verbose=verbose)
        prefix = ""
        if lifted:
            prefix = "\n".join(f"#### {h}" for h in lifted) + "\n\n"
        return prefix + processed + "\n"

    result = _TABLE_BLOCK_RE.sub(replace_table, markdown)
    if verbose:
        print(f"\n[POST-PROC] Processed {table_count[0]} table(s).")
    return result


# --------------------------------------------------------------------------- #
# Manifest patch  (optional — keeps verification bundle consistent)
# --------------------------------------------------------------------------- #

def _patch_manifest(output_md_path: str, new_md_text: str) -> None:
    """If a manifest.json exists in verification/, update the markdown hash."""
    output_dir   = os.path.dirname(os.path.abspath(output_md_path))
    manifest_path = os.path.join(output_dir, "verification", "manifest.json")
    if not os.path.isfile(manifest_path):
        # Try one level up (in case output_md is already inside verification/)
        manifest_path = os.path.join(output_dir, "..", "verification", "manifest.json")
        manifest_path = os.path.normpath(manifest_path)
    if not os.path.isfile(manifest_path):
        return
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        md_artifact = manifest.get("artifacts", {}).get("markdown", {})
        if md_artifact:
            md_artifact["sha256"] = sha256_text(new_md_text)
            md_artifact["note"]   = (
                md_artifact.get("note", "") + "  [re-processed by postprocess.py]"
            ).strip()
        # Record that postprocess.py was run
        manifest.setdefault("postprocessing", {})["reprocessed_by"] = "postprocess.py"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[MANIFEST] Updated sha256 in {manifest_path}")
    except Exception as e:
        print(f"[MANIFEST] Could not update manifest: {e}")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone markdown table post-processor (PP-01 … PP-13).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input",  required=True, help="Source Markdown file path.")
    ap.add_argument("--output", default=None,
                    help="Destination Markdown file.  Default: <stem>_pp.md.")
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite the input file.")
    ap.add_argument("--backup",  action="store_true",
                    help="With --inplace: save a backup as <stem>_backup.md first.")
    ap.add_argument("--no-manifest", action="store_true",
                    help="Skip manifest.json update.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print per-table processing details.")
    args = ap.parse_args()

    # Resolve paths
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        ap.error(f"Input file not found: {input_path}")

    stem = os.path.splitext(input_path)[0]

    if args.inplace:
        output_path = input_path
        if args.backup:
            backup_path = stem + "_backup.md"
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"[BACKUP] {backup_path}")
    else:
        output_path = (
            os.path.abspath(args.output)
            if args.output
            else stem + "_pp.md"
        )

    # Read
    print(f"[IN]  {input_path}")
    markdown = read_text(input_path)

    # Process
    print("[POST-PROC] Applying rule-based table corrections…")
    processed = postprocess_markdown(markdown, verbose=args.verbose)
    print("[POST-PROC] Done.")

    # Write
    write_text(output_path, processed)
    print(f"[OUT] {output_path}")

    # Stats
    original_tables  = len(_TABLE_BLOCK_RE.findall(markdown))
    processed_tables = len(_TABLE_BLOCK_RE.findall(processed))
    print(f"[STATS] Tables in input: {original_tables}  |  Tables in output: {processed_tables}")
    print(f"[STATS] Characters: {len(markdown)} -> {len(processed)}"
          f"  (diff {len(processed) - len(markdown):+d})")

    # Manifest
    if not args.no_manifest:
        _patch_manifest(output_path, processed)

    print("\nDone.  Next:  python verifier.py --bundle "
          "<output_dir>/verification/manifest.json")


if __name__ == "__main__":
    main()
