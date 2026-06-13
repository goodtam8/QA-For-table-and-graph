"""
verifier.py — Confidence-scoring verifier for PDF->Markdown conversions.

Reads ONLY the verification bundle produced by parser.py (manifest.json and
the artifacts it points to).  It does not import Marker and does not need the
PDF for the core checks — everything structural travels in the bundle.  The
PDF is touched only for the optional independent table re-check / stability.

What it does
------------
Scores each document on five dimensions and, critically, reports WHICH
sections of the markdown are inaccurate vs. the corresponding PDF region
(per page), then routes the document:

    PASS            high overall score, no critical rule failed
    RETRY_OCR_LLM   text/table weak -> re-run with --force_ocr / --use_llm
    MANUAL_REVIEW   low overall score or a critical rule failed

Dimensions (0..1, weighted into overall 0..100):
    text         normalized token recall: PDF text vs markdown text (per page)
    structure    overlap of block-type counts (headings/tables/lists/...) vs
                 a structural expectation derived from the PDF text
    table        per-page table presence + row/col plausibility vs PDF tables
    stability    diff between two markdown runs (if stability bundle present)
    cleanliness  penalties: OCR garbage, repeated headers/footers, empty
                 sections, broken/blank links, replacement chars

Failure severities:
    critical  -> forces MANUAL_REVIEW   (e.g. whole page missing, table dropped)
    major     -> strong score penalty
    minor     -> small score penalty / informational

Usage
-----
    python verifier.py --bundle ./hsbc_output/verification/manifest.json
    python verifier.py --bundle .../manifest.json --report ./report.json
    # thresholds tunable for RAG vs archival/legal use:
    python verifier.py --bundle ... --min-overall 85 --min-table 70
"""

import os
import re
import json
import argparse
import difflib
from collections import Counter


# --------------------------------------------------------------------------- #
# Tokenisation / normalisation
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[0-9a-z\u00c0-\uffff]+", re.IGNORECASE)


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokens(text: str) -> Counter:
    return Counter(_WORD_RE.findall(normalize(text)))


def strip_markdown(md: str) -> str:
    """Remove markdown syntax so we compare on visible text only."""
    md = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)        # code fences
    md = re.sub(r"`[^`]*`", " ", md)                            # inline code
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)               # images
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)            # links -> text
    md = re.sub(r"<[^>]+>", " ", md)                            # html tags
    md = re.sub(r"[#>*_~|`-]", " ", md)                         # md punctuation
    return md


def recall(reference: str, candidate: str) -> float:
    """Token recall: fraction of reference tokens present in candidate."""
    ref, cand = tokens(reference), tokens(candidate)
    if not ref:
        return 1.0
    covered = sum(min(c, cand.get(t, 0)) for t, c in ref.items())
    return covered / max(1, sum(ref.values()))


# --------------------------------------------------------------------------- #
# Bundle loading
# --------------------------------------------------------------------------- #
def load_bundle(manifest_path: str) -> dict:
    base = os.path.dirname(os.path.abspath(manifest_path))
    out_dir = os.path.dirname(base)  # verification/ -> output dir
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    def rel(p):
        return os.path.normpath(os.path.join(out_dir, p))

    arts = manifest["artifacts"]

    def load_json(key):
        a = arts.get(key)
        if not a:
            return None
        with open(rel(a["path"]), encoding="utf-8") as f:
            return json.load(f)

    def load_text(key):
        a = arts.get(key)
        if not a:
            return None
        with open(rel(a["path"]), encoding="utf-8") as f:
            return f.read()

    return {
        "manifest": manifest,
        "out_dir": out_dir,
        "markdown": load_text("markdown"),
        "marker_metadata": load_json("marker_metadata"),
        "md_block_counts": load_json("md_block_counts"),
        "pdf_text_by_page": load_json("pdf_text_by_page"),
        "pdf_tables_by_page": load_json("pdf_tables_by_page"),
        "stability_run2": load_text("stability_run2"),
    }


# --------------------------------------------------------------------------- #
# Map markdown to pages (using paginate markers if present, else heuristics)
# --------------------------------------------------------------------------- #
_PAGE_MARKER = re.compile(r"\n\{(\d+)\}-{10,}\n")  # Marker --paginate_output style


def split_markdown_by_page(md: str, n_pages: int) -> list:
    """
    Best-effort split of the markdown into n_pages chunks so we can do
    per-page recall.  If pagination markers exist, use them; otherwise split
    proportionally by length (recall is robust to coarse alignment).
    """
    if not md:
        return ["" for _ in range(n_pages)]
    parts = _PAGE_MARKER.split(md)
    if len(parts) > 1:
        # parts = [pre, pageno, chunk, pageno, chunk, ...]
        chunks = []
        i = 1
        while i < len(parts) - 1:
            chunks.append(parts[i + 1])
            i += 2
        if parts[0].strip():
            chunks.insert(0, parts[0])
        # pad/truncate to n_pages
        if len(chunks) >= n_pages:
            return chunks[:n_pages]
        return chunks + [""] * (n_pages - len(chunks))
    # No markers: proportional split.
    if n_pages <= 1:
        return [md]
    L = len(md)
    step = L / n_pages
    return [md[int(i * step):int((i + 1) * step)] for i in range(n_pages)]


# --------------------------------------------------------------------------- #
# Markdown table parsing (for table fidelity)
# --------------------------------------------------------------------------- #
def parse_md_tables(md: str) -> list:
    """Return [{rows, cols}] for every markdown pipe-table found."""
    tables = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            block = [lines[i]]
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            header = [c for c in lines[i].split("|") if c.strip() != ""]
            tables.append({"rows": len(block) - 1, "cols": len(header)})  # minus separator
            i = j
        else:
            i += 1
    # also count HTML tables
    for m in re.finditer(r"<table.*?</table>", md, flags=re.DOTALL | re.IGNORECASE):
        rows = len(re.findall(r"<tr", m.group(0), flags=re.IGNORECASE))
        cols = max((len(re.findall(r"<t[dh]", r, flags=re.IGNORECASE))
                    for r in re.split(r"</tr>", m.group(0), flags=re.IGNORECASE)), default=0)
        tables.append({"rows": rows, "cols": cols})
    return tables


# --------------------------------------------------------------------------- #
# Cleanliness signals
# --------------------------------------------------------------------------- #
def cleanliness_signals(md: str) -> dict:
    issues = []
    score = 1.0

    # replacement / control chars => OCR garbage
    repl = md.count("\ufffd")
    if repl:
        issues.append({"severity": "major", "type": "ocr_replacement_chars",
                       "detail": f"{repl} U+FFFD replacement characters"})
        score -= min(0.3, repl * 0.01)

    # long runs of non-word gibberish
    gib = len(re.findall(r"[^\w\s]{8,}", md))
    if gib > 5:
        issues.append({"severity": "minor", "type": "gibberish_runs",
                       "detail": f"{gib} long punctuation/symbol runs"})
        score -= min(0.1, gib * 0.005)

    # repeated header/footer lines (same short line many times)
    line_counts = Counter(l.strip() for l in md.splitlines() if 3 < len(l.strip()) < 80)
    repeated = [(l, c) for l, c in line_counts.items() if c >= 5]
    if repeated:
        issues.append({"severity": "minor", "type": "repeated_lines",
                       "detail": f"{len(repeated)} lines repeated >=5x (likely running header/footer)",
                       "examples": [l for l, _ in repeated[:3]]})
        score -= min(0.1, len(repeated) * 0.01)

    # empty sections: heading immediately followed by another heading / nothing
    headings = [(m.start(), m.group(0)) for m in re.finditer(r"^#{1,6}\s.*$", md, flags=re.MULTILINE)]
    empty = 0
    for idx, (pos, h) in enumerate(headings):
        nxt = headings[idx + 1][0] if idx + 1 < len(headings) else len(md)
        body = strip_markdown(md[pos + len(h):nxt]).strip()
        if len(body) < 3:
            empty += 1
    if empty:
        issues.append({"severity": "minor", "type": "empty_sections",
                       "detail": f"{empty} headings with empty bodies"})
        score -= min(0.15, empty * 0.02)

    # blank / broken links
    blank_links = len(re.findall(r"\]\(\s*\)", md)) + len(re.findall(r"\]\(#\)", md))
    if blank_links:
        issues.append({"severity": "minor", "type": "blank_links",
                       "detail": f"{blank_links} links with empty/anchor-only targets"})
        score -= min(0.1, blank_links * 0.01)

    return {"score": max(0.0, score), "issues": issues}


# --------------------------------------------------------------------------- #
# Structure expectation derived from PDF text (independent of Marker)
# --------------------------------------------------------------------------- #
def expected_structure_from_pdf(pdf_pages: list) -> dict:
    """Cheap heuristics on raw PDF text to know what SHOULD exist."""
    full = "\n".join(pdf_pages)
    # bullet/numbered list lines
    list_lines = len(re.findall(r"^\s*(?:[-*•]|\d+[.)\t])\s+", full, flags=re.MULTILINE))
    # heading-ish lines: ALLCAPS or "A.", "C2." style or Title Case short lines
    heading_lines = len(re.findall(r"^\s*[A-Z][A-Z0-9 ./&'-]{4,}\s*$", full, flags=re.MULTILINE))
    heading_lines += len(re.findall(r"^\s*[A-G]\d?\.\s", full, flags=re.MULTILINE))
    return {"expected_list_lines": list_lines, "expected_headings": heading_lines}


# --------------------------------------------------------------------------- #
# Core verification
# --------------------------------------------------------------------------- #
def verify(bundle: dict, cfg: dict) -> dict:
    md = bundle["markdown"] or ""
    md_text = strip_markdown(md)
    meta = bundle["marker_metadata"] or {}
    block_counts = bundle["md_block_counts"] or {"total": {}, "per_page": {}}
    pdf_text = bundle["pdf_text_by_page"] or {"pages": []}
    pdf_tables = bundle["pdf_tables_by_page"] or {"pages": []}

    pdf_pages = pdf_text.get("pages", [])
    n_pages_pdf = len(pdf_pages)
    n_pages_meta = len(meta.get("page_stats", []) or [])

    findings = []          # all severity-tagged issues
    section_flags = []     # per-page "which section is inaccurate"

    # ---- 1) TEXT RECALL (overall + per page) ------------------------------ #
    overall_text_recall = recall("\n".join(pdf_pages), md_text)
    md_pages = split_markdown_by_page(md, max(n_pages_pdf, 1))
    page_recalls = []
    for i, pg in enumerate(pdf_pages):
        r = recall(pg, md_pages[i] if i < len(md_pages) else "")
        page_recalls.append(round(r, 4))
        if r < cfg["page_recall_critical"] and len(tokens(pg)) > 20:
            sev = "critical"
        elif r < cfg["page_recall_major"] and len(tokens(pg)) > 20:
            sev = "major"
        else:
            sev = None
        if sev:
            # snippet of likely-missing content: tokens in pdf, absent in md page
            cand = tokens(md_pages[i] if i < len(md_pages) else "")
            missing = [t for t in tokens(pg) if cand.get(t, 0) == 0]
            section_flags.append({
                "page": i + 1, "dimension": "text", "severity": sev,
                "recall": round(r, 3),
                "detail": f"Page {i+1} text recall {r:.0%} is below threshold.",
                "sample_missing_terms": missing[:15],
            })
            findings.append(section_flags[-1])
    text_score = overall_text_recall

    # ---- 2) STRUCTURE MATCH ---------------------------------------------- #
    totals = block_counts.get("total", {})
    md_tables_parsed = parse_md_tables(md)
    exp = expected_structure_from_pdf(pdf_pages)

    n_headings = totals.get("SectionHeader", 0)
    n_lists = totals.get("ListItem", 0) + totals.get("ListGroup", 0)
    n_tables_json = totals.get("Table", 0) + totals.get("TableGroup", 0)
    n_figs = totals.get("Figure", 0) + totals.get("Picture", 0) + totals.get("FigureGroup", 0)

    def ratio(actual, expected):
        if expected <= 0:
            return 1.0
        return min(1.0, actual / expected)

    head_ratio = ratio(n_headings, exp["expected_headings"])
    list_ratio = ratio(n_lists, exp["expected_list_lines"])
    structure_score = round(0.5 * head_ratio + 0.5 * list_ratio, 4)
    if head_ratio < 0.4 and exp["expected_headings"] > 5:
        findings.append({"dimension": "structure", "severity": "major",
                         "detail": f"Only {n_headings} headings detected vs ~{exp['expected_headings']} expected from PDF."})
    if list_ratio < 0.3 and exp["expected_list_lines"] > 10:
        findings.append({"dimension": "structure", "severity": "major",
                         "detail": f"Only {n_lists} list items vs ~{exp['expected_list_lines']} list-like lines in PDF (bullets likely flattened)."})

    # ---- 3) TABLE FIDELITY (per page) ------------------------------------ #
    pdf_tbl_pages = pdf_tables.get("pages", [])
    total_pdf_tables = sum(len(p) for p in pdf_tbl_pages)
    total_md_tables = len(md_tables_parsed) or n_tables_json
    per_page_counts = block_counts.get("per_page", {})

    table_findings = []
    if total_pdf_tables == 0:
        table_score = 1.0  # nothing to verify
    else:
        # presence: did we produce roughly as many tables as the PDF has?
        presence = min(1.0, total_md_tables / max(1, total_pdf_tables))
        # row plausibility: compare total rows
        pdf_rows = sum(t.get("rows", 0) for p in pdf_tbl_pages for t in p if isinstance(t, dict))
        md_rows = sum(t["rows"] for t in md_tables_parsed)
        row_ratio = ratio(md_rows, pdf_rows) if pdf_rows else 1.0
        table_score = round(0.6 * presence + 0.4 * row_ratio, 4)

        if total_md_tables < total_pdf_tables:
            sev = "critical" if total_md_tables == 0 else "major"
            table_findings.append({
                "dimension": "table", "severity": sev,
                "detail": f"Found {total_md_tables} table(s) in markdown but PDF has ~{total_pdf_tables}. "
                          f"{total_pdf_tables - total_md_tables} table(s) likely dropped or merged.",
            })
        if pdf_rows and md_rows < 0.7 * pdf_rows:
            table_findings.append({
                "dimension": "table", "severity": "major",
                "detail": f"Markdown tables have ~{md_rows} rows vs ~{pdf_rows} in PDF (rows likely lost).",
            })
        # per-page: which page lost its table
        for i, p in enumerate(pdf_tbl_pages):
            pdf_n = len([t for t in p if isinstance(t, dict) and "rows" in t])
            md_n = per_page_counts.get(str(i), {}).get("Table", 0) + \
                   per_page_counts.get(str(i), {}).get("TableGroup", 0)
            if pdf_n > 0 and md_n == 0:
                table_findings.append({
                    "page": i + 1, "dimension": "table", "severity": "critical",
                    "detail": f"Page {i+1}: PDF has {pdf_n} table(s) but markdown has none on this page.",
                })
        findings.extend(table_findings)
        section_flags.extend([f for f in table_findings if "page" in f])

    if total_pdf_tables and table_score < cfg["min_table"] / 100.0:
        findings.append({"dimension": "table", "severity": "major",
                         "detail": f"Table score {table_score:.0%} below minimum for table-heavy doc."})

    # ---- 4) PAGE CONSISTENCY --------------------------------------------- #
    page_consistency_ok = True
    if n_pages_meta and n_pages_pdf and n_pages_meta != n_pages_pdf:
        page_consistency_ok = False
        findings.append({"dimension": "page", "severity": "critical",
                         "detail": f"Marker reports {n_pages_meta} pages but PDF text extraction found {n_pages_pdf}."})
    # whole-page dropout already captured via per-page recall == 0
    for i, r in enumerate(page_recalls):
        if r == 0.0 and len(tokens(pdf_pages[i])) > 20:
            findings.append({"page": i + 1, "dimension": "page", "severity": "critical",
                             "detail": f"Page {i+1} appears entirely missing from markdown (0% recall)."})

    # ---- 5) STABILITY ---------------------------------------------------- #
    if bundle["stability_run2"] is not None:
        a = strip_markdown(md)
        b = strip_markdown(bundle["stability_run2"])
        sm = difflib.SequenceMatcher(None, a, b)
        stability_score = round(sm.ratio(), 4)
        if stability_score < 0.97:
            findings.append({"dimension": "stability", "severity": "minor",
                             "detail": f"Two conversion runs differ ({stability_score:.1%} similar) — nondeterministic output."})
    else:
        stability_score = None  # not measured

    # ---- 6) CLEANLINESS -------------------------------------------------- #
    clean = cleanliness_signals(md)
    cleanliness_score = clean["score"]
    findings.extend(clean["issues"])

    # ---- AGGREGATE OVERALL SCORE ----------------------------------------- #
    weights = cfg["weights"]
    components = {
        "text": text_score,
        "structure": structure_score,
        "table": table_score,
        "cleanliness": cleanliness_score,
    }
    if stability_score is not None:
        components["stability"] = stability_score
    # renormalise weights over present components
    used_w = {k: weights[k] for k in components}
    wsum = sum(used_w.values())
    overall = sum(components[k] * used_w[k] for k in components) / wsum * 100.0

    n_critical = sum(1 for f in findings if f.get("severity") == "critical")
    n_major = sum(1 for f in findings if f.get("severity") == "major")
    n_minor = sum(1 for f in findings if f.get("severity") == "minor")

    # ---- ROUTING --------------------------------------------------------- #
    if n_critical > 0 or overall < cfg["manual_review_below"]:
        route = "MANUAL_REVIEW"
    elif (text_score < cfg["retry_text_below"] or
          (total_pdf_tables and table_score < cfg["retry_table_below"])):
        route = "RETRY_OCR_LLM"
    elif overall >= cfg["min_overall"] and n_major == 0:
        route = "PASS"
    else:
        route = "MANUAL_REVIEW"

    return {
        "document": bundle["manifest"]["source_pdf"]["base_name"],
        "overall_score": round(overall, 1),
        "route": route,
        "scores": {k: round(v * 100, 1) for k, v in components.items()},
        "page_count": {"pdf": n_pages_pdf, "marker_meta": n_pages_meta,
                       "consistent": page_consistency_ok},
        "tables": {"pdf": total_pdf_tables, "markdown": total_md_tables},
        "severity_counts": {"critical": n_critical, "major": n_major, "minor": n_minor},
        "per_page_text_recall": page_recalls,
        "inaccurate_sections": section_flags,   # <-- WHICH sections are off
        "findings": findings,
        "thresholds": {k: cfg[k] for k in
                       ("min_overall", "min_table", "manual_review_below",
                        "retry_text_below", "retry_table_below")},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_CFG = {
    "weights": {"text": 0.40, "structure": 0.20, "table": 0.25,
                "cleanliness": 0.10, "stability": 0.05},
    "min_overall": 85.0,          # overall pass threshold
    "min_table": 70.0,            # min table score for table-heavy docs
    "manual_review_below": 70.0,  # below this -> always manual review
    "retry_text_below": 0.80,     # weak text -> retry with OCR/LLM
    "retry_table_below": 0.50,    # weak tables -> retry
    "page_recall_critical": 0.35, # per-page recall below -> critical
    "page_recall_major": 0.65,    # per-page recall below -> major
}


def main():
    ap = argparse.ArgumentParser(description="Verify a PDF->Markdown conversion bundle.")
    ap.add_argument("--bundle", required=True, help="Path to verification/manifest.json")
    ap.add_argument("--report", default=None, help="Where to write JSON report")
    ap.add_argument("--min-overall", type=float, default=DEFAULT_CFG["min_overall"])
    ap.add_argument("--min-table", type=float, default=DEFAULT_CFG["min_table"])
    ap.add_argument("--manual-below", type=float, default=DEFAULT_CFG["manual_review_below"])
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    cfg["min_overall"] = args.min_overall
    cfg["min_table"] = args.min_table
    cfg["manual_review_below"] = args.manual_below

    bundle = load_bundle(args.bundle)
    report = verify(bundle, cfg)

    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(args.bundle)), "verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- human-readable summary ---- #
    print(f"Document : {report['document']}")
    print(f"Overall  : {report['overall_score']}/100   ->  ROUTE: {report['route']}")
    print(f"Scores   : {report['scores']}")
    print(f"Pages    : PDF={report['page_count']['pdf']} "
          f"Marker={report['page_count']['marker_meta']} "
          f"consistent={report['page_count']['consistent']}")
    print(f"Tables   : PDF={report['tables']['pdf']} MD={report['tables']['markdown']}")
    print(f"Severity : {report['severity_counts']}")
    if report["inaccurate_sections"]:
        print("\nInaccurate sections (route to review):")
        for s in report["inaccurate_sections"]:
            loc = f"p{s.get('page')}" if s.get("page") else "-"
            print(f"  [{s['severity'].upper():8}] {loc:4} {s['dimension']:9} {s['detail']}")
    print(f"\nFull report -> {report_path}")


if __name__ == "__main__":
    main()
