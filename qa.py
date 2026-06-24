#!/usr/bin/env python3
"""
qa.py — QA system for PDF-to-Markdown converted documents.
Now reads VERIFIER annotations in the annotated .md file and uses the
four-level verdict (CORRECT | PARTIAL_CORRECT | PARTIAL_INCORRECT | INCORRECT)
to qualify answers and warn the user when relevant tables have errors.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE             = Path(__file__).parent
OUTPUT_DIR       = BASE / "hsbc_output"
# Prefer the annotated version (with VERIFIER tags); fall back to raw markdown
ANNOTATED_MD     = OUTPUT_DIR / "annotated_hsbc.md"
CONTEXT_PATH     = OUTPUT_DIR / "hsbc.md"
REPORT_PATH      = OUTPUT_DIR / "verification" / "verification_report.json"
PAGE_IMAGES_DIR  = OUTPUT_DIR / "verification" / "page_images"
PDF_TEXT_PATH    = OUTPUT_DIR / "verification" / "pdf_text_by_page.json"


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFIER ANNOTATION PARSER
# ═══════════════════════════════════════════════════════════════════════════════

# Matches: <!-- VERIFIER: status=PARTIAL_CORRECT | confidence=0.83 | correct_ratio=0.85 | errors=WRONG_VALUE -->
_VERIFIER_TAG_RE = re.compile(
    r"""<!-- VERIFIER:\s*
        status=(?P<status>[A-Z_]+)\s*\|\s*
        confidence=(?P<confidence>[\d\.]+)\s*\|\s*
        correct_ratio=(?P<correct_ratio>[\d\.]+)\s*\|\s*
        errors=(?P<errors>[^\-]*?)
        \s*-->""",
    re.VERBOSE,
)
# Matches: <!-- /VERIFIER -->
_VERIFIER_CLOSE_RE = re.compile(r"<!--\s*/VERIFIER\s*-->")
# Matches: <!-- PARTIAL_DETAIL: correct_rows=4 incorrect_rows=2 ratio=0.67 | note -->
_PARTIAL_DETAIL_RE = re.compile(
    r"<!-- PARTIAL_DETAIL:\s*"
    r"correct_rows=(?P<correct_rows>\d+)\s+"
    r"incorrect_rows=(?P<incorrect_rows>\d+)\s+"
    r"ratio=(?P<ratio>[\d\.]+)"
    r"(?:\s*\|\s*(?P<note>[^-]*?))?\s*-->",
)
# Matches: <!-- PDF_REF: page=3 | section="..." | pdf_table_index=2 -->
_PDF_REF_RE = re.compile(
    r"<!-- PDF_REF:\s*page=(?P<page>[\w]+)\s*\|\s*"
    r'section="(?P<section>[^"]*)"\s*\|\s*'
    r"pdf_table_index=(?P<pdf_table_index>[\w]+)\s*-->",
)

# Verdict ordering (1=best, 4=worst)
_VERDICT_RANK = {
    "CORRECT": 1,
    "PARTIAL_CORRECT": 2,
    "PARTIAL_INCORRECT": 3,
    "INCORRECT": 4,
}

_VERDICT_ICON = {
    "CORRECT": "✓",
    "PARTIAL_CORRECT": "◑",
    "PARTIAL_INCORRECT": "◐",
    "INCORRECT": "✗",
}


def parse_annotated_blocks(md_text: str) -> list[dict]:
    """
    Extract all VERIFIER-annotated blocks from the markdown file.
    Returns a list of dicts with keys:
        status, confidence, correct_ratio, errors (list[str]),
        partial_detail (dict|None), pdf_ref (dict|None),
        content (the raw table text between open and close tags)
    """
    blocks = []
    pos = 0
    while True:
        m_open = _VERIFIER_TAG_RE.search(md_text, pos)
        if not m_open:
            break
        m_close = _VERIFIER_CLOSE_RE.search(md_text, m_open.end())
        if not m_close:
            break

        # Everything between the VERIFIER open and /VERIFIER close
        content = md_text[m_open.end():m_close.start()]

        # Optional PARTIAL_DETAIL after /VERIFIER
        pd_match = _PARTIAL_DETAIL_RE.search(md_text, m_close.end(), m_close.end() + 400)
        partial_detail = None
        if pd_match:
            partial_detail = {
                "correct_rows": int(pd_match.group("correct_rows")),
                "incorrect_rows": int(pd_match.group("incorrect_rows")),
                "ratio": float(pd_match.group("ratio")),
                "note": (pd_match.group("note") or "").strip(),
            }

        # Optional PDF_REF after the close tag
        pdf_ref_match = _PDF_REF_RE.search(md_text, m_close.end(), m_close.end() + 500)
        pdf_ref = None
        if pdf_ref_match:
            pdf_ref = {
                "page": pdf_ref_match.group("page"),
                "section": pdf_ref_match.group("section"),
                "pdf_table_index": pdf_ref_match.group("pdf_table_index"),
            }

        errors_str = m_open.group("errors").strip()
        blocks.append({
            "status": m_open.group("status"),
            "confidence": float(m_open.group("confidence")),
            "correct_ratio": float(m_open.group("correct_ratio")),
            "errors": [e.strip() for e in errors_str.split(",") if e.strip()],
            "partial_detail": partial_detail,
            "pdf_ref": pdf_ref,
            "content": content.strip(),
        })
        pos = m_close.end()

    return blocks


def strip_verifier_tags(md_text: str) -> str:
    """Return clean markdown text with all VERIFIER comment tags removed."""
    text = _VERIFIER_TAG_RE.sub("", md_text)
    text = _VERIFIER_CLOSE_RE.sub("", text)
    text = _PARTIAL_DETAIL_RE.sub("", text)
    text = _PDF_REF_RE.sub("", text)
    return text


def build_data_quality_note(blocks: list[dict], relevant_pages: list[int]) -> str:
    """
    Build a concise data-quality disclaimer for the QA system prompt,
    focused on tables that may be on the pages relevant to the question.
    """
    if not blocks:
        return ""

    # Filter to blocks whose page overlaps with relevant pages (best effort)
    def _page(b) -> Optional[int]:
        pr = b.get("pdf_ref")
        if pr and pr.get("page", "unknown") != "unknown":
            try:
                return int(pr["page"])
            except ValueError:
                pass
        return None

    relevant_blocks = blocks
    if relevant_pages:
        on_page = [b for b in blocks if _page(b) in relevant_pages]
        if on_page:
            relevant_blocks = on_page

    worst_verdict = max(
        relevant_blocks, key=lambda b: _VERDICT_RANK.get(b["status"], 0)
    )

    # Only mention data quality when there are issues
    if worst_verdict["status"] == "CORRECT":
        return ""

    lines = ["=== DATA QUALITY NOTICE ==="]
    summary_counts = {}
    for b in relevant_blocks:
        summary_counts[b["status"]] = summary_counts.get(b["status"], 0) + 1

    for verdict in ("PARTIAL_CORRECT", "PARTIAL_INCORRECT", "INCORRECT"):
        if summary_counts.get(verdict, 0) > 0:
            icon = _VERDICT_ICON[verdict]
            lines.append(
                f"{icon} {summary_counts[verdict]} table(s) labelled {verdict} "
                f"on the pages relevant to this question."
            )

    lines.append("")
    lines.append(
        "For each PARTIAL_CORRECT or PARTIAL_INCORRECT table: some cell values "
        "may be inaccurate due to PDF-to-Markdown conversion errors. "
        "Cite your answers with caution and flag any figures from these tables. "
        "For INCORRECT tables: do not rely on the data — cross-check against the original PDF."
    )
    lines.append("=== END DATA QUALITY NOTICE ===")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_context() -> tuple[str, list[dict]]:
    """
    Load the best available markdown source.
    Returns (clean_text, annotated_blocks).
    """
    if ANNOTATED_MD.exists():
        raw = ANNOTATED_MD.read_text(encoding="utf-8")
        blocks = parse_annotated_blocks(raw)
        clean  = strip_verifier_tags(raw)
        return clean, blocks

    # Fallback: use raw markdown, no annotation blocks
    return CONTEXT_PATH.read_text(encoding="utf-8"), []


def load_verification_report() -> Optional[dict]:
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_pdf_text_by_page() -> list[str]:
    if not PDF_TEXT_PATH.exists():
        return []
    with open(PDF_TEXT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pages", [])


# ═══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION / FLAGGED PAGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_flagged_pages(report: Optional[dict]) -> set[int]:
    if not report:
        return set()
    flagged: set[int] = set()
    for section in report.get("inaccurate_sections", []):
        page = section.get("page")
        if page is not None:
            flagged.add(int(page))
    for finding in report.get("findings", []):
        page = finding.get("page")
        if page is not None:
            flagged.add(int(page))
    for i, r in enumerate(report.get("per_page_text_recall", []), start=1):
        if isinstance(r, (int, float)) and r < 0.35:
            flagged.add(i)
    return flagged


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-BASED PAGE LOCALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def locate_relevant_pages(question: str, pdf_pages: list[str]) -> list[int]:
    """
    Ask the LLM which PDF page(s) directly contain the data for the question.
    Returns a sorted list of 1-based page numbers.
    """
    if not pdf_pages:
        return []

    total_pages = len(pdf_pages)
    page_dump = "\n\n".join(
        f"[Page {i}]\n{text[:800].strip()}"
        for i, text in enumerate(pdf_pages, start=1)
        if text.strip()
    )

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document navigation assistant. "
                    "You will be given a user question and excerpts of text "
                    "extracted directly from each page of a PDF document. "
                    f"The document has {total_pages} pages total. "
                    "Your task: identify which page number(s) DIRECTLY contain "
                    "the specific data needed to answer the question. "
                    "Only include a page if that page itself contains the answer, "
                    "not pages that merely mention the same topic in passing. "
                    "If the answer spans multiple pages, list all of them. "
                    "Return ONLY a JSON array of integers, e.g. [3] or [5,6]. "
                    "If you cannot identify specific pages, return []."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Page excerpts:\n{page_dump[:12000]}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=80,
    )

    raw = (response.choices[0].message.content or "").strip()
    m = re.search(r"\[([^\]]+)\]", raw)
    if not m:
        return []
    try:
        return sorted({int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()})
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER — incorporates partial-label quality notes
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(
    context: str,
    report: Optional[dict],
    annotation_blocks: list[dict],
    relevant_pages: list[int],
    flagged_pages: set[int],
) -> str:
    """
    Build the full system prompt for the QA LLM call, incorporating:
    - The clean markdown context
    - Data-quality notice derived from VERIFIER annotations
    - Flagged-page warning (from verification_report.json)
    - Explicit instructions on how to handle each verdict tier
    """
    # ── Base instruction ──────────────────────────────────────────────────────
    base = (
        "You are an expert financial document analyst. "
        "Answer the user's question using ONLY the information in the provided document excerpt. "
        "Do not use external knowledge. "
        "Be concise and accurate. "
        "When you reference a table value, cite it as: [Table on page X]."
    )

    # ── Verdict-tier handling instructions ───────────────────────────────────
    tier_instructions = (
        "\n\n=== HOW TO HANDLE TABLE DATA QUALITY TIERS ===\n"
        "The document tables have been verified by an automated system with four quality tiers:\n"
        "  CORRECT (✓)           — fully reliable; cite values without caveat.\n"
        "  PARTIAL_CORRECT (◑)   — mostly reliable but some values may be wrong.\n"
        "                          Cite figures but add: '(may contain minor conversion errors)'.\n"
        "  PARTIAL_INCORRECT (◐) — moderate errors; use with caution.\n"
        "                          Add: '(data accuracy uncertain — verify against original PDF)'.\n"
        "  INCORRECT (✗)         — significantly wrong; do NOT quote specific values.\n"
        "                          State instead: 'This table has conversion errors; "
        "please refer to the original PDF for accurate figures.'\n"
        "=== END TIER INSTRUCTIONS ===\n"
    )

    # ── Data quality notice (from annotations) ───────────────────────────────
    quality_note = build_data_quality_note(annotation_blocks, relevant_pages)

    # ── Flagged-page warning ──────────────────────────────────────────────────
    page_warning = ""
    if flagged_pages and relevant_pages:
        overlap = set(relevant_pages) & flagged_pages
        if overlap:
            page_warning = (
                f"\n\nWARNING: Page(s) {sorted(overlap)} were flagged as having "
                f"significant rendering issues in the conversion report. "
                "Treat all data from these pages with extra caution."
            )

    # ── Assemble ──────────────────────────────────────────────────────────────
    prompt = base + tier_instructions
    if quality_note:
        prompt += "\n\n" + quality_note
    if page_warning:
        prompt += page_warning
    prompt += (
        "\n\nDocument excerpt:\n"
        "---\n"
        + context[:14000]
        + "\n---"
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QA FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def answer_question(question: str) -> str:
    """
    Full pipeline:
    1. Load annotated markdown (or raw fallback)
    2. Load verification report + PDF page text
    3. Locate relevant pages via LLM
    4. Determine flagged pages + annotation-block quality
    5. Build system prompt with tier-aware quality notes
    6. Call the LLM to answer
    7. Append a structured data-quality footer to the answer
    """
    # Step 1: Load context
    context, annotation_blocks = load_context()

    # Step 2: Load supporting data
    report      = load_verification_report()
    pdf_pages   = load_pdf_text_by_page()

    # Step 3: Locate relevant pages
    relevant_pages = locate_relevant_pages(question, pdf_pages)

    # Step 4: Flagged pages
    flagged_pages = get_flagged_pages(report)

    # Step 5: Build system prompt
    system_prompt = build_system_prompt(
        context, report, annotation_blocks, relevant_pages, flagged_pages
    )

    # Step 6: LLM call
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    answer = (response.choices[0].message.content or "").strip()

    # Step 7: Append data-quality footer
    footer = _build_answer_footer(annotation_blocks, relevant_pages, flagged_pages)
    if footer:
        answer += "\n\n" + footer

    return answer


def _build_answer_footer(
    annotation_blocks: list[dict],
    relevant_pages: list[int],
    flagged_pages: set[int],
) -> str:
    """
    Build a markdown-formatted footer summarising the data-quality status
    of tables referenced in the answer.
    """
    if not annotation_blocks:
        return ""

    # Filter to relevant pages where possible
    def _page(b) -> Optional[int]:
        pr = b.get("pdf_ref")
        if pr and pr.get("page", "unknown") != "unknown":
            try:
                return int(pr["page"])
            except ValueError:
                pass
        return None

    relevant_blocks = annotation_blocks
    if relevant_pages:
        on_page = [b for b in annotation_blocks if _page(b) in relevant_pages]
        if on_page:
            relevant_blocks = on_page

    non_correct = [b for b in relevant_blocks if b["status"] != "CORRECT"]
    if not non_correct:
        return ""

    lines = ["---", "**📊 Table Data Quality Summary**"]
    for b in non_correct:
        icon   = _VERDICT_ICON.get(b["status"], "?")
        pr     = b.get("pdf_ref") or {}
        page   = pr.get("page", "?")
        sect   = pr.get("section", "")
        ratio  = b.get("correct_ratio", 0.0)
        errors = ", ".join(b["errors"][:4]) if b["errors"] else "none"

        location = f"page {page}" + (f" / {sect}" if sect else "")
        pd = b.get("partial_detail")
        pd_detail = ""
        if pd:
            pd_detail = (
                f" — {pd['correct_rows']} rows OK, "
                f"{pd['incorrect_rows']} rows with issues"
            )

        lines.append(
            f"- {icon} **{b['status']}** ({location}): "
            f"correct_ratio={ratio:.0%}{pd_detail}. "
            f"Errors: {errors}"
        )

    # Add flagged-page note if applicable
    if flagged_pages and relevant_pages:
        overlap = set(relevant_pages) & flagged_pages
        if overlap:
            lines.append(
                f"- ⚠️ Page(s) {sorted(overlap)} flagged for rendering issues in conversion report."
            )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_loop() -> None:
    print("=" * 60)
    print("QA System for PDF-to-Markdown Documents (v2 — Partial Labels)")
    if ANNOTATED_MD.exists():
        print(f"Using annotated markdown: {ANNOTATED_MD}")
    else:
        print(f"No annotated markdown found — using raw: {CONTEXT_PATH}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print("\nAnswer:")
        print("-" * 40)
        try:
            result = answer_question(question)
            print(result)
        except Exception as e:
            print(f"[error] {e}")
        print("-" * 40)


if __name__ == "__main__":
    interactive_loop()
