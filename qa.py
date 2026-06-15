import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE            = Path(__file__).parent
OUTPUT_DIR      = BASE / "hsbc_output"
CONTEXT_PATH    = OUTPUT_DIR / "hsbc.md"
REPORT_PATH     = OUTPUT_DIR / "verification" / "verification_report.json"
PAGE_IMAGES_DIR = OUTPUT_DIR / "verification" / "page_images"
PDF_TEXT_PATH   = OUTPUT_DIR / "verification" / "pdf_text_by_page.json"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_context() -> str:
    return CONTEXT_PATH.read_text(encoding="utf-8")


def load_verification_report() -> dict | None:
    """Return the parsed verification_report.json, or None if not present."""
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_pdf_text_by_page() -> list[str]:
    """
    Load verification/pdf_text_by_page.json and return the list of per-page
    text strings (1-based: index 0 == page 1).
    Returns an empty list if the file is missing.
    """
    if not PDF_TEXT_PATH.exists():
        return []
    with open(PDF_TEXT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pages", [])


# ── Hallucination / inaccuracy detection ───────────────────────────────────────

def get_flagged_pages(report: dict) -> set[int]:
    """
    Collect every page number that the verifier has flagged as inaccurate.

    Sources examined:
      • report["inaccurate_sections"]  – per-page dimensional flags
      • report["findings"]             – global findings that carry a page key
      • report["per_page_text_recall"] – pages whose text recall is critically low
      • report["route"]                – if MANUAL_REVIEW, everything is suspect
    """
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

    if report.get("route") == "MANUAL_REVIEW":
        n_pages = report.get("page_count", {}).get("pdf", 0)
        if n_pages:
            flagged.update(range(1, n_pages + 1))

    return flagged


# ── LLM-based page localization ────────────────────────────────────────────────

def locate_relevant_pages(question: str, pdf_pages: list[str]) -> list[int]:
    """
    Use the per-page PDF text (from pdf_text_by_page.json) to ask the LLM
    which page(s) contain the information needed to answer the question.

    This approach is reliable because pdf_text_by_page.json is extracted
    directly from the PDF (not from the Marker-rendered markdown), so it
    is not affected by rendering bugs or marker-injection failures.

    Returns a sorted list of 1-based page numbers.
    If the LLM cannot identify specific pages it returns [].
    """
    if not pdf_pages:
        return []

    total_pages = len(pdf_pages)

    # Build a compact per-page digest (first 800 chars per page is enough
    # for topic identification without blowing up the context window).
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
                    "If the answer is on one page, return only that page. "
                    "Prefer fewer pages over more; only include additional pages "
                    "if the answer genuinely spans multiple pages. "
                    "Reply with ONLY a JSON array of integers, e.g. [3] or [3,4]. "
                    "If you cannot determine the page(s), reply with []. "
                    "Do NOT include any explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Per-page PDF text:\n{page_dump}"
                ),
            },
        ],
        temperature=0,
        max_tokens=32,
    )

    raw = response.choices[0].message.content.strip()
    match = re.search(r"\[([\d,\s]*)\]", raw)
    if not match:
        return []
    try:
        nums = [int(x) for x in match.group(1).split(",") if x.strip()]
        return sorted(set(p for p in nums if 1 <= p <= total_pages))
    except ValueError:
        return []


# ── classify question ──────────────────────────────────────────────────────────

def classify_question(
    question: str,
    flagged_pages: set[int],
    report: dict,
    pdf_pages: list[str],
) -> tuple[str, list[int]]:
    """
    Determine whether the question intersects with flagged pages.

    Strategy (in priority order):
      1. Extract explicit page numbers from the question text.
      2. Ask the LLM to locate the relevant pages using pdf_text_by_page.json.
      3. Fall back to keyword overlap with flagged finding details
         (scoped — returns only the matching flagged pages, not all of them).

    Returns
    -------
    classification : "safe" | "flagged" | "unknown"
    relevant_pages : list of page numbers the question seems to target
                     (only the RELEVANT ones, never all flagged pages)
    """
    if not flagged_pages:
        return "safe", []

    total_pages = report.get("page_count", {}).get("pdf", 0) or len(pdf_pages) or 21

    # ─── 1. Explicit page references in the question text ─────────────────────
    page_hint_re = re.compile(
        r"\b(?:page|p\.?|pg\.?)\s*(\d{1,3})|\b(\d{1,3})\s*(?:st|nd|rd|th)?\s*page\b",
        re.IGNORECASE,
    )
    explicit_pages = [
        int(m.group(1) or m.group(2)) for m in page_hint_re.finditer(question)
    ]
    explicit_pages = list(dict.fromkeys(explicit_pages))  # deduplicate

    if explicit_pages:
        flagged_hit = [p for p in explicit_pages if p in flagged_pages]
        if flagged_hit:
            return "flagged", flagged_hit
        return "safe", explicit_pages

    # ─── 2. LLM-based page localization via pdf_text_by_page.json ─────────────
    llm_pages = locate_relevant_pages(question, pdf_pages)

    if llm_pages:
        flagged_hit = [p for p in llm_pages if p in flagged_pages]
        safe_hit    = [p for p in llm_pages if p not in flagged_pages]

        if flagged_hit and not safe_hit:
            # All relevant pages are flagged → refuse and show images.
            # Surface only the flagged pages that are truly relevant
            # (the LLM already narrowed these down, so trust the result).
            return "flagged", flagged_hit

        if flagged_hit and safe_hit:
            # Mixed: some flagged, some safe → partial answer + images for flagged
            return "unknown", flagged_hit

        # No relevant page is flagged → safe to answer
        return "safe", llm_pages

    # ─── 3. Keyword-overlap fallback (scoped to flagged pages only) ───────────
    # Does NOT return all flagged pages — only the ones whose finding
    # detail text overlaps with the question words.
    flagged_details = " ".join(
        s.get("detail", "") for s in report.get("inaccurate_sections", [])
    ).lower()
    q_words = set(re.findall(r"[a-z]{3,}", question.lower()))
    overlap = q_words & set(re.findall(r"[a-z]{3,}", flagged_details))

    if len(overlap) >= 2:
        targeted: list[int] = []
        for section in report.get("inaccurate_sections", []):
            page = section.get("page")
            if page is None:
                continue
            detail_words = set(re.findall(r"[a-z]{3,}", section.get("detail", "").lower()))
            if detail_words & overlap:
                targeted.append(int(page))
        targeted = sorted(set(targeted))
        if targeted:
            return "unknown", targeted

    return "unknown", []


# ── Page-image lookup ──────────────────────────────────────────────────────────

def find_page_images(pages: list[int]) -> list[Path]:
    """
    Return Path objects for every page image matching the given page numbers.
    Supports common naming conventions produced by tableimg.py.
    """
    if not PAGE_IMAGES_DIR.exists():
        return []

    images: list[Path] = []
    for page in pages:
        candidates = [
            PAGE_IMAGES_DIR / f"page_{page}.png",
            PAGE_IMAGES_DIR / f"page_{page}.jpg",
            PAGE_IMAGES_DIR / f"page{page}.png",
            PAGE_IMAGES_DIR / f"page{page}.jpg",
            PAGE_IMAGES_DIR / f"page_{page:03d}.png",
            PAGE_IMAGES_DIR / f"page_{page:03d}.jpg",
        ]
        for c in candidates:
            if c.exists():
                images.append(c)
                break
        else:
            globs = list(PAGE_IMAGES_DIR.glob(f"*page*{page}*"))
            if globs:
                images.append(sorted(globs)[0])

    seen: set[Path] = set()
    unique: list[Path] = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique.append(img)
    return unique


# ── LLM answer ─────────────────────────────────────────────────────────────────

def answerdirectly(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful QA assistant. "
                    "Answer the user's question using only the provided HSBC Markdown document. "
                    "If the answer is not in the document, say that you cannot find it in the provided document."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the HSBC Markdown document:\n\n"
                    f"```markdown\n{context}\n```\n\n"
                    f"User question:\n{question}"
                ),
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ── Main entry-point ───────────────────────────────────────────────────────────

def answer(question: str) -> dict:
    """
    Answer a question about the HSBC document with hallucination-awareness.

    Returns a dict:
        "status"  : "answered" | "flagged" | "partial"
        "answer"  : str  (LLM answer or refusal message)
        "images"  : list[str]  (absolute paths to relevant page images)
        "warning" : str | None

    Status meanings
    ---------------
    "answered"  No flagged sections involved; answer is provided as-is.
    "partial"   Question may touch flagged sections; answer provided with warning.
    "flagged"   Question clearly targets unreliable content; answer refused and
                original page image(s) returned for manual calculation.
    """
    context   = load_context()
    report    = load_verification_report()
    pdf_pages = load_pdf_text_by_page()   # per-page ground-truth text from PDF

    if report is None:
        return {
            "status": "answered",
            "answer": answerdirectly(question, context),
            "images": [],
            "warning": (
                "⚠️  No verification report found. "
                "The accuracy of the parsed document could not be verified."
            ),
        }

    flagged_pages = get_flagged_pages(report)
    classification, relevant_pages = classify_question(
        question, flagged_pages, report, pdf_pages
    )

    # ── FLAGGED ────────────────────────────────────────────────────────────────
    if classification == "flagged":
        images = find_page_images(relevant_pages)
        flagged_findings = [
            s for s in report.get("inaccurate_sections", [])
            if s.get("page") in relevant_pages
        ]
        detail_lines = [
            f"  • Page {s.get('page', '?')}: [{s.get('severity', '-').upper()}] "
            f"{s.get('detail', '')}"
            for s in flagged_findings[:10]
        ]
        detail_block = "\n".join(detail_lines) if detail_lines else "  (see full report)"

        warning = (
            f"⚠️  The conversion of page(s) {sorted(relevant_pages)} was flagged as "
            f"INACCURATE by the verifier.\n"
            f"Verification issues:\n{detail_block}\n\n"
            f"The markdown text for this section is unreliable. "
            f"Please refer to the original page image(s) below and calculate manually."
        )

        return {
            "status": "flagged",
            "answer": (
                "I cannot provide a reliable answer for this question because the "
                "relevant section(s) of the document were marked as inaccurate during "
                "verification. Please consult the original page image(s) provided."
            ),
            "images": [str(p) for p in images],
            "warning": warning,
        }

    # ── UNKNOWN / PARTIAL ──────────────────────────────────────────────────────
    if classification == "unknown":
        images     = find_page_images(relevant_pages)
        llm_answer = answerdirectly(question, context)

        pages_str = str(sorted(relevant_pages)) if relevant_pages else "(unknown)"
        warning = (
            f"⚠️  Some page(s) relevant to this question were flagged as potentially "
            f"inaccurate ({pages_str}). "
            f"The answer below is based on the parsed markdown, which may contain "
            f"errors in those sections. "
            f"Please cross-check against the original page images if precision is critical."
        )

        return {
            "status": "partial",
            "answer": llm_answer,
            "images": [str(p) for p in images],
            "warning": warning,
        }

    # ── SAFE ───────────────────────────────────────────────────────────────────
    return {
        "status": "answered",
        "answer": answerdirectly(question, context),
        "images": [],
        "warning": None,
    }


# ── CLI helper ─────────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    """Pretty-print an answer() result dict to stdout."""
    print()
    if result["warning"]:
        print(result["warning"])
        print()
    print("Answer:", result["answer"])
    if result["images"]:
        print()
        print("Relevant page image(s) for manual verification:")
        for img in result["images"]:
            print(" ", img)
    print()


if __name__ == "__main__":
    question = input("Ask a question about the HSBC document: ")
    result = answer(question)
    _print_result(result)
