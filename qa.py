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

# ── Paths ─────────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUTPUT_DIR      = BASE / "hsbc_output"
CONTEXT_PATH    = OUTPUT_DIR / "hsbc.md"
REPORT_PATH     = OUTPUT_DIR / "verification" / "verification_report.json"
PAGE_IMAGES_DIR = OUTPUT_DIR / "verification" / "page_images"


# ── Verification report loader ──────────────────────────────────────────────────────────────────────────

def load_verification_report() -> dict | None:
    """Return the parsed verification_report.json, or None if not present."""
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Hallucination / inaccuracy detection ─────────────────────────────────────────────────────────────

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


# ── LLM-based page localization ─────────────────────────────────────────────────────────────────────────

def locate_relevant_pages(question: str, context: str, total_pages: int) -> list[int]:
    """
    Ask the LLM to identify which page(s) of the document contain information
    needed to answer the question.  Returns a sorted list of 1-based page numbers.

    The markdown document produced by the parser contains page-break markers of
    the form  <!-- Page N -->  that the LLM can use to locate sections.
    If the LLM cannot identify specific pages it returns an empty list.
    """
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document navigation assistant. "
                    "The user will give you a question and a Markdown document. "
                    "The document contains page-break comments like <!-- Page 3 -->. "
                    f"The document has {total_pages} pages total. "
                    "Your task: identify which page number(s) contain the information "
                    "needed to answer the question. "
                    "Reply with ONLY a JSON array of integers, e.g. [3] or [3,4]. "
                    "If you cannot determine the page(s), reply with []. "
                    "Do NOT include any explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Document:\n```markdown\n{context}\n```"
                ),
            },
        ],
        temperature=0,
        max_tokens=64,
    )

    raw = response.choices[0].message.content.strip()
    # Parse the JSON array robustly
    match = re.search(r"\[([\d,\s]*)\]", raw)
    if not match:
        return []
    try:
        nums = [int(x) for x in match.group(1).split(",") if x.strip()]
        return sorted(set(p for p in nums if 1 <= p <= total_pages))
    except ValueError:
        return []


# ── classify question ─────────────────────────────────────────────────────────────────────────────────────────

def classify_question(
    question: str,
    flagged_pages: set[int],
    report: dict,
    context: str,
) -> tuple[str, list[int]]:
    """
    Determine whether the question intersects with flagged pages.

    Strategy (in priority order):
      1. Extract explicit page numbers from the question text.
      2. Ask the LLM to locate the relevant pages in the document.
      3. Fall back to keyword overlap with flagged finding details.

    Returns
    -------
    classification : "safe" | "flagged" | "unknown"
    relevant_pages : list of page numbers the question seems to target
                     (only the RELEVANT ones, never all flagged pages)
    """
    if not flagged_pages:
        return "safe", []

    total_pages = report.get("page_count", {}).get("pdf", 0) or 21

    # ─── 1. Explicit page references in the question text ───
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

    # ─── 2. LLM-based page localization (the key fix) ───
    llm_pages = locate_relevant_pages(question, context, total_pages)

    if llm_pages:
        flagged_hit = [p for p in llm_pages if p in flagged_pages]
        safe_hit    = [p for p in llm_pages if p not in flagged_pages]

        if flagged_hit and not safe_hit:
            # All relevant pages are flagged → refuse and return images
            return "flagged", flagged_hit

        if flagged_hit and safe_hit:
            # Some relevant pages are flagged, some are not → partial
            return "unknown", flagged_hit   # only return the FLAGGED subset as images

        # No relevant page is flagged → safe to answer
        return "safe", llm_pages

    # ─── 3. Keyword-overlap fallback (conservative, but scoped to flagged pages only) ───
    # Unlike the original code, we do NOT return ALL flagged pages as relevant;
    # we just mark the classification as "unknown" without specific page images
    # unless keyword overlap is very strong.
    flagged_details = " ".join(
        s.get("detail", "") for s in report.get("inaccurate_sections", [])
    ).lower()
    q_words = set(re.findall(r"[a-z]{3,}", question.lower()))
    overlap = q_words & set(re.findall(r"[a-z]{3,}", flagged_details))

    if len(overlap) >= 2:
        # Return only the flagged pages whose detail text contains the overlapping words,
        # not the entire flagged set.
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


# ── Page-image lookup ───────────────────────────────────────────────────────────────────────────────────

def find_page_images(pages: list[int]) -> list[Path]:
    """
    Return Path objects for every page image matching the given page numbers.
    Supports common naming conventions produced by parser.py.
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


# ── Context loader ────────────────────────────────────────────────────────────────────────────────────────

def load_context() -> str:
    return CONTEXT_PATH.read_text(encoding="utf-8")


# ── LLM answer ──────────────────────────────────────────────────────────────────────────────────────────────

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


# ── Main entry-point ──────────────────────────────────────────────────────────────────────────────────

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
    context = load_context()
    report  = load_verification_report()

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
    # Pass context so classify_question can call the LLM locator
    classification, relevant_pages = classify_question(
        question, flagged_pages, report, context
    )

    # ── FLAGGED: question clearly targets unreliable content ───────────────────────────
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

    # ── UNKNOWN / PARTIAL: possible overlap with flagged content ───────────────────────
    if classification == "unknown":
        # relevant_pages here contains only the FLAGGED pages that touch the question.
        images      = find_page_images(relevant_pages)
        llm_answer  = answerdirectly(question, context)

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

    # ── SAFE: no overlap with flagged content ──────────────────────────────────────────────────
    return {
        "status": "answered",
        "answer": answerdirectly(question, context),
        "images": [],
        "warning": None,
    }


# ── CLI helper ────────────────────────────────────────────────────────────────────────────────────────────

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
