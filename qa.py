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

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
OUTPUT_DIR      = _BASE / "hsbc_output"
CONTEXT_PATH    = OUTPUT_DIR / "hsbc.md"
REPORT_PATH     = OUTPUT_DIR / "verification" / "verification_report.json"
PAGE_IMAGES_DIR = OUTPUT_DIR / "verification" / "page_images"


# ── Verification report loader ────────────────────────────────────────────────

def load_verification_report() -> dict | None:
    """Return the parsed verification_report.json, or None if not present."""
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Hallucination / inaccuracy detection ─────────────────────────────────────

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

    # 1) inaccurate_sections
    for section in report.get("inaccurate_sections", []):
        page = section.get("page")
        if page is not None:
            flagged.add(int(page))

    # 2) findings that carry an explicit page key
    for finding in report.get("findings", []):
        page = finding.get("page")
        if page is not None:
            flagged.add(int(page))

    # 3) per-page text recall below the critical threshold (< 35 %)
    for i, r in enumerate(report.get("per_page_text_recall", []), start=1):
        if isinstance(r, (int, float)) and r < 0.35:
            flagged.add(i)

    # 4) if the overall route is MANUAL_REVIEW, flag every page so nothing
    #    slips through unchecked (conservative but safe for financial docs)
    if report.get("route") == "MANUAL_REVIEW":
        n_pages = report.get("page_count", {}).get("pdf", 0)
        if n_pages:
            flagged.update(range(1, n_pages + 1))

    return flagged


def classify_question(question: str, flagged_pages: set[int],
                       report: dict) -> tuple[str, list[int]]:
    """
    Determine whether the question intersects with flagged pages.

    Returns
    -------
    classification : "safe" | "flagged" | "unknown"
    relevant_pages : list of page numbers the question seems to target
    """
    if not flagged_pages:
        return "safe", []

    # Extract explicit page-number references from the question
    page_hint_re = re.compile(
        r"\b(?:page|p\.?|pg\.?)\s*(\d{1,3})|\b(\d{1,3})\s*(?:st|nd|rd|th)?\s*page\b",
        re.IGNORECASE,
    )
    raw_nums = [int(m.group(1) or m.group(2)) for m in page_hint_re.finditer(question)]
    raw_nums = list(dict.fromkeys(raw_nums))  # deduplicate, preserve order

    if raw_nums:
        # User mentioned specific pages – check intersection directly
        flagged_hit = [p for p in raw_nums if p in flagged_pages]
        if flagged_hit:
            return "flagged", flagged_hit
        return "safe", raw_nums

    # No explicit page numbers – use keyword overlap with flagged finding details
    flagged_details = " ".join(
        s.get("detail", "") for s in report.get("inaccurate_sections", [])
    ).lower()
    q_words = set(re.findall(r"[a-z]{3,}", question.lower()))
    overlap = q_words & set(re.findall(r"[a-z]{3,}", flagged_details))

    # If ≥ 2 meaningful words from the question appear in flagged-finding
    # descriptions, treat it as "unknown" (conservative for financial data).
    if len(overlap) >= 2:
        return "unknown", sorted(flagged_pages)

    return "unknown", sorted(flagged_pages)


# ── Page-image lookup ─────────────────────────────────────────────────────────

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
            # Glob fallback for any file containing the page number
            globs = list(PAGE_IMAGES_DIR.glob(f"*page*{page}*"))
            if globs:
                images.append(sorted(globs)[0])

    # Deduplicate preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique.append(img)
    return unique


# ── Context loader ────────────────────────────────────────────────────────────

def load_context() -> str:
    return CONTEXT_PATH.read_text(encoding="utf-8")


# ── LLM answer (unchanged – kept for backward compatibility) ──────────────────

def answerdirectly(question: str) -> str:
    context = load_context()

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
                "content": f"""
Here is the HSBC Markdown document:

```markdown
{context}
```

User question:
{question}
""",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# ── Main entry-point ──────────────────────────────────────────────────────────

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
    report = load_verification_report()

    if report is None:
        # No verification report present – answer but warn the user.
        return {
            "status": "answered",
            "answer": answerdirectly(question),
            "images": [],
            "warning": (
                "⚠️  No verification report found. "
                "The accuracy of the parsed document could not be verified."
            ),
        }

    flagged_pages = get_flagged_pages(report)
    classification, relevant_pages = classify_question(question, flagged_pages, report)

    # ── FLAGGED: question clearly targets unreliable content ──────────────────
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

    # ── UNKNOWN / PARTIAL: possible overlap with flagged content ─────────────
    if classification == "unknown":
        images = find_page_images(list(flagged_pages))
        llm_answer = answerdirectly(question)

        warning = (
            f"⚠️  Some pages in this document were flagged as potentially inaccurate "
            f"(page(s) {sorted(flagged_pages)}). "
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

    # ── SAFE: no overlap with flagged content ─────────────────────────────────
    return {
        "status": "answered",
        "answer": answerdirectly(question),
        "images": [],
        "warning": None,
    }


# ── CLI helper ────────────────────────────────────────────────────────────────

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
