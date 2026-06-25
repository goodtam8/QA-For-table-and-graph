from pathlib import Path
import os
import re
import requests
from prompt import STAGE2_PROMPT
from dotenv import load_dotenv

load_dotenv()

API_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT     = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
API_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
ENDPOINT       = (
    f"https://hkust.azure-api.net/openai/deployments/{DEPLOYMENT}"
    f"/chat/completions?api-version={API_VERSION}"
)

INPUT_FILE  = Path("/Users/goodtam8/Documents/Programming/QA-For-table-and-graph/hsbc_output/hsbc.md")
OUTPUT_DIR  = Path("/Users/goodtam8/Documents/Programming/QA-For-table-and-graph/hsbc_output")
OUTPUT_FILE = OUTPUT_DIR / "hsbc_revised.md"

# ~3000 words per chunk keeps prompt+output well within the 16k–65k output limits.
# Adjust down if you still see truncation.
CHUNK_WORD_LIMIT = 3000


def strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", text, re.IGNORECASE)
    return match.group(1).strip() if match else text


def split_into_chunks(markdown: str, word_limit: int) -> list[str]:
    """
    Split markdown into chunks that respect paragraph boundaries and stay
    under `word_limit` words each.
    """
    paragraphs = re.split(r"\n{2,}", markdown)
    chunks, current_chunk, current_words = [], [], 0

    for para in paragraphs:
        para_words = len(para.split())
        # If adding this paragraph would exceed the limit, flush first
        if current_words + para_words > word_limit and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_words = [], 0
        current_chunk.append(para)
        current_words += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def refine_chunk(chunk: str, chunk_index: int, total_chunks: int) -> str:
    """Send a single chunk to the LLM and return the refined text."""
    system_message = (
        "You are an expert document converter and reviewer specializing in structured "
        "financial documents. Return only the fully revised markdown content for this "
        f"section (chunk {chunk_index + 1} of {total_chunks}). "
        "Do not add explanations, JSON, commentary, or a preamble."
    )
    user_message = (
        f"{STAGE2_PROMPT}\n\n"
        "Below is the markdown section that must be revised. "
        "Preserve all content; do not truncate or summarize.\n\n"
        "```markdown\n"
        f"{chunk}\n"
        "```"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        # Explicitly request a generous output budget.
        # Adjust to the actual limit of your deployment.
        "max_completion_tokens": 16000,
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    print(f"  → Sending chunk {chunk_index + 1}/{total_chunks} "
          f"({len(chunk.split())} words) …")
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=600)
    print(f"    Status: {response.status_code}")
    response.raise_for_status()

    data = response.json()
    finish_reason = data["choices"][0].get("finish_reason", "unknown")
    if finish_reason != "stop":
        print(f"    ⚠ WARNING: finish_reason='{finish_reason}' for chunk {chunk_index + 1}. "
              "Content may be truncated — consider reducing CHUNK_WORD_LIMIT.")

    content = data["choices"][0]["message"]["content"]
    return strip_code_fences(content)


def main():
    if not API_KEY:
        raise SystemExit("Missing AZURE_OPENAI_API_KEY environment variable.")
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_markdown = INPUT_FILE.read_text(encoding="utf-8")

    chunks = split_into_chunks(stage1_markdown, word_limit=CHUNK_WORD_LIMIT)
    print(f"Split document into {len(chunks)} chunks.")

    revised_chunks = []
    for i, chunk in enumerate(chunks):
        revised = refine_chunk(chunk, i, len(chunks))
        revised_chunks.append(revised)

    revised_markdown = "\n\n".join(revised_chunks)
    OUTPUT_FILE.write_text(revised_markdown, encoding="utf-8")
    print(f"\nSaved revised markdown ({len(revised_chunks)} chunks) to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()