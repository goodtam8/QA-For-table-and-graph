from pathlib import Path
import os
import re
import requests
from prompt import STAGE2_PROMPT
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-02-01-preview")
ENDPOINT = f"https://hkust.azure-api.net/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"

INPUT_FILE = Path("/Users/goodtam8/Documents/Programming/QA-For-table-and-graph/hsbc_output/hsbc.md")
OUTPUT_DIR = Path("/Users/goodtam8/Documents/Programming/QA-For-table-and-graph/hsbc_output")
OUTPUT_FILE = OUTPUT_DIR / "hsbc_revised.md"

def strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", text, re.IGNORECASE)
    return match.group(1).strip() if match else text

def main():
    if not API_KEY:
        raise SystemExit("Missing AZURE_OPENAI_API_KEY environment variable.")

    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_markdown = INPUT_FILE.read_text(encoding="utf-8")

    system_message = (
        "You are an expert document converter and reviewer specializing in structured "
        "financial documents. Return only the fully revised markdown document. "
        "Do not add explanations, JSON, or commentary."
    )

    user_message = (
        f"{STAGE2_PROMPT}\n\n"
        "Below is the Stage 1 markdown that must be revised.\n\n"
        "```markdown\n"
        f"{stage1_markdown}\n"
        "```"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=600)
    print(response.status_code)
    print(response.text)
    response.raise_for_status()

    data = response.json()
    revised_markdown = data["choices"][0]["message"]["content"]
    revised_markdown = strip_code_fences(revised_markdown)

    OUTPUT_FILE.write_text(revised_markdown, encoding="utf-8")
    print(f"Saved revised markdown to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()