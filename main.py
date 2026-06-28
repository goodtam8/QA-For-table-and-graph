"""
main.py — End-to-end pipeline: PDF → Markdown → Verify → Annotated → QA

Steps:
    1. parser.py   — Convert PDF to markdown (Marker)
    2. verifier.py — Run general rule-based verification → annotated markdown
    3. qa.py       — Answer user questions using annotated markdown

Usage:
    python main.py                          # Full pipeline
    python main.py --skip-parse            # Skip PDF parsing (use existing markdown)
    python main.py --skip-verify           # Skip verification (use existing annotated.md)
    python main.py --llm                   # Enable LLM-assisted verification
    python main.py --question "..."        # Answer a single question and exit
"""

import sys
import argparse

from parser import main as run_pdf_job
from verifier import verify
from qa import answer, _print_result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end HSBC document QA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps run by default:
    1. parser.py  — PDF → Markdown (via Marker)
    2. verifier.py — Run rule-based verification → hsbc_annotated.md
    3. qa.py      — Interactive QA session

Use --skip-parse to reuse existing hsbc.md.
Use --skip-verify to reuse existing hsbc_annotated.md.
Use --llm to enable LLM-assisted verification for ambiguous tables.
Use --question to answer a single question and exit.
        """,
    )
    ap.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip PDF parsing step (use existing hsbc.md)",
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step (use existing hsbc_annotated.md)",
    )
    ap.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-assisted verification for ambiguous tables",
    )
    ap.add_argument(
        "--question", "-q",
        default=None,
        help="Answer a single question and exit (don't enter interactive mode)",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed verification results",
    )

    args = ap.parse_args()

    # ── Step 1: PDF → Markdown ─────────────────────────────────────────────
    if not args.skip_parse:
        print("=" * 60)
        print("STEP 1: PDF → Markdown (Marker)")
        print("=" * 60)
        try:
            run_pdf_job()
        except Exception as e:
            print(f"[ERROR] PDF parsing failed: {e}")
            sys.exit(1)
        print()

    # ── Step 2: Verify → Annotated Markdown ─────────────────────────────────
    if not args.skip_verify:
        print("=" * 60)
        print("STEP 2: Rule-Based Verification")
        print("=" * 60)
        try:
            summary = verify(
                markdown_path="hsbc_output/hsbc.md",
                output_path="hsbc_output/hsbc_annotated.md",
                use_llm=args.llm,
                verbose=args.verbose,
            )
            if "error" in summary:
                print(f"[ERROR] Verification failed: {summary['error']}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Verification failed: {e}")
            sys.exit(1)
        print()
    else:
        print("[SKIP] Verification step skipped (using existing annotated markdown)")

    # ── Step 3: QA ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: QA System Ready")
    print("=" * 60)
    print("Annotated markdown loaded. Ready to answer questions.")
    print("Use --question to answer a single question.")
    print()

    if args.question:
        result = answer(args.question)
        _print_result(result)
    else:
        print("Enter your questions below (Ctrl+C to exit):\n")
        while True:
            try:
                q = input("Question: ").strip()
                if not q:
                    continue
                result = answer(q)
                _print_result(result)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break


if __name__ == "__main__":
    main()
