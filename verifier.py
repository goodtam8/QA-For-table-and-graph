"""
verifier.py — Main entry point for the general rule-based table verifier.

Usage:
    python verifier.py                          # Verify hsbc.md, write hsbc_annotated.md
    python verifier.py --input hsbc.md          # Specify input markdown
    python verifier.py --output annotated.md     # Specify output path
    python verifier.py --llm                    # Enable LLM-assisted verification

This script:
    1. Reads the raw markdown (from Marker output)
    2. Loads pdf_tables_by_page.json for ground-truth PDF table data
    3. Loads verification_rules.json for rule definitions
    4. Runs the annotator against each table
    5. Writes the annotated markdown with VERIFY + PDF_REF comment tags
    6. Prints a summary report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

from openai import AzureOpenAI

from verifier import (
    MarkdownTableParser,
    RulesEngine,
    ConfidenceScorer,
    MarkdownAnnotator,
    load_pdf_tables_by_page,
    get_pdf_table_for_md_table,
)

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "hsbc_output"
DEFAULT_MARKDOWN = OUTPUT_DIR / "hsbc.md"
DEFAULT_ANNOTATED = OUTPUT_DIR / "hsbc_annotated.md"
DEFAULT_PDF_TABLES = OUTPUT_DIR / "verification" / "pdf_tables_by_page.json"
DEFAULT_PAGE_IMAGES = OUTPUT_DIR / "verification" / "page_images"
DEFAULT_RULES = BASE_DIR / "verification_rules.json"
DEFAULT_REPORT = OUTPUT_DIR / "verification" / "annotated_report.json"


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #

def make_llm_client() -> AzureOpenAI | None:
    try:
        return AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
    except Exception as e:
        print(f"[WARN] Could not create LLM client: {e}")
        return None


# --------------------------------------------------------------------------- #
# Main verify function
# --------------------------------------------------------------------------- #

def verify(
    markdown_path: str | Path = DEFAULT_MARKDOWN,
    output_path: str | Path = DEFAULT_ANNOTATED,
    pdf_tables_path: str | Path = DEFAULT_PDF_TABLES,
    page_images_dir: str | Path = DEFAULT_PAGE_IMAGES,
    rules_path: str | Path = DEFAULT_RULES,
    report_path: str | Path = DEFAULT_REPORT,
    use_llm: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run the full verification pipeline on a markdown document.

    Returns a summary dict.
    """
    md_path = Path(markdown_path)
    out_path = Path(output_path)
    pdf_tables_path = Path(pdf_tables_path)
    page_images_path = Path(page_images_dir)
    rules_p = Path(rules_path)
    report_p = Path(report_path)

    # ── Load markdown ────────────────────────────────────────────────────
    if not md_path.exists():
        print(f"[ERROR] Markdown file not found: {md_path}")
        print(f"  Run: python parser.py --input hsbc.pdf --output ./hsbc_output")
        return {"error": f"Markdown not found: {md_path}"}

    markdown_text = md_path.read_text(encoding="utf-8")
    print(f"[OK] Loaded markdown: {md_path}")
    print(f"     ({len(markdown_text)} chars)")

    # ── Load PDF table ground truth ───────────────────────────────────────
    if pdf_tables_path.exists():
        with open(pdf_tables_path, encoding="utf-8") as f:
            pdf_tables_data = json.load(f)
        print(f"[OK] Loaded PDF tables: {pdf_tables_path}")
    else:
        print(f"[WARN] PDF tables file not found: {pdf_tables_path}")
        print(f"       Rule checks that need PDF ground truth will be skipped.")
        pdf_tables_data = {"pages": []}

    # ── Build annotator ──────────────────────────────────────────────────
    llm_client = make_llm_client() if use_llm else None

    rules_engine = RulesEngine(rules_path=str(rules_p))
    confidence_scorer = ConfidenceScorer(rules_path=str(rules_p))

    annotator = MarkdownAnnotator(
        rules_engine=rules_engine,
        confidence_scorer=confidence_scorer,
        pdf_tables_by_page=pdf_tables_data,
        page_images_dir=str(page_images_path),
        llm_client=llm_client,
    )

    # ── Parse tables first (to build page map) ────────────────────────────
    parser = MarkdownTableParser(markdown_text)
    print(f"[OK] Found {len(parser.tables)} tables in markdown")

    # Build PDF page map: table_index → PDF page (from page markers or heuristics)
    pdf_page_map: dict[int, int] = {}
    for tbl in parser.tables:
        if tbl.md_page is not None:
            pdf_page_map[tbl.table_index] = tbl.md_page

    # ── Annotate ─────────────────────────────────────────────────────────
    if use_llm and llm_client:
        print("[OK] Running LLM-assisted verification...")
        annotated = annotator.annotate_with_llm(markdown_text, pdf_page_map)
    else:
        print("[OK] Running heuristic rule verification...")
        annotated = annotator.annotate(markdown_text, pdf_page_map)

    # ── Write output ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(annotated, encoding="utf-8")
    print(f"[OK] Annotated markdown written: {out_path}")

    # ── Write report ──────────────────────────────────────────────────────
    summary = annotator.get_verification_summary()
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Verification report written: {report_p}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total tables:   {summary['total_tables']}")
    print(f"  CORRECT:        {summary['correct']}")
    print(f"  INCORRECT:      {summary['incorrect']}")
    print(f"  REVIEW:         {summary['review']}")
    print(f"{'=' * 60}\n")

    if verbose:
        for v in annotator.verifications:
            status = "CORRECT" if v.verdict.label == "CORRECT" else "INCORRECT"
            print(f"  Table {v.table_index + 1}: {status} (confidence={v.verdict.confidence:.2f})")
            if v.verdict.all_errors:
                for err in v.verdict.all_errors:
                    print(f"    - {err}")
            if v.md_page:
                print(f"    Page: {v.md_page}, Section: {v.section or 'N/A'}")

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="General rule-based table verifier — annotate markdown with VERIFY tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verifier.py
    python verifier.py --input hsbc_output/hsbc.md --output annotated.md
    python verifier.py --llm --verbose
        """,
    )
    ap.add_argument(
        "--input", "-i",
        default=str(DEFAULT_MARKDOWN),
        help=f"Input markdown path (default: {DEFAULT_MARKDOWN})",
    )
    ap.add_argument(
        "--output", "-o",
        default=str(DEFAULT_ANNOTATED),
        help=f"Output annotated markdown path (default: {DEFAULT_ANNOTATED})",
    )
    ap.add_argument(
        "--pdf-tables",
        default=str(DEFAULT_PDF_TABLES),
        help=f"Path to pdf_tables_by_page.json (default: {DEFAULT_PDF_TABLES})",
    )
    ap.add_argument(
        "--page-images",
        default=str(DEFAULT_PAGE_IMAGES),
        help=f"Page images directory (default: {DEFAULT_PAGE_IMAGES})",
    )
    ap.add_argument(
        "--rules",
        default=str(DEFAULT_RULES),
        help=f"Path to verification_rules.json (default: {DEFAULT_RULES})",
    )
    ap.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help=f"Output report JSON path (default: {DEFAULT_REPORT})",
    )
    ap.add_argument(
        "--llm", "-l",
        action="store_true",
        help="Enable LLM-assisted verification for ambiguous tables",
    )
    ap.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-table results",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Run verification without writing output files (dry run)",
    )

    args = ap.parse_args()

    if args.check_only:
        print("[DRY RUN] Skipping file writes.")
        markdown_text = Path(args.input).read_text(encoding="utf-8")
        parser = MarkdownTableParser(markdown_text)
        rules_engine = RulesEngine(rules_path=args.rules)
        confidence_scorer = ConfidenceScorer(rules_path=args.rules)

        with open(args.pdf_tables, encoding="utf-8") as f:
            pdf_tables_data = json.load(f)

        print(f"[DRY RUN] Found {len(parser.tables)} tables")
        for tbl in parser.tables:
            pdf_cells, _ = get_pdf_table_for_md_table(tbl, pdf_tables_data, tbl.md_page)
            result = rules_engine.check_table(tbl, pdf_cells)
            verdict = confidence_scorer.compute_verdict(result)
            status = "CORRECT" if verdict.label == "CORRECT" else "INCORRECT"
            print(f"  Table {tbl.table_index + 1}: {status} (conf={verdict.confidence:.2f}) "
                  f"- {verdict.all_errors or 'no errors'}")
        return

    summary = verify(
        markdown_path=args.input,
        output_path=args.output,
        pdf_tables_path=args.pdf_tables,
        page_images_dir=args.page_images,
        rules_path=args.rules,
        report_path=args.report,
        use_llm=args.llm,
        verbose=args.verbose,
    )

    if "error" in summary:
        sys.exit(1)


if __name__ == "__main__":
    main()
