"""
parser.py �� PDF -> Markdown converter (Marker) that ALSO exports a
self-contained "verification bundle" for a fully decoupled verifier.

Decoupling contract
--------------------
The parser writes everything the verifier needs into a single output
directory.  The verifier never imports Marker, never needs Marker internals,
and only optionally re-reads the original PDF (for stability / independent
table cross-checks).  All structural "ground truth" travels in the bundle.

Artifacts written to <output_directory>/ :

  hsbc.md                      Final markdown (Marker, markdown renderer).
  hsbc_meta.json               Marker metadata (table_of_contents, page_stats).
  hsbc.json                    Marker JSON block tree (block_type structure).
  verification/
    manifest.json              Index + integrity hashes of every artifact.
    pdf_text_by_page.json      Ground-truth text extracted directly from PDF,
                               per page (independent of Marker).
    pdf_tables_by_page.json    Tables re-extracted directly from the PDF
                               (row/col counts + cell text) per page.
    md_block_counts.json       Block-type counts derived from Marker JSON,
                               aggregated and per-page.
    table_bboxes.json          Bounding box polygons for every Table block
                               found in the Marker JSON tree, keyed by
                               page_index.  Used by verifier/qa for
                               vision-fallback cropping.
    page_images/               (generated separately by tableimg.py)
      page_001.png             High-res PNG renders of each PDF page.
      page_002.png             Run:  python tableimg.py --input <pdf>
      ...                            --output <output>/verification/page_images
    stability/                 (optional) second-run markdown for rerun diff.
      hsbc_run2.md

Run:
    python parser.py --input hsbc.pdf --output ./hsbc_output
    python parser.py --input hsbc.pdf --output ./hsbc_output --stability-rerun
"""

import os
import json
import hashlib
import argparse
from collections import Counter, defaultdict
from multiprocessing import freeze_support

from dotenv import load_dotenv

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from marker.config.parser import ConfigParser


# --------------------------------------------------------------------------- #
# Marker configuration
# --------------------------------------------------------------------------- #
def build_config(output_format: str) -> dict:
    """Marker config. output_format is swapped to also emit a JSON block tree."""
    return {
        "output_format": output_format,
        "use_llm": True,
        "force_ocr": True,
        "llm_service": "marker.services.azure_openai.AzureOpenAIService",
        "azure_endpoint": "https://hkust.azure-api.net/openai",
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "deployment_name": "gpt-4o-mini",
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }


def make_converter(output_format: str, models: dict) -> PdfConverter:
    config_parser = ConfigParser(build_config(output_format))
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# 1) Ground-truth PDF text, per page  (independent of Marker)
# --------------------------------------------------------------------------- #
def extract_pdf_text_by_page(pdf_path: str) -> dict:
    """
    Return {"engine": <name>, "pages": [text, ...]}.
    Tries pdftext (what Marker uses), then pdfplumber, then PyPDF2.
    The verifier uses this as the recall baseline, so we record the engine.
    """
    try:
        from pdftext.extraction import plain_text_output  # type: ignore
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(pdf_path)
        n = len(pdf)
        pdf.close()
        pages = []
        for i in range(n):
            pages.append(plain_text_output(pdf_path, page_range=[i]))
        return {"engine": "pdftext", "pages": pages}
    except Exception:
        pass

    try:
        import pdfplumber  # type: ignore

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for pg in pdf.pages:
                pages.append(pg.extract_text() or "")
        return {"engine": "pdfplumber", "pages": pages}
    except Exception:
        pass

    from PyPDF2 import PdfReader  # type: ignore

    reader = PdfReader(pdf_path)
    pages = [(pg.extract_text() or "") for pg in reader.pages]
    return {"engine": "pypdf2", "pages": pages}


# --------------------------------------------------------------------------- #
# 2) Ground-truth PDF tables, per page  (independent of Marker)
# --------------------------------------------------------------------------- #
def extract_pdf_tables_by_page(pdf_path: str) -> dict:
    """
    Return {"engine": ..., "pages": [[{rows, cols, cells}, ...], ...]}.
    Uses pdfplumber if available; degrades gracefully to empty.
    The verifier compares these against tables found in the markdown.
    """
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return {"engine": "none", "pages": [], "note": "pdfplumber not installed"}

    pages_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            page_tables = []
            try:
                for tbl in pg.extract_tables() or []:
                    rows = len(tbl)
                    cols = max((len(r) for r in tbl), default=0)
                    cells = [
                        ["" if c is None else str(c).strip() for c in row]
                        for row in tbl
                    ]
                    page_tables.append({"rows": rows, "cols": cols, "cells": cells})
            except Exception as e:  # pragma: no cover
                page_tables.append({"error": str(e)})
            pages_out.append(page_tables)
    return {"engine": "pdfplumber", "pages": pages_out}


# --------------------------------------------------------------------------- #
# 3) Block-type counts from Marker JSON tree
# --------------------------------------------------------------------------- #
def _walk_blocks(node, per_page, page_id, total):
    """Recursively count block_type, attributing to the current page."""
    bt = getattr(node, "block_type", None) or (
        node.get("block_type") if isinstance(node, dict) else None
    )
    if bt == "Page":
        nid = getattr(node, "id", None) or (
            node.get("id") if isinstance(node, dict) else None
        )
        if nid and "/page/" in str(nid):
            try:
                page_id = int(str(nid).split("/page/")[1].split("/")[0])
            except Exception:
                page_id = page_id
    if bt:
        total[bt] += 1
        per_page[page_id][bt] += 1
    children = getattr(node, "children", None)
    if children is None and isinstance(node, dict):
        children = node.get("children")
    for child in children or []:
        _walk_blocks(child, per_page, page_id, total)


def block_counts_from_json(json_rendered) -> dict:
    """
    json_rendered is the Marker JSON-renderer output (has .children = pages).
    Returns {"total": {block_type: n}, "per_page": {page_id: {block_type: n}}}.
    """
    total = Counter()
    per_page = defaultdict(Counter)
    pages = getattr(json_rendered, "children", None)
    if pages is None and isinstance(json_rendered, dict):
        pages = json_rendered.get("children")
    for pi, page in enumerate(pages or []):
        _walk_blocks(page, per_page, pi, total)
    return {
        "total": dict(total),
        "per_page": {str(k): dict(v) for k, v in sorted(per_page.items())},
    }


# --------------------------------------------------------------------------- #
# 4) Table bounding boxes from Marker JSON tree
# --------------------------------------------------------------------------- #
def extract_table_bboxes_from_json(json_tree) -> list:
    """
    Walks the Marker JSON tree to extract polygons (bounding boxes) for all
    Table blocks.  Returns a list of:
        {"page_index": <int>, "polygon": [[x,y], [x,y], [x,y], [x,y]]}
    Used by verifier and qa for vision-fallback cropping via tableimg.py.
    """
    tables = []

    def _walk(node, current_page):
        bt = (node.get("block_type") if isinstance(node, dict)
              else getattr(node, "block_type", None))

        if bt == "Page":
            nid = (node.get("id") if isinstance(node, dict)
                   else getattr(node, "id", None))
            if nid and "/page/" in str(nid):
                try:
                    current_page = int(str(nid).split("/page/")[1].split("/")[0])
                except Exception:
                    pass

        if bt == "Table":
            poly = (node.get("polygon") if isinstance(node, dict)
                    else getattr(node, "polygon", None))
            if poly:
                tables.append({"page_index": current_page, "polygon": poly})

        children = (node.get("children") if isinstance(node, dict)
                    else getattr(node, "children", None))
        for child in children or []:
            _walk(child, current_page)

    _walk(json_tree, current_page=0)
    return tables


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="PDF->Markdown parser with verification bundle")
    ap.add_argument("--input",  default="hsbc.pdf",      help="Input PDF path")
    ap.add_argument("--output", default="./hsbc_output", help="Output directory")
    ap.add_argument("--stability-rerun", action="store_true",
                    help="Run markdown conversion twice for rerun-stability check")
    args = ap.parse_args()

    input_filename  = args.input
    base_name       = os.path.splitext(os.path.basename(input_filename))[0]
    output_directory = args.output
    verif_dir       = os.path.join(output_directory, "verification")
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(verif_dir, exist_ok=True)

    # Build models once and reuse across converters (markdown + json).
    models = create_model_dict()

    # ---- (a) Markdown conversion (primary deliverable) -------------------- #
    md_converter = make_converter("markdown", models)
    rendered_md  = md_converter(input_filename)
    save_output(rendered_md, output_directory, base_name)  # writes .md, meta, images
    markdown_text, _, _images = text_from_rendered(rendered_md)
    md_path = os.path.join(output_directory, f"{base_name}.md")
    write_text(md_path, markdown_text)

    # ---- (b) Marker metadata (TOC + page_stats/block_counts) -------------- #
    meta      = rendered_md.metadata
    meta_dict = meta if isinstance(meta, dict) else getattr(meta, "__dict__", {})
    meta_path = os.path.join(output_directory, f"{base_name}_meta.json")
    write_json(meta_path, meta_dict)

    # ---- (c) JSON block tree (structural truth from Marker) --------------- #
    json_converter = make_converter("json", models)
    rendered_json  = json_converter(input_filename)
    try:
        json_tree = rendered_json.model_dump(mode="json")
    except Exception:
        json_tree = json.loads(
            json.dumps(rendered_json, default=lambda o: getattr(o, "__dict__", str(o)))
        )
    json_path = os.path.join(output_directory, f"{base_name}.json")
    write_json(json_path, json_tree)

    md_block_counts = block_counts_from_json(rendered_json)
    bc_path = os.path.join(verif_dir, "md_block_counts.json")
    write_json(bc_path, md_block_counts)

    # ---- (c2) Table bounding boxes (spatial index for vision fallback) ---- #
    table_bboxes = extract_table_bboxes_from_json(json_tree)
    bbox_path    = os.path.join(verif_dir, "table_bboxes.json")
    write_json(bbox_path, table_bboxes)

    # ---- (d) Ground-truth PDF text per page ------------------------------- #
    pdf_text      = extract_pdf_text_by_page(input_filename)
    pdf_text_path = os.path.join(verif_dir, "pdf_text_by_page.json")
    write_json(pdf_text_path, pdf_text)

    # ---- (e) Ground-truth PDF tables per page ----------------------------- #
    pdf_tables      = extract_pdf_tables_by_page(input_filename)
    pdf_tables_path = os.path.join(verif_dir, "pdf_tables_by_page.json")
    write_json(pdf_tables_path, pdf_tables)

    # ---- (f) Optional stability rerun ------------------------------------- #
    stability_path = None
    if args.stability_rerun:
        rerun_converter = make_converter("markdown", models)
        rerun_rendered  = rerun_converter(input_filename)
        rerun_text, _, _ = text_from_rendered(rerun_rendered)
        stability_path  = os.path.join(verif_dir, "stability", f"{base_name}_run2.md")
        write_text(stability_path, rerun_text)

    # ---- (g) Manifest: the contract the verifier reads -------------------- #
    #
    # page_images is produced by tableimg.py, not by parser.py itself.
    # We record its expected location so verifier.py and qa.py can discover
    # the PNGs without knowing where tableimg.py puts them.
    page_images_dir = os.path.join(verif_dir, "page_images")

    manifest = {
        "schema_version": "1.1",
        "source_pdf": {
            "path":      os.path.abspath(input_filename),
            "sha256":    sha256_file(input_filename),
            "base_name": base_name,
        },
        "marker_config": {
            "use_llm":         True,
            "force_ocr":       True,
            "deployment_name": "gpt-4o-mini",
            "output_formats":  ["markdown", "json"],
        },
        "page_count_from_meta": len(meta_dict.get("page_stats", []) or []),
        "artifacts": {
            "markdown": {
                "path":   os.path.relpath(md_path, output_directory),
                "sha256": sha256_text(markdown_text),
            },
            "marker_metadata": {
                "path": os.path.relpath(meta_path, output_directory),
            },
            "marker_json_tree": {
                "path": os.path.relpath(json_path, output_directory),
            },
            "md_block_counts": {
                "path": os.path.relpath(bc_path, output_directory),
            },
            # FIX 1: table_bboxes is now registered in the manifest so
            # verifier.py can load it via load_bundle() without hardcoding paths.
            "table_bboxes": {
                "path":  os.path.relpath(bbox_path, output_directory),
                "count": len(table_bboxes),
            },
            "pdf_text_by_page": {
                "path":   os.path.relpath(pdf_text_path, output_directory),
                "engine": pdf_text.get("engine"),
            },
            "pdf_tables_by_page": {
                "path":   os.path.relpath(pdf_tables_path, output_directory),
                "engine": pdf_tables.get("engine"),
            },
            # FIX 2: page_images directory registered so verifier/qa can locate
            # PNG crops for vision fallback.  The directory may not exist yet
            # (tableimg.py must be run separately), so we record its expected
            # path and a note rather than failing if it is absent.
            "page_images": {
                "path": os.path.relpath(page_images_dir, output_directory),
                "note": (
                    "Generated by tableimg.py.  "
                    "Run: python tableimg.py --input <pdf> "
                    f"--output {page_images_dir}"
                ),
            },
            "stability_run2": (
                {"path": os.path.relpath(stability_path, output_directory)}
                if stability_path else None
            ),
        },
    }
    manifest_path = os.path.join(verif_dir, "manifest.json")
    write_json(manifest_path, manifest)

    # ---- Console summary -------------------------------------------------- #
    print(markdown_text[:1500])
    print("\n--- block_type totals (from Marker JSON) ---")
    print(json.dumps(md_block_counts["total"], indent=2, ensure_ascii=False))
    print(f"\n[OK] Markdown            -> {md_path}")
    print(f"[OK] Marker metadata     -> {meta_path}")
    print(f"[OK] Marker JSON tree    -> {json_path}")
    print(f"[OK] Table bboxes        -> {bbox_path}  ({len(table_bboxes)} tables)")
    print(f"[OK] Verification bundle -> {verif_dir}")
    print(f"[OK] Manifest            -> {manifest_path}")
    print(f"\n[NOTE] Page images not generated by parser.py.")
    print(f"       Run separately:  python tableimg.py --input {input_filename} "
          f"--output {page_images_dir}")
    print("\nNext:  python verifier.py --bundle "
          f"{os.path.join(output_directory, 'verification', 'manifest.json')}")


if __name__ == "__main__":
    freeze_support()
    main()