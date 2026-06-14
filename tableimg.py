"""
pdf_renderer.py — Standalone PDF to Image renderer for Vision Fallback
"""
import os
import argparse
import pypdfium2 as pdfium

def export_page_images(pdf_path: str, output_dir: str, scale: float = 2.0) -> list:
    """
    Render each page of the PDF to a high-res PNG file.
    scale=2.0 gives ~144 DPI, which is ideal for Azure OpenAI Vision.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    image_paths = []
    
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        
        img_filename = f"page_{i+1:03d}.png"
        img_path = os.path.join(output_dir, img_filename)
        pil_image.save(img_path)
        image_paths.append(img_filename)
        
    pdf.close()
    return image_paths

def main():
    ap = argparse.ArgumentParser(description="Render PDF pages to images for Vision Fallback")
    ap.add_argument("--input", default="hsbc.pdf", help="Input PDF path")
    ap.add_argument("--output", default="./hsbc_output/verification/page_images", help="Output directory for images")
    args = ap.parse_args()
    
    images = export_page_images(args.input, args.output)
    print(f"[OK] Extracted {len(images)} page images to {args.output}")

if __name__ == "__main__":
    main()