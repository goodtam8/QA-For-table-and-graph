import os
from pathlib import Path
import pymupdf4llm


def main():
    input_filename = "hsbc.pdf"
    base_name = Path(input_filename).stem
    output_directory = Path("./hsbc_output")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_md = output_directory / f"{base_name}.md"

    # Convert PDF to Markdown
    markdown_text = pymupdf4llm.to_markdown(
        input_filename,
        write_images=True,   # optional: extract images and reference them in markdown
    )

    # Save markdown file
    output_md.write_text(markdown_text, encoding="utf-8")

    # Optional preview
    print(markdown_text[:2000])
    print(f"\n✅ Output successfully saved to: {output_md}")


if __name__ == "__main__":
    main()