"""Markdown export using Docling (primary) and PyMuPDF4LLM (PDF fallback)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .utils.logging import debug_print, info_print

if TYPE_CHECKING:
    from ..config import MarkdownConfig

__all__ = ["handle"]


def _rewrite_image_paths(md_text: str, images_dir: str) -> str:
    """Rewrite absolute/relative image paths to relative paths (images/filename)."""
    abs_path = os.path.abspath(images_dir)
    # Try multiple representations that pymupdf4llm might produce
    for prefix in (
        abs_path.replace("\\", "/") + "/",
        abs_path + os.sep,
        os.path.relpath(images_dir).replace("\\", "/") + "/",
        os.path.relpath(images_dir) + os.sep,
    ):
        md_text = md_text.replace(prefix, "images/")
    return md_text


def _convert_with_docling(input_file: str, *, debug: bool = False) -> str:
    """Convert a document to markdown using Docling."""
    from .utils.converter import convert_document

    result = convert_document(input_file, debug=debug)
    doc = result.document
    return doc.export_to_markdown()


def _convert_with_pymupdf4llm(
    input_file: str,
    markdown_dir: str,
    *,
    include_images: bool,
    dpi: int,
    debug: bool,
) -> str:
    """Convert a PDF to markdown using pymupdf4llm."""
    import pymupdf4llm

    kwargs: dict = {"dpi": dpi}
    if include_images:
        images_dir = os.path.join(markdown_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        kwargs["write_images"] = True
        kwargs["image_path"] = images_dir

    md_text = pymupdf4llm.to_markdown(input_file, **kwargs)

    if include_images:
        md_text = _rewrite_image_paths(md_text, images_dir)

    return md_text


def handle(
    input_file: str,
    output_folder: str,
    config: "MarkdownConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Convert a document to Markdown format.

    Uses Docling as the primary converter for all formats.
    Falls back to pymupdf4llm for PDF files when Docling produces
    empty output.

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: MarkdownConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Result dictionary with conversion metadata.
    """
    include_images = config.include_images if config else True
    dpi = config.dpi if config else 150

    input_path = Path(input_file)
    debug_print(f"Converting to markdown: {input_path.name}", debug=debug)

    markdown_dir = os.path.join(output_folder, "markdown")
    os.makedirs(markdown_dir, exist_ok=True)

    # Always prefer pymupdf4llm via the converted PDF (best image handling)
    from .utils.converter import get_pdf_path
    effective_path = get_pdf_path(input_file)

    if effective_path.lower().endswith(".pdf"):
        md_text = _convert_with_pymupdf4llm(
            effective_path,
            markdown_dir,
            include_images=include_images,
            dpi=dpi,
            debug=debug,
        )
        # Fall back to Docling if pymupdf4llm produced empty output
        if not md_text.strip():
            debug_print("pymupdf4llm produced empty markdown, falling back to Docling", debug=debug)
            md_text = _convert_with_docling(input_file, debug=debug)
    else:
        md_text = _convert_with_docling(input_file, debug=debug)

        # Fall back message for non-PDF
        if not md_text.strip():
            debug_print(f"Docling produced empty markdown for {input_path.suffix}", debug=debug)

    # Write markdown output
    output_filename = f"{input_path.stem}.md"
    output_path = os.path.join(markdown_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Count pages (approximate from page separators or newlines)
    page_count = md_text.count("-----") + 1 if "-----" in md_text else 1

    result = {
        "source": input_path.name,
        "output_file": f"markdown/{output_filename}",
        "characters": len(md_text),
        "pages": page_count,
    }

    info_print(f"Converted to markdown → {output_filename}")
    debug_print(f"Output: {output_path} ({len(md_text)} chars)", debug=debug)

    return result
