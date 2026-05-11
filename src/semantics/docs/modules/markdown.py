"""Markdown export using PyMuPDF4LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf4llm

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


def handle(
    input_file: str,
    output_folder: str,
    config: "MarkdownConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Convert a document to Markdown format.

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

    # Create markdown subfolder
    markdown_dir = os.path.join(output_folder, "markdown")
    os.makedirs(markdown_dir, exist_ok=True)

    # Build kwargs for to_markdown
    kwargs: dict = {
        "dpi": dpi,
    }

    if include_images:
        images_dir = os.path.join(markdown_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        kwargs["write_images"] = True
        kwargs["image_path"] = images_dir

    md_text = pymupdf4llm.to_markdown(input_file, **kwargs)

    # Fix image paths to be relative to the markdown file
    if include_images:
        md_text = _rewrite_image_paths(md_text, images_dir)

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
