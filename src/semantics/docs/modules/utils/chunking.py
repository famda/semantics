"""Text chunking and document-context extraction utilities.

Used by overview, classify, and captions modules to split long documents
into manageable pieces and to pull a short context snippet for prompts.
"""

from __future__ import annotations

import json
import os
from typing import Optional

__all__ = ["chunk_text", "extract_document_text", "extract_document_context"]


def extract_document_text(output_folder: str, *, max_chars: int = 0) -> str:
    """Read all text from *structure.json* produced by structured extraction.

    Args:
        output_folder: Output directory that contains ``structured/structure.json``.
        max_chars: Maximum characters to return (0 = unlimited).

    Returns:
        Concatenated text from the document elements.
    """
    structure_path = os.path.join(output_folder, "structured", "structure.json")
    if not os.path.exists(structure_path):
        return ""

    with open(structure_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    parts: list[str] = []
    total = 0
    for el in data.get("elements", []):
        text = (el.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if max_chars and total >= max_chars:
            break

    joined = "\n".join(parts)
    if max_chars:
        joined = joined[:max_chars]
    return joined


def extract_document_context(output_folder: str, *, max_chars: int = 500) -> str:
    """Return a short context snippet (title + opening paragraph).

    Useful for injecting into VLM prompts so the model understands the
    surrounding document when captioning images or classifying.
    """
    return extract_document_text(output_folder, max_chars=max_chars)


def chunk_text(text: str, *, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """Split *text* into chunks respecting paragraph boundaries.

    Each chunk will be at most *max_chars* characters.  An *overlap* of
    characters is carried over from the end of the previous chunk so the
    model has continuity context.

    Returns:
        List of text chunks.  A single-element list if the text fits.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 1  # +1 for the newline separator
        if current_len + para_len > max_chars and current:
            chunks.append("\n".join(current))
            # Keep overlap from the tail of the current chunk
            tail = "\n".join(current)
            overlap_text = tail[-overlap:] if len(tail) > overlap else tail
            current = [overlap_text]
            current_len = len(overlap_text)
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n".join(current))

    return chunks
