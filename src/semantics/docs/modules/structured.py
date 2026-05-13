"""Structured document extraction using Docling.

Converts any supported document format (PDF, DOCX, PPTX, XLSX, HTML,
images, etc.) into a unified ``structure.json`` with elements, bounding
boxes, and page numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .utils.logging import debug_print, info_print

if TYPE_CHECKING:
    from ..config import StructuredConfig

__all__ = ["handle"]


def _bbox_from_prov(prov) -> list[float] | None:
    """Extract [x0, y0, x1, y1] from a docling provenance entry."""
    if not prov:
        return None
    p = prov[0] if isinstance(prov, list) else prov
    bbox_obj = getattr(p, "bbox", None)
    if bbox_obj is None:
        return None
    try:
        coords = [bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b]
        return [round(c, 2) for c in coords]
    except Exception:
        return None


def _page_from_prov(prov) -> int | None:
    """Extract 1-based page number from a docling provenance entry."""
    if not prov:
        return None
    p = prov[0] if isinstance(prov, list) else prov
    page_no = getattr(p, "page_no", None)
    if page_no is not None:
        return int(page_no) + 1  # docling uses 0-based
    return None


def _build_elements(doc) -> list[dict]:
    """Walk DoclingDocument and produce a flat element list."""
    elements: list[dict] = []

    # --- Text items ---
    for item in getattr(doc, "texts", []):
        label = getattr(item, "label", "text")
        prov = getattr(item, "prov", None)
        elements.append({
            "type": str(label).capitalize() if label else "Text",
            "text": getattr(item, "text", "") or "",
            "metadata": {
                "page_number": _page_from_prov(prov),
                "bbox": _bbox_from_prov(prov),
            },
        })

    # --- Table items ---
    for item in getattr(doc, "tables", []):
        prov = getattr(item, "prov", None)
        # Try to get text/html representations
        text = ""
        html = ""
        try:
            data = item.export_to_dataframe(doc=doc)
            text = data.to_csv(index=False)
        except Exception:
            text = getattr(item, "text", "") or ""
        try:
            html = item.export_to_html(doc=doc)
        except Exception:
            pass

        elements.append({
            "type": "Table",
            "text": text,
            "metadata": {
                "page_number": _page_from_prov(prov),
                "bbox": _bbox_from_prov(prov),
                "text_as_html": html,
            },
        })

    # --- Picture items ---
    for item in getattr(doc, "pictures", []):
        prov = getattr(item, "prov", None)
        elements.append({
            "type": "Image",
            "text": getattr(item, "text", "") or "",
            "metadata": {
                "page_number": _page_from_prov(prov),
                "bbox": _bbox_from_prov(prov),
            },
        })

    # --- Key-value items ---
    for item in getattr(doc, "key_value_items", []):
        prov = getattr(item, "prov", None)
        elements.append({
            "type": "KeyValue",
            "text": getattr(item, "text", "") or "",
            "metadata": {
                "page_number": _page_from_prov(prov),
                "bbox": _bbox_from_prov(prov),
            },
        })

    # Sort by page number, then by vertical position (top of bbox)
    def _sort_key(el):
        meta = el.get("metadata") or {}
        page = meta.get("page_number") or 0
        bbox = meta.get("bbox") or [0, 0, 0, 0]
        return (page, bbox[1])

    elements.sort(key=_sort_key)
    return elements


def handle(
    input_file: str,
    output_folder: str,
    config: "StructuredConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Extract structured content from a document.

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: StructuredConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Result dictionary with extraction metadata.
    """
    include_metadata = config.include_metadata if config else True

    input_path = Path(input_file)
    debug_print(f"Converting document: {input_path.name}", debug=debug)

    from .utils.converter import convert_document

    result = convert_document(input_file, debug=debug)
    doc = result.document

    elements = _build_elements(doc)

    # Strip metadata if not requested
    if not include_metadata:
        for el in elements:
            el.pop("metadata", None)

    # Count by type
    type_counts: dict[str, int] = {}
    for e in elements:
        t = e.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    output_data = {
        "source": str(input_path.name),
        "total_elements": len(elements),
        "element_types": type_counts,
        "elements": elements,
    }

    # Write JSON output
    structured_dir = os.path.join(output_folder, "structured")
    os.makedirs(structured_dir, exist_ok=True)
    output_filename = "structure.json"
    output_path = os.path.join(structured_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    info_print(f"Extracted {len(elements)} elements → {output_filename}")
    if type_counts.get("Image") or type_counts.get("Table"):
        img_count = type_counts.get("Image", 0)
        tab_count = type_counts.get("Table", 0)
        info_print(f"  Including {img_count} image(s) and {tab_count} table(s)")
    debug_print(f"Output written to {output_path}", debug=debug)

    return output_data
