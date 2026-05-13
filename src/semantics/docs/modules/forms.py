"""Form and key-value pair extraction using Docling.

Extracts key-value pairs from documents using the Docling document
structure. Useful for forms, invoices, and other structured documents.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .utils.logging import debug_print, info_print

if TYPE_CHECKING:
    from ..config import FormsConfig

__all__ = ["handle"]


def _extract_key_values(doc) -> list[dict]:
    """Extract key-value items from a DoclingDocument."""
    kv_items = []
    for item in getattr(doc, "key_value_items", []):
        prov = getattr(item, "prov", None)
        page_no = None
        bbox = None
        if prov:
            p = prov[0] if isinstance(prov, list) else prov
            raw_page = getattr(p, "page_no", None)
            if raw_page is not None:
                page_no = int(raw_page) + 1
            bbox_obj = getattr(p, "bbox", None)
            if bbox_obj is not None:
                try:
                    bbox = [
                        round(bbox_obj.l, 2),
                        round(bbox_obj.t, 2),
                        round(bbox_obj.r, 2),
                        round(bbox_obj.b, 2),
                    ]
                except Exception:
                    pass

        text = getattr(item, "text", "") or ""
        # Try to split on common key-value separators
        key, value = _split_key_value(text)

        kv_items.append({
            "key": key,
            "value": value,
            "raw_text": text,
            "page": page_no,
            "bbox": bbox,
        })

    return kv_items


def _split_key_value(text: str) -> tuple[str, str]:
    """Attempt to split text into key and value parts."""
    for sep in (":", "=", " - "):
        if sep in text:
            parts = text.split(sep, 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key:
                return key, value
    return text.strip(), ""


def _extract_table_fields(doc) -> list[dict]:
    """Extract form-like fields from two-column tables."""
    fields = []
    for item in getattr(doc, "tables", []):
        try:
            df = item.export_to_dataframe(doc=doc)
        except Exception:
            continue

        # Only process tables that look like form fields (2 columns)
        if len(df.columns) != 2:
            continue

        prov = getattr(item, "prov", None)
        page_no = None
        if prov:
            p = prov[0] if isinstance(prov, list) else prov
            raw_page = getattr(p, "page_no", None)
            if raw_page is not None:
                page_no = int(raw_page) + 1

        for _, row in df.iterrows():
            vals = list(row.values)
            key = str(vals[0]).strip() if vals[0] else ""
            value = str(vals[1]).strip() if len(vals) > 1 and vals[1] else ""
            if key and key.lower() not in ("", "nan", "none"):
                fields.append({
                    "key": key,
                    "value": value,
                    "raw_text": f"{key}: {value}",
                    "page": page_no,
                    "source": "table",
                })

    return fields


def handle(
    input_file: str,
    output_folder: str,
    config: "FormsConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Extract form fields and key-value pairs from a document.

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: FormsConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Dictionary with extracted key-value pairs.
    """
    info_print("Extracting form fields")

    input_path = Path(input_file)
    debug_print(f"Processing forms from: {input_path.name}", debug=debug)

    from .utils.converter import convert_document

    result = convert_document(input_file, debug=debug)
    doc = result.document

    # Extract key-value items from document structure
    kv_items = _extract_key_values(doc)
    debug_print(f"Found {len(kv_items)} key-value item(s) from document structure", debug=debug)

    # Also extract form-like fields from 2-column tables
    table_fields = _extract_table_fields(doc)
    debug_print(f"Found {len(table_fields)} field(s) from form tables", debug=debug)

    forms_dir = os.path.join(output_folder, "forms")
    os.makedirs(forms_dir, exist_ok=True)

    output_data = {
        "source": input_path.name,
        "total_fields": len(kv_items) + len(table_fields),
        "key_value_items": kv_items,
        "table_fields": table_fields,
    }

    output_path = os.path.join(forms_dir, "forms.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total = output_data["total_fields"]
    info_print(f"Extracted {total} form field(s) → forms/")
    debug_print(f"Results saved to {output_path}", debug=debug)

    return output_data
