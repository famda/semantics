"""Structured document extraction using Unstructured and PyMuPDF."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import pymupdf
from unstructured.partition.auto import partition

from .utils.logging import debug_print, info_print

if TYPE_CHECKING:
    from ..config import StructuredConfig

__all__ = ["handle"]


def _partition_document(input_file: str, *, debug: bool = False) -> List:
    """Run unstructured partition on the input document."""
    try:
        elements = list(partition(filename=input_file))
    except Exception as exc:
        if debug:
            debug_print(f"Failed to partition document: {exc}", debug=debug)
        raise
    return elements


def _extract_pymupdf_assets(input_file: str, *, min_image_size: int = 100, debug: bool = False) -> dict:
    """Use PyMuPDF to detect images and tables per page.

    Returns a dict keyed by 1-based page number, each value having
    ``images`` (list of image metadata dicts) and ``tables`` (list of
    table dicts with ``html`` and ``text`` keys).
    """
    pages: dict[int, dict] = {}

    suffix = Path(input_file).suffix.lower()
    if suffix != ".pdf":
        debug_print(f"PyMuPDF enrichment skipped for non-PDF ({suffix})", debug=debug)
        return pages

    doc = pymupdf.open(input_file)
    seen_xrefs: set[int] = set()

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pno = page_num + 1
            page_data: dict = {"images": [], "tables": []}

            # --- images ---
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    img_data = doc.extract_image(xref)
                except Exception:
                    continue
                if not img_data or not img_data.get("image"):
                    continue
                w, h = img_data.get("width", 0), img_data.get("height", 0)
                if w < min_image_size or h < min_image_size:
                    continue
                # Get bounding box on page
                bbox = None
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        r = rects[0]
                        bbox = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
                except Exception:
                    pass
                page_data["images"].append({
                    "xref": xref,
                    "width": w,
                    "height": h,
                    "bbox": bbox,
                    "format": img_data.get("ext", "png"),
                })

            # --- tables ---
            try:
                tab_finder = page.find_tables()
                for tab_idx, table in enumerate(tab_finder.tables):
                    html_parts = ["<table>"]
                    text_rows = []
                    try:
                        extracted = table.extract()
                    except Exception:
                        continue
                    for row in extracted:
                        cells = [c if c is not None else "" for c in row]
                        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
                        text_rows.append(" | ".join(cells))
                    html_parts.append("</table>")
                    page_data["tables"].append({
                        "table_index": tab_idx + 1,
                        "rows": len(extracted),
                        "cols": len(extracted[0]) if extracted else 0,
                        "text": "\n".join(text_rows),
                        "text_as_html": "\n".join(html_parts),
                    })
            except Exception:
                debug_print(f"Table detection failed on page {pno}", debug=debug)

            if page_data["images"] or page_data["tables"]:
                pages[pno] = page_data
    finally:
        doc.close()

    total_images = sum(len(p["images"]) for p in pages.values())
    total_tables = sum(len(p["tables"]) for p in pages.values())
    debug_print(
        f"PyMuPDF found {total_images} image(s) and {total_tables} table(s)",
        debug=debug,
    )
    return pages


def _build_enriched_elements(
    elements: List,
    pymupdf_assets: dict,
    *,
    include_metadata: bool = True,
) -> list[dict]:
    """Merge unstructured elements with PyMuPDF image/table data.

    Images and tables from PyMuPDF are injected into the element list
    at the end of the page where they were detected, preserving reading
    order as much as possible.  Duplicate entries are avoided: if
    unstructured already produced a ``Table`` or ``Image`` element on the
    same page, the PyMuPDF version is skipped.
    """
    element_dicts: list[dict] = []
    # Track which pages already have Image/Table elements from unstructured
    pages_with_images: set[int] = set()
    pages_with_tables: set[int] = set()

    for el in elements:
        entry = el.to_dict()
        if not include_metadata:
            entry.pop("metadata", None)
        pno = None
        if include_metadata:
            pno = (entry.get("metadata") or {}).get("page_number")
        if entry.get("type") == "Image" and pno:
            pages_with_images.add(pno)
        if entry.get("type") == "Table" and pno:
            pages_with_tables.add(pno)
        element_dicts.append(entry)

    if not pymupdf_assets:
        return element_dicts

    # Group element indices by page for insertion
    last_index_per_page: dict[int, int] = {}
    for idx, entry in enumerate(element_dicts):
        pno = (entry.get("metadata") or {}).get("page_number")
        if pno is not None:
            last_index_per_page[pno] = idx

    # Collect new entries to insert (sorted by page, descending so inserts don't shift indices)
    inserts: list[tuple[int, dict]] = []

    for pno in sorted(pymupdf_assets.keys()):
        assets = pymupdf_assets[pno]
        insert_at = last_index_per_page.get(pno)
        if insert_at is None:
            insert_at = len(element_dicts)
        else:
            insert_at += 1  # insert after the last element on this page

        # Add images if unstructured didn't detect any on this page
        if pno not in pages_with_images:
            for img_counter, img in enumerate(assets.get("images", []), start=1):
                filename = f"{pno:08d}_{img_counter:03d}.{img['format']}"
                img_metadata = {
                    "page_number": pno,
                    "image_width": img["width"],
                    "image_height": img["height"],
                    "image_format": img["format"],
                    "image_path": f"images/{filename}",
                    "source": "pymupdf",
                }
                if img.get("bbox") is not None:
                    img_metadata["image_bbox"] = img["bbox"]
                inserts.append((insert_at, {
                    "type": "Image",
                    "element_id": f"pymupdf_img_p{pno}_{img_counter}",
                    "text": "",
                    "metadata": img_metadata,
                }))
                insert_at += 1

        # Add tables if unstructured didn't detect any on this page
        if pno not in pages_with_tables:
            for tab in assets.get("tables", []):
                inserts.append((insert_at, {
                    "type": "Table",
                    "element_id": f"pymupdf_tab_p{pno}_{tab['table_index']}",
                    "text": tab["text"],
                    "metadata": {
                        "page_number": pno,
                        "text_as_html": tab["text_as_html"],
                        "table_rows": tab["rows"],
                        "table_cols": tab["cols"],
                        "source": "pymupdf",
                    },
                }))
                insert_at += 1

    # Insert in reverse order to preserve indices
    for pos, entry in reversed(inserts):
        element_dicts.insert(pos, entry)

    return element_dicts


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
    debug_print(f"Partitioning document: {input_path.name}", debug=debug)

    elements = _partition_document(input_file, debug=debug)

    # Use PyMuPDF to discover images and tables that unstructured may miss
    pymupdf_assets = _extract_pymupdf_assets(input_file, debug=debug)

    # Build enriched element list
    element_dicts = _build_enriched_elements(
        elements, pymupdf_assets, include_metadata=include_metadata
    )

    # Count by type
    type_counts: dict[str, int] = {}
    for e in element_dicts:
        t = e.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Build output structure
    result = {
        "source": str(input_path.name),
        "total_elements": len(element_dicts),
        "element_types": type_counts,
        "elements": element_dicts,
    }

    # Write JSON output
    structured_dir = os.path.join(output_folder, "structured")
    os.makedirs(structured_dir, exist_ok=True)
    output_filename = "structure.json"
    output_path = os.path.join(structured_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as handle_file:
        json.dump(result, handle_file, ensure_ascii=False, indent=2)

    info_print(f"Extracted {len(element_dicts)} elements → {output_filename}")
    if type_counts.get("Image") or type_counts.get("Table"):
        img_count = type_counts.get("Image", 0)
        tab_count = type_counts.get("Table", 0)
        info_print(f"  Including {img_count} image(s) and {tab_count} table(s)")
    debug_print(f"Output written to {output_path}", debug=debug)

    return result
