"""Image extraction from documents using PyMuPDF (PDF) and Docling (other formats)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from .utils.logging import debug_print, info_print

if TYPE_CHECKING:
    from ..config import ImagesConfig

__all__ = ["handle"]


def _extract_images_from_page(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    page_num: int,
    output_dir: str,
    *,
    seen_xrefs: set,
    min_size: int,
    debug: bool,
) -> list[dict]:
    """Extract all images from a single page, skipping duplicates and tiny images."""
    extracted = []
    img_list = page.get_images(full=True)
    pno = page_num + 1
    page_img_counter = 0

    for img_info in img_list:
        xref = img_info[0]
        smask = img_info[1]

        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            img_data = doc.extract_image(xref)
        except Exception:
            debug_print(f"Failed to extract image xref={xref}", debug=debug)
            continue

        if not img_data or not img_data.get("image"):
            continue

        width = img_data.get("width", 0)
        height = img_data.get("height", 0)

        if width < min_size or height < min_size:
            debug_print(
                f"Skipping small image xref={xref} ({width}x{height})", debug=debug
            )
            continue

        ext = img_data.get("ext", "png")
        image_bytes = img_data["image"]

        # Handle image masks (alpha channel)
        if smask > 0:
            try:
                pix1 = pymupdf.Pixmap(image_bytes)
                mask_data = doc.extract_image(smask)
                if mask_data and mask_data.get("image"):
                    mask_pix = pymupdf.Pixmap(mask_data["image"])
                    pix = pymupdf.Pixmap(pix1, mask_pix)
                    image_bytes = pix.tobytes(ext)
            except Exception:
                debug_print(
                    f"Failed to apply mask for xref={xref}, using raw image",
                    debug=debug,
                )

        # Get bounding box on page
        bbox = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                bbox = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
        except Exception:
            debug_print(f"Could not get bbox for xref={xref}", debug=debug)

        page_img_counter += 1
        filename = f"{pno:08d}_{page_img_counter:03d}.{ext}"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        entry = {
            "filename": filename,
            "page": pno,
            "index": page_img_counter,
            "width": width,
            "height": height,
            "bbox": bbox,
            "format": ext,
            "xref": xref,
        }
        extracted.append(entry)
        debug_print(
            f"Extracted {filename} ({width}x{height}, {ext})", debug=debug
        )

    return extracted


def _extract_images_pymupdf(
    input_file: str,
    images_dir: str,
    *,
    min_size: int,
    debug: bool,
) -> list[dict]:
    """Extract images from a PDF using PyMuPDF."""
    doc = pymupdf.open(input_file)
    all_images: list[dict] = []
    seen_xrefs: set = set()

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = _extract_images_from_page(
                doc,
                page,
                page_num,
                images_dir,
                seen_xrefs=seen_xrefs,
                min_size=min_size,
                debug=debug,
            )
            all_images.extend(page_images)
    finally:
        doc.close()

    return all_images


def _extract_images_docling(
    input_file: str,
    images_dir: str,
    *,
    min_size: int,
    debug: bool,
) -> list[dict]:
    """Extract images from a non-PDF document using Docling."""
    from .utils.converter import convert_document

    result = convert_document(input_file, debug=debug)
    doc = result.document

    all_images: list[dict] = []
    img_counter = 0

    for item in getattr(doc, "pictures", []):
        prov = getattr(item, "prov", None)
        image_obj = getattr(item, "image", None)
        if image_obj is None:
            continue

        pil_image = getattr(image_obj, "pil_image", None)
        if pil_image is None:
            continue

        width, height = pil_image.size
        if width < min_size or height < min_size:
            debug_print(
                f"Skipping small docling image ({width}x{height})", debug=debug
            )
            continue

        # Determine page
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

        img_counter += 1
        pno = page_no or 1
        filename = f"{pno:08d}_{img_counter:03d}.png"
        filepath = os.path.join(images_dir, filename)
        pil_image.save(filepath, format="PNG")

        all_images.append({
            "filename": filename,
            "page": page_no,
            "index": img_counter,
            "width": width,
            "height": height,
            "bbox": bbox,
            "format": "png",
        })
        debug_print(f"Extracted {filename} ({width}x{height})", debug=debug)

    return all_images


def handle(
    input_file: str,
    output_folder: str,
    config: "ImagesConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Extract images from a document.

    Uses PyMuPDF for PDF files (best raw image quality) and Docling
    for all other supported formats.

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: ImagesConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Result dictionary with extraction metadata.
    """
    min_size = config.min_size if config else 100

    input_path = Path(input_file)
    images_dir = os.path.join(output_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    debug_print(f"Extracting images from: {input_path.name}", debug=debug)

    # Use PyMuPDF for native PDFs (best quality + page/bbox).
    # For non-PDF originals, use Docling which provides page/bbox via
    # the internal PDF conversion.  LibreOffice-converted PDFs embed
    # images in a way PyMuPDF cannot resolve positions for.
    if input_path.suffix.lower() == ".pdf":
        all_images = _extract_images_pymupdf(
            str(input_path), images_dir, min_size=min_size, debug=debug
        )
    else:
        all_images = _extract_images_docling(
            input_file, images_dir, min_size=min_size, debug=debug
        )

    # Write manifest
    manifest = {
        "source": input_path.name,
        "total_images": len(all_images),
        "images": all_images,
    }
    manifest_path = os.path.join(images_dir, "images.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    info_print(f"Extracted {len(all_images)} images → images/")
    debug_print(f"Manifest written to {manifest_path}", debug=debug)

    return manifest
