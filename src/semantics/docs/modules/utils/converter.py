"""Shared Docling DocumentConverter singleton.

Provides a cached converter and conversion result so the document is
parsed once and consumed by all downstream modules.  Non-PDF files are
transparently converted to PDF via LibreOffice before Docling ingestion
so that full page-level provenance (page_no, bbox) is available for
every format.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult
    from docling.document_converter import DocumentConverter

_lock = threading.Lock()
_converter: Optional["DocumentConverter"] = None
_cache: dict[str, "ConversionResult"] = {}
_pdf_paths: dict[str, str] = {}
_PDF_TMP_DIR = "/tmp/docling_pdf"


def _get_converter(*, ocr: bool = True, debug: bool = False) -> "DocumentConverter":
    """Return a lazily-initialised DocumentConverter singleton."""
    global _converter
    if _converter is not None:
        return _converter

    with _lock:
        if _converter is not None:
            return _converter

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = ocr
        pdf_opts.do_table_structure = True
        pdf_opts.generate_picture_images = True

        # Try EasyOCR first, fall back to RapidOCR, then disable OCR
        if ocr:
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                import easyocr as _  # noqa: F401 — verify it's actually installed
                pdf_opts.ocr_options = EasyOcrOptions()
            except (ImportError, Exception):
                try:
                    from docling.datamodel.pipeline_options import RapidOcrOptions
                    pdf_opts.ocr_options = RapidOcrOptions()
                except (ImportError, Exception):
                    pdf_opts.do_ocr = False

        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            }
        )
        return _converter


def _ensure_pdf(input_file: str) -> str:
    """Convert *input_file* to PDF via LibreOffice if it is not already a PDF.

    Returns the path to the (possibly converted) PDF.  Falls back to the
    original file when LibreOffice is unavailable or conversion fails.
    """
    key = str(Path(input_file).resolve())
    if key in _pdf_paths:
        return _pdf_paths[key]

    if Path(input_file).suffix.lower() == ".pdf":
        _pdf_paths[key] = input_file
        return input_file

    os.makedirs(_PDF_TMP_DIR, exist_ok=True)

    try:
        proc = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", _PDF_TMP_DIR,
                input_file,
            ],
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0:
            _pdf_paths[key] = input_file
            return input_file

        pdf_name = Path(input_file).stem + ".pdf"
        pdf_path = os.path.join(_PDF_TMP_DIR, pdf_name)
        if os.path.isfile(pdf_path):
            _pdf_paths[key] = pdf_path
            return pdf_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    _pdf_paths[key] = input_file
    return input_file


def get_pdf_path(input_file: str) -> str:
    """Return the effective PDF path for *input_file*.

    Must be called after :func:`convert_document` (which triggers the
    conversion).  If no conversion was performed, returns the original
    path.
    """
    key = str(Path(input_file).resolve())
    return _pdf_paths.get(key, input_file)


def convert_document(
    input_file: str,
    *,
    ocr: bool = True,
    debug: bool = False,
) -> "ConversionResult":
    """Convert a document, caching the result by absolute path.

    Non-PDF documents are first converted to PDF via LibreOffice so
    that Docling can provide full page-level provenance.  All modules
    should call this instead of creating their own converter.
    """
    key = str(Path(input_file).resolve())
    if key in _cache:
        return _cache[key]

    # Convert to PDF first (no-op when already a PDF)
    effective_path = _ensure_pdf(input_file)

    converter = _get_converter(ocr=ocr, debug=debug)
    result = converter.convert(effective_path)
    _cache[key] = result
    return result


def _cleanup_temp_pdfs() -> None:
    """Remove temporary PDF files created by :func:`_ensure_pdf`."""
    if os.path.isdir(_PDF_TMP_DIR):
        shutil.rmtree(_PDF_TMP_DIR, ignore_errors=True)


def clear_cache() -> None:
    """Drop cached conversion results (useful between CLI runs in tests)."""
    _cache.clear()
    _pdf_paths.clear()
    _cleanup_temp_pdfs()
