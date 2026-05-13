"""Table extraction from documents using PyMuPDF and Docling."""

from __future__ import annotations

import csv
import io
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, List

import pymupdf

from .utils.logging import debug_print, info_print

if TYPE_CHECKING:
    from ..config import TablesConfig

__all__ = ["handle"]


class _HTMLTableParser(HTMLParser):
    """Minimal HTML table parser that extracts rows and cells."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs):
        if tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag: str):
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append("".join(self._current_cell).strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data: str):
        if self._in_cell:
            self._current_cell.append(data)


def _html_table_to_csv(html: str) -> str:
    """Convert an HTML table string to CSV format."""
    parser = _HTMLTableParser()
    parser.feed(html)

    output = io.StringIO()
    writer = csv.writer(output)
    for row in parser.rows:
        writer.writerow(row)
    return output.getvalue()


def _extract_tables_pymupdf(input_file: str, *, debug: bool = False) -> list[dict]:
    """Use PyMuPDF to detect and extract tables from a PDF."""
    tables: list[dict] = []
    doc = pymupdf.open(input_file)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pno = page_num + 1
            try:
                tab_finder = page.find_tables()
            except Exception:
                debug_print(f"Table detection failed on page {pno}", debug=debug)
                continue
            for tab_idx, table in enumerate(tab_finder.tables):
                try:
                    extracted = table.extract()
                except Exception:
                    continue
                if not extracted:
                    continue

                # Build HTML
                html_parts = ["<table>"]
                text_rows = []
                for row in extracted:
                    cells = [c if c is not None else "" for c in row]
                    html_parts.append(
                        "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                    )
                    text_rows.append(" | ".join(cells))
                html_parts.append("</table>")

                tables.append({
                    "page": pno,
                    "text": "\n".join(text_rows),
                    "text_as_html": "\n".join(html_parts),
                    "rows": len(extracted),
                    "cols": len(extracted[0]) if extracted else 0,
                    "bbox": list(table.bbox) if hasattr(table, "bbox") else None,
                })
    finally:
        doc.close()

    debug_print(f"PyMuPDF found {len(tables)} table(s)", debug=debug)
    return tables


def _extract_tables_docling(input_file: str, *, debug: bool = False) -> list[dict]:
    """Use Docling to extract tables from any supported format."""
    from .utils.converter import convert_document

    result = convert_document(input_file, debug=debug)
    doc = result.document

    tables: list[dict] = []
    for item in getattr(doc, "tables", []):
        prov = getattr(item, "prov", None)

        # Page number
        page_no = None
        if prov:
            p = prov[0] if isinstance(prov, list) else prov
            raw = getattr(p, "page_no", None)
            if raw is not None:
                page_no = int(raw) + 1

        # Try DataFrame export for CSV text
        text = ""
        rows_count = 0
        cols_count = 0
        try:
            df = item.export_to_dataframe(doc=doc)
            text = df.to_csv(index=False)
            rows_count = len(df)
            cols_count = len(df.columns)
        except Exception:
            text = getattr(item, "text", "") or ""

        # Try HTML export
        html = ""
        try:
            html = item.export_to_html(doc=doc)
        except Exception:
            pass

        # Bounding box
        bbox = None
        if prov:
            p = prov[0] if isinstance(prov, list) else prov
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

        tables.append({
            "page": page_no,
            "text": text,
            "text_as_html": html,
            "rows": rows_count,
            "cols": cols_count,
            "bbox": bbox,
        })

    debug_print(f"Docling found {len(tables)} table(s)", debug=debug)
    return tables


def handle(
    input_file: str,
    output_folder: str,
    config: "TablesConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Extract tables from a document and export as CSV/HTML.

    Uses PyMuPDF for PDF files (better table detection) and falls back
    to unstructured for other document types.

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: TablesConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Result dictionary with extraction metadata.
    """
    input_path = Path(input_file)
    tables_dir = os.path.join(output_folder, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    debug_print(f"Extracting tables from: {input_path.name}", debug=debug)

    # Always prefer PyMuPDF via the converted PDF (better table detection + page/bbox)
    from .utils.converter import get_pdf_path
    effective_path = get_pdf_path(input_file)

    if effective_path.lower().endswith(".pdf"):
        table_data = _extract_tables_pymupdf(effective_path, debug=debug)
    else:
        table_data = _extract_tables_docling(input_file, debug=debug)

    exported: list[dict] = []

    for idx, table in enumerate(table_data, start=1):
        table_id = f"table_{idx:03d}"
        html_content = table.get("text_as_html", "")
        text_content = table.get("text", "")

        entry = {
            "id": table_id,
            "page": table.get("page"),
            "rows": table.get("rows"),
            "cols": table.get("cols"),
            "bbox": table.get("bbox"),
            "text": text_content[:200],  # truncated preview
        }

        # Write HTML version
        if html_content:
            html_path = os.path.join(tables_dir, f"{table_id}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            entry["html_file"] = f"{table_id}.html"

            # Convert to CSV
            csv_content = _html_table_to_csv(html_content)
            if csv_content.strip():
                csv_path = os.path.join(tables_dir, f"{table_id}.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    f.write(csv_content)
                entry["csv_file"] = f"{table_id}.csv"
        else:
            # Fallback: write raw text as a single-cell CSV
            csv_path = os.path.join(tables_dir, f"{table_id}.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for line in text_content.split("\n"):
                    writer.writerow([line.strip()])
            entry["csv_file"] = f"{table_id}.csv"

        exported.append(entry)
        debug_print(f"Exported {table_id} (page {entry['page']})", debug=debug)

    result = {
        "source": input_path.name,
        "total_tables": len(exported),
        "tables": exported,
    }

    # Write tables manifest
    manifest_path = os.path.join(tables_dir, "tables.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    info_print(f"Extracted {len(exported)} table(s) → tables/")
    return result
