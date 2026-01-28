"""Materialize structured JSON alongside crawled Markdown pages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from ..config import ExtractConfig

_METADATA_PATTERN = re.compile(r"<!--\s*(?P<key>[^:]+):\s*(?P<value>.*?)\s*-->")

@dataclass
class ExtractedPage:
    url: str
    title: str
    elements: List[dict]

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "title": self.title,
            "elements": self.elements,
        }

def _load_markdown(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _parse_metadata(markdown: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for match in _METADATA_PATTERN.finditer(markdown):
        key = match.group("key").strip().lower()
        value = match.group("value").strip()
        metadata[key] = value
    return metadata


def _derive_title(markdown: str, elements: Iterable) -> str:
    # Try to use structured title from unstructured output
    for element in elements:
        category = getattr(element, "category", None)
        text = getattr(element, "text", "") or ""
        if category == "Title" and text.strip():
            return text.strip()

    # Fallback to first Markdown heading
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()

    return ""


def _build_page(path: Path, *, verbose: bool = False) -> Optional[ExtractedPage]:
    markdown_content = _load_markdown(path)
    metadata = _parse_metadata(markdown_content)

    elements_raw: List = []
    if markdown_content.strip():
        try:
            elements_raw = list(partition(filename=str(path)))
        except Exception as exc:  # pragma: no cover - depends on external parser
            if verbose:
                print(f"Failed to parse structured content for {path}: {exc}")

    elements = elements_raw
    title = _derive_title(markdown_content, elements)
    element_dicts = [element.to_dict() for element in elements]

    url = metadata.get("source", "")

    if not title:
        title = path.stem.replace("-", " ").replace("_", " ").strip().title()

    return ExtractedPage(url=url, title=title, elements=element_dicts)


def _discover_markdown_files(content_dir: Path) -> List[Path]:
    return sorted(content_dir.rglob("*.md"))


def extract_content(content_dir: Path, *, verbose: bool = False) -> List[Path]:
    markdown_files = _discover_markdown_files(content_dir)
    if not markdown_files:
        print(f"No markdown files found in {content_dir}")
        return []

    output_paths: List[Path] = []
    for path in markdown_files:
        page = _build_page(path, verbose=verbose)
        if not page:
            continue

        json_path = path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(page.to_dict(), handle, ensure_ascii=False, indent=2)

        output_paths.append(json_path)

        if verbose:
            print(f"Extracted structured content to {json_path}")

    if verbose:
        print(f"Processed {len(output_paths)} markdown files in {content_dir}")

    return output_paths


def handle(
    content_dir: Path,
    output_folder: str,
    config: "ExtractConfig | None" = None,
    *,
    debug: bool = False,
) -> List[Path]:
    """Main entry point for structured content extraction.

    Args:
        content_dir: Directory containing Markdown files.
        output_folder: Directory for output files (unused, JSON written alongside Markdown).
        config: ExtractConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        List of paths to extracted JSON files.
    """
    if not content_dir.exists() or not content_dir.is_dir():
        if debug:
            print(f"Content directory not found: {content_dir}")
        return []

    return extract_content(content_dir, verbose=debug)
