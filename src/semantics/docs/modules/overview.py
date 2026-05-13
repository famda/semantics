"""Document overview generation using Qwen3-VL.

Reads the structured extraction output and produces a concise but
detailed overview of the document suitable for RAG pipelines,
semantic search, and agentic systems.

Outputs:
  - ``overview/overview.json`` — structured overview data
  - ``overview/overview.md``   — human-readable markdown summary
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .utils.chunking import chunk_text, extract_document_text
from .utils.logging import debug_print, gray_debug_output, info_print

if TYPE_CHECKING:
    from ..config import OverviewConfig

__all__ = ["handle"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CHUNK_PROMPT = (
    "Summarize the following document section concisely. "
    "Preserve key facts, named entities, technical details, relationships, "
    "and any numerical data. Do NOT add opinions, commentary, or information "
    "not present in the text. Write in third person.\n\n"
    "--- DOCUMENT SECTION ---\n{text}\n--- END ---\n\n"
    "Summary:"
)

_SYNTHESIZE_PROMPT = (
    "The following are summaries of consecutive sections of the same document. "
    "Combine them into a single coherent document overview that captures the "
    "full scope, purpose, key findings, structure, and important details. "
    "Do NOT add information not present in the summaries. "
    "Write in third person. Use clear paragraphs.\n\n"
    "--- SECTION SUMMARIES ---\n{text}\n--- END ---\n\n"
    "Document overview:"
)

_SINGLE_PROMPT = (
    "Write a concise overview of the following document. "
    "Preserve key facts, named entities, technical details, relationships, "
    "and any numerical data. Do NOT add opinions, commentary, or information "
    "not present in the text. Write in third person. Use clear paragraphs.\n\n"
    "--- DOCUMENT ---\n{text}\n--- END ---\n\n"
    "Document overview:"
)

_METADATA_PROMPT = (
    "Based on this document overview, provide:\n"
    "1. The 5-10 most important key topics or themes\n"
    "2. The single best document type classification (e.g., technical specification, "
    "design document, user guide, datasheet, report, proposal, policy, contract, "
    "invoice, whitepaper, research paper, presentation, manual)\n\n"
    "Overview:\n{text}\n\n"
    "Respond ONLY with a JSON object in this exact format:\n"
    '{{"key_topics": ["topic1", "topic2", ...], "document_type": "type"}}'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_array(text: str) -> list[str]:
    """Best-effort extraction of a JSON array from model output."""
    import re

    # Try to find a JSON array in the text (greedy to match the outermost brackets)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(item).strip() for item in arr if str(item).strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: split on commas or newlines, strip bracket/quote chars
    items = re.split(r"[,\n]+", text)
    return [
        item.strip().strip("-").strip("•").strip('"').strip("'").strip("[").strip("]").strip()
        for item in items
        if item.strip().strip("-").strip("•").strip('"').strip("'").strip("[").strip("]").strip()
    ][:10]


def _parse_metadata(text: str) -> tuple[list[str], str]:
    """Extract key_topics and document_type from a combined JSON response."""
    import re

    # Strip <think>…</think> blocks that Qwen3 may emit
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                topics = [
                    str(t).strip()
                    for t in obj.get("key_topics", [])
                    if str(t).strip()
                ][:10]
                doc_type = str(obj.get("document_type", "")).strip().strip('"').strip("'")
                if topics and doc_type:
                    return topics, doc_type
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract topics array and doc_type separately
    topics = _parse_json_array(text)
    doc_type = "unknown"
    for line in text.split("\n"):
        if "document_type" in line.lower():
            # Try to extract value after colon or quotes
            parts = line.split(":", 1)
            if len(parts) > 1:
                doc_type = parts[1].strip().strip('"').strip("'").strip(",")
                break
    return topics, doc_type


def _build_markdown(
    source: str,
    summary: str,
    key_topics: list[str],
    document_type: str,
) -> str:
    """Render the overview as a Markdown document."""
    lines = [
        f"# Document Overview: {source}",
        "",
        f"**Document type:** {document_type}",
        "",
        "## Summary",
        "",
        summary,
        "",
    ]
    if key_topics:
        lines.append("## Key Topics")
        lines.append("")
        for topic in key_topics:
            lines.append(f"- {topic}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "OverviewConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[dict]:
    """Generate a document overview from structured extraction output.

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: OverviewConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Overview data dictionary, or None if no text is available.
    """
    model_id = config.model_id if config else "Qwen/Qwen3-VL-2B-Instruct"
    max_tokens = config.max_tokens if config else 1024
    chunk_size = config.chunk_size if config else 6000

    info_print(f"model: {model_id}")
    debug_print(f"max_tokens={max_tokens}, chunk_size={chunk_size}", debug=debug)

    # Read document text
    full_text = extract_document_text(output_folder)
    if not full_text.strip():
        info_print("No text found for overview (run with -s first)")
        return None

    word_count = len(full_text.split())
    info_print(f"Document text: {word_count} words, {len(full_text)} chars")

    # Load VLM
    from .utils.vlm import generate_text, load_vlm

    with gray_debug_output(debug):
        model, processor = load_vlm(model_id)

    # --- Generate summary ---
    chunks = chunk_text(full_text, max_chars=chunk_size)
    info_print(f"Processing {len(chunks)} text chunk(s)")

    if len(chunks) == 1:
        # Small document — single pass
        prompt = _SINGLE_PROMPT.format(text=chunks[0])
        with gray_debug_output(debug):
            summary = generate_text(
                model, processor, prompt, max_tokens=max_tokens,
            )
        debug_print(f"Single-pass summary: {len(summary)} chars", debug=debug)
    else:
        # Large document — summarize each chunk, then synthesize
        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            prompt = _CHUNK_PROMPT.format(text=chunk)
            with gray_debug_output(debug):
                cs = generate_text(
                    model, processor, prompt, max_tokens=max_tokens // 2,
                )
            chunk_summaries.append(cs)
            debug_print(
                f"Chunk {i}/{len(chunks)} summary: {len(cs)} chars", debug=debug,
            )

        combined = "\n\n".join(
            f"[Section {i}]\n{s}" for i, s in enumerate(chunk_summaries, 1)
        )
        prompt = _SYNTHESIZE_PROMPT.format(text=combined)
        with gray_debug_output(debug):
            summary = generate_text(
                model, processor, prompt, max_tokens=max_tokens,
            )
        debug_print(f"Synthesized overview: {len(summary)} chars", debug=debug)

    # --- Extract key topics + document type in one call ---
    prompt = _METADATA_PROMPT.format(text=summary)
    with gray_debug_output(debug):
        metadata_raw = generate_text(
            model, processor, prompt, max_tokens=192, temperature=0.3,
        )
    key_topics, document_type = _parse_metadata(metadata_raw)
    debug_print(f"Key topics: {key_topics}", debug=debug)
    debug_print(f"Document type: {document_type}", debug=debug)

    # --- Build output ---
    source_name = Path(input_file).name
    output_data = {
        "source": source_name,
        "model": model_id,
        "summary": summary,
        "key_topics": key_topics,
        "document_type": document_type,
        "word_count": word_count,
    }

    overview_dir = os.path.join(output_folder, "overview")
    os.makedirs(overview_dir, exist_ok=True)

    json_path = os.path.join(overview_dir, "overview.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    md_content = _build_markdown(source_name, summary, key_topics, document_type)
    md_path = os.path.join(overview_dir, "overview.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    info_print(f"Overview: {len(summary)} chars, {len(key_topics)} topics → overview/")
    info_print(f"Document type: {document_type}")

    return output_data
