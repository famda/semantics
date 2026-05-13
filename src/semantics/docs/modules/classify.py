"""LLM-based document classification and tagging.

Classifies a document by feeding its content to Qwen3-VL in text-only
mode and asking it to determine the document type and relevant tags.
Requires structured extraction to have been run first (reads structure.json).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .utils.chunking import extract_document_text
from .utils.logging import debug_print, gray_debug_output, info_print

if TYPE_CHECKING:
    from ..config import ClassifyConfig

__all__ = ["handle"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = (
    "You are a document classification expert. Analyze the following document "
    "content and determine:\n"
    "1. The primary document type (single label)\n"
    "2. A confidence score from 0.0 to 1.0\n"
    "3. Up to {max_tags} relevant tags that describe the document's themes\n"
    "4. A brief reasoning for your classification\n\n"
    "{label_guidance}"
    "Document content:\n"
    "---\n{text}\n---\n\n"
    "Respond ONLY with a JSON object in this exact format:\n"
    '{{"classification": "document type", "confidence": 0.95, '
    '"tags": ["tag1", "tag2"], "reasoning": "brief explanation"}}'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_classification(raw: str) -> dict | None:
    """Best-effort extraction of a JSON object from model output."""
    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "classification" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole string
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "classification" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "ClassifyConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[dict]:
    """Classify a document using LLM-based analysis.

    Requires structured extraction to have been run first (-s flag).

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: ClassifyConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Dictionary with classification results or None if no text.
    """
    model_id = config.model if config else "Qwen/Qwen3-VL-2B-Instruct"
    max_tokens = config.max_tokens if config else 256
    candidate_labels = (
        config.candidate_labels
        if config
        else [
            "invoice", "contract", "report", "letter", "resume",
            "scientific paper", "manual", "form", "presentation",
            "legal document", "financial statement", "memo",
            "newsletter", "brochure", "engineering document",
            "architecture document", "cyber security report",
            "technical specification", "user guide", "policy document",
            "compliance document", "project plan", "meeting notes",
            "proposal", "whitepaper", "datasheet",
        ]
    )
    max_tags = config.max_tags if config else 8

    info_print(f"model: {model_id}")

    classify_dir = os.path.join(output_folder, "classification")
    os.makedirs(classify_dir, exist_ok=True)

    # Read first ~4000 chars — title + intro is usually enough for classification
    text = extract_document_text(output_folder, max_chars=4000)
    if not text.strip():
        info_print("No text found for classification (run with -s first)")
        return None

    debug_print(f"Input text: {len(text)} chars", debug=debug)

    # Build label guidance
    if candidate_labels:
        label_guidance = (
            "Consider these categories (you may also suggest a better one): "
            + ", ".join(candidate_labels) + "\n\n"
        )
    else:
        label_guidance = ""

    prompt = _CLASSIFY_PROMPT.format(
        text=text,
        max_tags=max_tags,
        label_guidance=label_guidance,
    )

    # Load VLM and generate
    from .utils.vlm import generate_text, load_vlm

    with gray_debug_output(debug):
        model, processor = load_vlm(model_id)

    with gray_debug_output(debug):
        raw_output = generate_text(
            model, processor, prompt,
            max_tokens=max_tokens,
            temperature=0.3,
        )

    debug_print(f"Raw model output: {raw_output[:200]}", debug=debug)

    # Parse classification
    parsed = _parse_classification(raw_output)

    if parsed:
        classification = str(parsed.get("classification", "unknown")).strip()
        confidence = float(parsed.get("confidence", 0.0))
        tags = [str(t).strip() for t in parsed.get("tags", [])][:max_tags]
        reasoning = str(parsed.get("reasoning", "")).strip()
    else:
        # Fallback: use raw output as classification
        info_print("Warning: could not parse JSON from model output, using raw text")
        classification = raw_output.split("\n")[0].strip()[:100]
        confidence = 0.5
        tags = []
        reasoning = raw_output

    output_data = {
        "source": Path(input_file).name,
        "model": model_id,
        "classification": classification,
        "confidence": round(confidence, 4),
        "tags": tags,
        "reasoning": reasoning,
        "settings": {
            "candidate_labels": candidate_labels,
            "max_tags": max_tags,
        },
    }

    output_path = os.path.join(classify_dir, "classification.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    tag_str = f" | Tags: {', '.join(tags)}" if tags else ""
    info_print(f"Classification: {classification} ({confidence:.0%}){tag_str}")
    debug_print(f"Reasoning: {reasoning}", debug=debug)

    return output_data
