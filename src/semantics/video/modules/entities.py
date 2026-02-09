"""Named Entity Recognition module for video captions.

Extracts named entities (persons, organizations, locations, etc.) from
video caption text using transformer-based NER models.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import pipeline

from .utils.logging import debug_print, gray_debug_output, info_print

if TYPE_CHECKING:
    from ..config import NerConfig

__all__ = ["handle"]

# Module-level cache for NER pipeline reuse
_NER_PIPELINE_CACHE: dict[str, Any] = {}

# Common false positives from vision/caption artifacts (lowercase for matching)
_FALSE_POSITIVE_ENTITIES = frozenset({
    "um", "uh", "ah", "eh", "oh", "mm", "hmm",  # Filler words
    ".", "..", "...", ",", "!", "?",  # Punctuation
    "the", "a", "an", "and", "or", "but", "to", "of", "for",  # Common words
    "yous", "youse", "youre", "youll",  # Transcription artifacts
    "ai", "pro", "the pro", "ok", "okay",  # Generic terms
    "man", "woman", "person", "people",  # Generic vision terms
})

# Minimum length for valid entities (after stripping whitespace)
_MIN_ENTITY_LENGTH = 2


def handle(
    input_file: str,
    output_folder: str,
    config: "NerConfig | None" = None,
    *,
    captions_file: Optional[str] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract named entities from video captions.

    Args:
        input_file: Path to input video file (for reference).
        output_folder: Path to output directory.
        config: NerConfig instance or None for defaults.
        captions_file: Path to captions JSON file.
        debug: Enable verbose debug output.

    Returns:
        Dictionary with NER results or None if no captions available.
    """
    info_print("Performing Named Entity Recognition on captions")

    # Extract config values with inline defaults
    model_name = config.model_name if config else "Jean-Baptiste/roberta-large-ner-english"
    device = config.device if config else None
    batch_size = config.batch_size if config else 8
    confidence_threshold = config.confidence_threshold if config else 0.92
    aggregate_strategy = config.aggregate_strategy if config else "simple"
    caption_field = config.caption_field if config else "caption_more_detailed"

    # Create output folder
    entities_folder = os.path.join(output_folder, "entities")
    os.makedirs(entities_folder, exist_ok=True)

    # Load captions
    captions = _load_captions(output_folder, captions_file, debug)
    if not captions:
        print("WARN: No captions found for NER analysis")
        return None

    # Initialize NER pipeline
    ner_pipeline = _get_ner_pipeline(
        model_name=model_name,
        device=device,
        aggregation_strategy=aggregate_strategy,
        debug=debug,
    )

    # Process captions
    results = _process_captions(
        captions=captions,
        ner_pipeline=ner_pipeline,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        caption_field=caption_field,
        debug=debug,
    )

    # Build output
    output_data = {
        "video": input_file,
        "model": model_name,
        "settings": {
            "confidence_threshold": confidence_threshold,
            "aggregate_strategy": aggregate_strategy,
            "caption_field": caption_field,
        },
        "summary": _summarize_entities(results),
        "frames": results,
    }

    # Write output
    output_path = os.path.join(entities_folder, "video-entities.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    entity_count = output_data["summary"]["total_entities"]
    debug_print(f"NER complete - {entity_count} entities found", debug=debug)
    debug_print(f"NER results saved to {output_path}", debug=debug)
    return output_data


def _get_ner_pipeline(
    model_name: str,
    device: Optional[str],
    aggregation_strategy: str,
    debug: bool,
) -> Any:
    """Load and cache NER pipeline."""
    cache_key = f"{model_name}_{device}_{aggregation_strategy}"
    if cache_key in _NER_PIPELINE_CACHE:
        debug_print("Using cached NER pipeline", debug=debug)
        return _NER_PIPELINE_CACHE[cache_key]

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"Loading NER model '{model_name}' on {resolved_device}", debug=debug)

    with gray_debug_output(debug):
        ner = pipeline(
            "ner",
            model=model_name,
            device=0 if resolved_device == "cuda" else -1,
            aggregation_strategy=aggregation_strategy,
            framework="pt",
        )

    _NER_PIPELINE_CACHE[cache_key] = ner
    return ner


def _load_captions(
    output_folder: str,
    captions_file: Optional[str],
    debug: bool,
) -> List[dict]:
    """Load captions from JSON file."""
    # Try explicit path first
    if captions_file and os.path.exists(captions_file):
        debug_print(f"Loading captions from {captions_file}", debug=debug)
        with open(captions_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Try default captions folder
    captions_path = os.path.join(output_folder, "captions", "captions.json")
    if os.path.exists(captions_path):
        debug_print("Loading captions from captions folder", debug=debug)
        with open(captions_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return []


def _process_captions(
    captions: List[dict],
    ner_pipeline: Any,
    batch_size: int,
    confidence_threshold: float,
    caption_field: str,
    debug: bool,
) -> List[dict]:
    """Run NER on each caption."""
    results = []

    # Collect all texts for batch processing
    caption_texts = []
    caption_indices = []

    for idx, cap in enumerate(captions):
        # Try the configured field, fall back to alternatives
        text = cap.get(caption_field, "").strip()
        if not text:
            text = cap.get("caption_detailed", "").strip()
        if not text:
            text = cap.get("caption", "").strip()

        if text:
            caption_texts.append(text)
            caption_indices.append(idx)

    if not caption_texts:
        return results

    debug_print(f"Processing {len(caption_texts)} captions for NER", debug=debug)

    # Process in batches
    all_entities: List[List[dict]] = []
    with gray_debug_output(debug):
        for i in range(0, len(caption_texts), batch_size):
            batch = caption_texts[i : i + batch_size]
            batch_results = ner_pipeline(batch)
            # Handle single vs batch results
            # When batch size is 1, results may be a flat list of entities
            # When batch size > 1, results is a list of entity lists
            if batch and isinstance(batch_results, list):
                if len(batch) == 1:
                    # Single item: batch_results is a list of entities
                    # Check if it's nested (list of lists) or flat (list of dicts)
                    if batch_results and isinstance(batch_results[0], list):
                        all_entities.append(batch_results[0])
                    else:
                        all_entities.append(batch_results)
                else:
                    # Multiple items: batch_results is a list of entity lists
                    for result in batch_results:
                        if isinstance(result, list):
                            all_entities.append(result)
                        else:
                            # Single dict result - wrap it
                            all_entities.append([result] if result else [])

    # Map results back to captions
    for entity_list, cap_idx in zip(all_entities, caption_indices):
        cap = captions[cap_idx]

        # Get text used for NER
        text = cap.get(caption_field, "").strip()
        if not text:
            text = cap.get("caption_detailed", "").strip()
        if not text:
            text = cap.get("caption", "").strip()

        filtered = []
        for e in entity_list:
            entity_text = e.get("word", "").strip()
            entity_lower = entity_text.lower()
            score = e.get("score", 0)
            
            # Apply filters: confidence threshold, min length, false positive list
            if score < confidence_threshold:
                continue
            if len(entity_text) < _MIN_ENTITY_LENGTH:
                continue
            if entity_lower in _FALSE_POSITIVE_ENTITIES:
                continue
            
            filtered.append({
                "text": e.get("word", ""),
                "label": e.get("entity_group", e.get("entity", "")),
                "score": round(float(score), 4),
                "start": e.get("start"),
                "end": e.get("end"),
            })

        results.append(
            {
                "frame": cap.get("frame"),
                "frame_path": cap.get("frame_path"),
                "caption_text": text,
                "entities": filtered,
            }
        )

    return results


def _summarize_entities(results: List[dict]) -> dict:
    """Generate summary statistics."""
    all_entities = []
    for r in results:
        all_entities.extend(r.get("entities", []))

    label_counts = Counter(e["label"] for e in all_entities)
    unique_texts = set(e["text"].lower().strip() for e in all_entities if e["text"])

    return {
        "total_frames": len(results),
        "total_entities": len(all_entities),
        "entity_counts": dict(label_counts),
        "unique_entities": len(unique_texts),
    }
