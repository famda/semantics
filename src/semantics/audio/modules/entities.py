"""Named Entity Recognition module for audio transcriptions.

Extracts named entities (persons, organizations, locations, etc.) from
transcription segments using transformer-based NER models.
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

# Common false positives from transcription artifacts (lowercase for matching)
_FALSE_POSITIVE_ENTITIES = frozenset({
    "um", "uh", "ah", "eh", "oh", "mm", "hmm",  # Filler words
    ".", "..", "...", ",", "!", "?",  # Punctuation
    "the", "a", "an", "and", "or", "but", "to", "of", "for",  # Common words
    "yous", "youse", "youre", "youll",  # Transcription artifacts
    "ai", "pro", "the pro", "ok", "okay",  # Generic terms
})

# Minimum length for valid entities (after stripping whitespace)
_MIN_ENTITY_LENGTH = 2


def handle(
    input_file: str,
    output_folder: str,
    config: "NerConfig | None" = None,
    *,
    segments_file: Optional[str] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract named entities from transcription segments.

    Args:
        input_file: Path to input audio file (for reference).
        output_folder: Path to output directory.
        config: NerConfig instance or None for defaults.
        segments_file: Path to transcription/alignment JSON with segments.
        debug: Enable verbose debug output.

    Returns:
        Dictionary with NER results or None if no segments available.
    """
    info_print("Performing Named Entity Recognition")

    # Extract config values with inline defaults
    model_name = config.model_name if config else "Jean-Baptiste/roberta-large-ner-english"
    device = config.device if config else None
    batch_size = config.batch_size if config else 8
    confidence_threshold = config.confidence_threshold if config else 0.92
    aggregate_strategy = config.aggregate_strategy if config else "simple"

    # Create output folder
    entities_folder = os.path.join(output_folder, "entities")
    os.makedirs(entities_folder, exist_ok=True)

    # Load segments from transcription or alignment
    segments = _load_segments(output_folder, segments_file, debug)
    if not segments:
        print("WARN: No segments found for NER analysis")
        return None

    # Initialize NER pipeline
    ner_pipeline = _get_ner_pipeline(
        model_name=model_name,
        device=device,
        aggregation_strategy=aggregate_strategy,
        debug=debug,
    )

    # Process segments
    results = _process_segments(
        segments=segments,
        ner_pipeline=ner_pipeline,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        debug=debug,
    )

    # Build output
    output_data = {
        "audio": input_file,
        "model": model_name,
        "settings": {
            "confidence_threshold": confidence_threshold,
            "aggregate_strategy": aggregate_strategy,
        },
        "summary": _summarize_entities(results),
        "segments": results,
    }

    # Write output
    output_path = os.path.join(entities_folder, "audio-entities.json")
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
        )

    _NER_PIPELINE_CACHE[cache_key] = ner
    return ner


def _load_segments(
    output_folder: str,
    segments_file: Optional[str],
    debug: bool,
) -> List[dict]:
    """Load segments from CTC alignment or transcription."""
    # Try explicit path first
    if segments_file and os.path.exists(segments_file):
        debug_print(f"Loading segments from {segments_file}", debug=debug)
        with open(segments_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("segments", [])

    # Try CTC alignment
    ctc_path = os.path.join(output_folder, "ctc", "alignment.json")
    if os.path.exists(ctc_path):
        debug_print("Loading segments from CTC alignment", debug=debug)
        with open(ctc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("segments", [])

    # Fall back to transcription
    transcript_path = os.path.join(output_folder, "transcription", "transcription.json")
    if os.path.exists(transcript_path):
        debug_print("Loading segments from transcription", debug=debug)
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("segments", [])

    return []


def _process_segments(
    segments: List[dict],
    ner_pipeline: Any,
    batch_size: int,
    confidence_threshold: float,
    debug: bool,
) -> List[dict]:
    """Run NER on each segment."""
    results = []

    # Collect all texts for batch processing
    segment_texts = []
    segment_indices = []
    for idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if text:
            segment_texts.append(text)
            segment_indices.append(idx)

    if not segment_texts:
        return results

    debug_print(f"Processing {len(segment_texts)} segments for NER", debug=debug)

    # Process in batches
    all_entities: List[List[dict]] = []
    with gray_debug_output(debug):
        for i in range(0, len(segment_texts), batch_size):
            batch = segment_texts[i : i + batch_size]
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

    # Map results back to segments
    for entity_list, seg_idx in zip(all_entities, segment_indices):
        seg = segments[seg_idx]
        text = seg.get("text", "").strip()

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
                "segment_id": seg.get("id", seg_idx),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": text,
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
        "total_segments": len(results),
        "total_entities": len(all_entities),
        "entity_counts": dict(label_counts),
        "unique_entities": len(unique_texts),
    }
