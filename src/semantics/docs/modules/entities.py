"""Named Entity Recognition for documents.

Extracts named entities (persons, organizations, locations, etc.) from
structured document text using transformer-based NER models. Requires
structured extraction to have been run first (reads structure.json).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import pipeline

from .utils.logging import debug_print, gray_debug_output, info_print

if TYPE_CHECKING:
    from ..config import NerConfig

__all__ = ["handle"]

_NER_PIPELINE_CACHE: dict[str, Any] = {}

_FALSE_POSITIVE_ENTITIES = frozenset({
    ".", "..", "...", ",", "!", "?",
    "the", "a", "an", "and", "or", "but", "to", "of", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "it", "its", "they", "them", "their",
})

_MIN_ENTITY_LENGTH = 2


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


def _load_structured_texts(output_folder: str, *, debug: bool = False) -> list[dict]:
    """Load text elements from structure.json."""
    structure_path = os.path.join(output_folder, "structured", "structure.json")
    if not os.path.exists(structure_path):
        debug_print(f"No structure.json found at {structure_path}", debug=debug)
        return []

    with open(structure_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements = data.get("elements", [])
    # Filter to text-bearing elements with meaningful content
    texts = []
    for el in elements:
        text = (el.get("text") or "").strip()
        if len(text) < 10:
            continue
        texts.append({
            "text": text,
            "type": el.get("type", "Text"),
            "page": (el.get("metadata") or {}).get("page_number"),
        })

    debug_print(f"Loaded {len(texts)} text elements for NER", debug=debug)
    return texts


def _process_texts(
    texts: list[dict],
    ner_pipeline: Any,
    batch_size: int,
    confidence_threshold: float,
    debug: bool,
) -> list[dict]:
    """Run NER on each text element."""
    results = []
    text_strings = [t["text"] for t in texts]

    if not text_strings:
        return results

    debug_print(f"Processing {len(text_strings)} text elements for NER", debug=debug)

    all_entities: list[list[dict]] = []
    with gray_debug_output(debug):
        for i in range(0, len(text_strings), batch_size):
            batch = text_strings[i : i + batch_size]
            batch_results = ner_pipeline(batch)
            if batch and isinstance(batch_results, list):
                if len(batch) == 1:
                    if batch_results and isinstance(batch_results[0], list):
                        all_entities.append(batch_results[0])
                    else:
                        all_entities.append(batch_results)
                else:
                    for result in batch_results:
                        if isinstance(result, list):
                            all_entities.append(result)
                        else:
                            all_entities.append([result] if result else [])

    for entity_list, text_info in zip(all_entities, texts):
        filtered = []
        for e in entity_list:
            entity_text = e.get("word", "").strip()
            entity_lower = entity_text.lower()
            score = e.get("score", 0)

            if score < confidence_threshold:
                continue
            if len(entity_text) < _MIN_ENTITY_LENGTH:
                continue
            if entity_lower in _FALSE_POSITIVE_ENTITIES:
                continue

            filtered.append({
                "text": entity_text,
                "label": e.get("entity_group", e.get("entity", "")),
                "score": round(float(score), 4),
                "start": e.get("start"),
                "end": e.get("end"),
            })

        if filtered:
            results.append({
                "source_text": text_info["text"][:200],
                "source_type": text_info["type"],
                "page": text_info["page"],
                "entities": filtered,
            })

    return results


def _summarize_entities(results: list[dict]) -> dict:
    """Build summary statistics from NER results."""
    entity_counts: Counter = Counter()
    label_counts: Counter = Counter()
    all_entities: list[dict] = []

    for r in results:
        for e in r.get("entities", []):
            entity_counts[e["text"]] += 1
            label_counts[e["label"]] += 1
            all_entities.append(e)

    return {
        "total_entities": len(all_entities),
        "unique_entities": len(entity_counts),
        "by_label": dict(label_counts),
        "top_entities": [
            {"text": text, "count": count}
            for text, count in entity_counts.most_common(20)
        ],
    }


def handle(
    input_file: str,
    output_folder: str,
    config: "NerConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[dict]:
    """Extract named entities from document text.

    Requires structured extraction to have been run first (-s flag).

    Args:
        input_file: Path to the input document file.
        output_folder: Path to the output directory.
        config: NerConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Dictionary with NER results or None if no text available.
    """
    info_print("Performing Named Entity Recognition")

    model_name = config.model if config else "Jean-Baptiste/roberta-large-ner-english"
    device = config.device if config else None
    batch_size = config.batch_size if config else 8
    confidence_threshold = config.confidence_threshold if config else 0.6
    aggregation_strategy = config.aggregation_strategy if config else "simple"

    entities_folder = os.path.join(output_folder, "entities")
    os.makedirs(entities_folder, exist_ok=True)

    texts = _load_structured_texts(output_folder, debug=debug)
    if not texts:
        info_print("No text elements found for NER (run with -s first)")
        return None

    ner_pipeline_inst = _get_ner_pipeline(
        model_name=model_name,
        device=device,
        aggregation_strategy=aggregation_strategy,
        debug=debug,
    )

    results = _process_texts(
        texts=texts,
        ner_pipeline=ner_pipeline_inst,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        debug=debug,
    )

    summary = _summarize_entities(results)

    output_data = {
        "source": Path(input_file).name,
        "model": model_name,
        "settings": {
            "confidence_threshold": confidence_threshold,
            "aggregation_strategy": aggregation_strategy,
        },
        "summary": summary,
        "elements": results,
    }

    output_path = os.path.join(entities_folder, "entities.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    entity_count = summary["total_entities"]
    info_print(f"Found {entity_count} entities → entities/")
    debug_print(f"NER results saved to {output_path}", debug=debug)

    return output_data
