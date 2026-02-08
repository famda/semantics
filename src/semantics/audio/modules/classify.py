"""Audio classification module using AST (Audio Spectrogram Transformer).

This module provides functionality for classifying audio content using
the pre-trained AST model from MIT. It supports chunked processing for
long audio files.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch

from global_helpers import AUDIO_CLASSIFICATION_CATEGORY_MAP
from .utils.chunks import (
    CHUNK_LENGTH_SECONDS as DEFAULT_CHUNK_LENGTH_SECONDS,
    cleanup_chunks,
    split_audio,
)
from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from ..config import ClassifyConfig

__all__ = ["handle"]

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import librosa
except ImportError as e:
    print(f"Error importing necessary libraries: {e}")
    exit(1)


def _get_classify_defaults() -> dict:
    """Get default values from ClassifyConfig to avoid circular imports."""
    try:
        from config import ClassifyConfig
        cfg = ClassifyConfig()
        return {
            "model_id": cfg.model_id,
            "expected_sample_rate": cfg.expected_sample_rate,
            "chunk_length": cfg.chunk_length,
            "top_n": cfg.top_n,
        }
    except Exception:
        # Fallback defaults if config import fails
        return {
            "model_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "expected_sample_rate": 16000,
            "chunk_length": 900,
            "top_n": 5,
        }


_MODEL_CACHE: Dict[
    str, Tuple[AutoFeatureExtractor, AutoModelForAudioClassification, torch.device]
] = {}


def _info(message: str) -> None:
    print(message)


def _debug(message: str, *, debug: bool) -> None:
    debug_print(message, debug=debug)


def _classification_components(
    *,
    model_id: str,
    debug: bool = False,
) -> Tuple[AutoFeatureExtractor, AutoModelForAudioClassification, torch.device]:
    """Lazily load and cache the AST model + feature extractor on first use."""
    global _MODEL_CACHE

    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with gray_debug_output(debug):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    try:
        with gray_debug_output(debug):
            model.to(device)
    except Exception:
        model.to(torch.device("cpu"))
        device = torch.device("cpu")

    _MODEL_CACHE[model_id] = (feature_extractor, model, device)
    return _MODEL_CACHE[model_id]


def handle(
    input_file: str,
    output_folder: str,
    config: "ClassifyConfig | None" = None,
    *,
    debug: bool = False,
) -> dict | None:
    """Perform audio classification using AST.

    This is the standardized entry point for the classification module.

    Args:
        input_file: Path to the input audio file.
        output_folder: Directory where output files will be written.
        config: ClassifyConfig instance with classification parameters, or None for defaults.
        debug: If True, emit verbose debug output.

    Returns:
        Dictionary with classification results including category and top classes.
    """
    defaults = _get_classify_defaults()
    chunk_length = config.chunk_length if config else defaults["chunk_length"]
    model_id = config.model_id if config else defaults["model_id"]
    expected_sample_rate = (
        config.expected_sample_rate if config else defaults["expected_sample_rate"]
    )
    top_n = config.top_n if config else defaults["top_n"]

    _info("INFO: Performing classification")

    base_temp_folder = output_folder
    output_json_folder = os.path.join(base_temp_folder, "classification")
    os.makedirs(output_json_folder, exist_ok=True)

    class_to_category: Dict[str, str] = {}
    for category, classes in AUDIO_CLASSIFICATION_CATEGORY_MAP.items():
        for cls_name in classes:
            class_to_category[cls_name] = category

    feature_extractor, model, device = _classification_components(
        model_id=model_id, debug=debug
    )
    _debug(f"DEBUG: Using device {device} for AST classification", debug=debug)

    if hasattr(model.config, "id2label"):  # type: ignore[union-attr]
        class_names = [model.config.id2label[i] for i in range(model.config.num_labels)]  # type: ignore[union-attr]
        _debug("DEBUG: Loaded AST class names.", debug=debug)
    else:
        print("Error: Could not retrieve class names (id2label) from model config.")
        return None

    chunks, chunk_dir = split_audio(
        input_file, base_temp_folder, "classify", chunk_length
    )

    aggregated_probabilities = None
    total_weight = 0.0

    try:
        for idx, chunk_path in enumerate(chunks, start=1):
            with gray_debug_output(debug):
                waveform, _ = librosa.load(
                    chunk_path, sr=expected_sample_rate, mono=True
                )
            if waveform is None or waveform.size == 0:
                _debug(f"DEBUG: Skipping empty chunk {chunk_path}", debug=debug)
                continue

            inputs = feature_extractor(
                waveform, sampling_rate=expected_sample_rate, return_tensors="pt"
            )  # type: ignore[operator]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                with gray_debug_output(debug):
                    outputs = model(**inputs)  # type: ignore[operator]

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            duration_weight = len(waveform) / float(expected_sample_rate)
            if not np.isfinite(duration_weight) or duration_weight <= 0:
                duration_weight = 1.0

            if aggregated_probabilities is None:
                aggregated_probabilities = probabilities * duration_weight
            else:
                aggregated_probabilities += probabilities * duration_weight

            total_weight += duration_weight
            if idx % 10 == 0:
                _debug(f"DEBUG: Processed {idx}/{len(chunks)} chunk(s)", debug=debug)
    finally:
        cleanup_chunks(chunk_dir)

    if aggregated_probabilities is None or total_weight == 0.0:
        probabilities = np.zeros(len(class_names), dtype=np.float32)
    else:
        probabilities = aggregated_probabilities / total_weight

    top_n = int(top_n) if top_n and top_n > 0 else 5
    num_classes = len(class_names)
    top_n = min(top_n, num_classes)
    top_class_indices = np.argsort(probabilities)[::-1][:top_n]

    output_data = {}
    top_ast_index = top_class_indices[0]
    top_ast_class = class_names[top_ast_index]
    predicted_category = class_to_category.get(top_ast_class, "Mixed/Other")
    output_data["category"] = predicted_category

    classes_list = []
    for i in top_class_indices:
        class_entry = {"class": class_names[i], "confidence": float(probabilities[i])}
        classes_list.append(class_entry)

    output_data["classes"] = classes_list

    output_json_path = os.path.join(output_json_folder, "classification.json")

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)

    return output_data
