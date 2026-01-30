"""Image classification module for video frames using YOLO classification models.

Classifies video frames using Ultralytics YOLO classification models and
produces structured JSON output with per-frame predictions.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import ClassificationConfig

__all__ = ["handle"]

os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# =============================================================================
# Internal Settings Dataclass
# =============================================================================


@dataclass
class _ClassificationSettings:
    """Internal settings container for classification module."""

    model: str
    conf_threshold: float
    top_k: int


# =============================================================================
# Cached Model Singleton
# =============================================================================

_YOLO_CLASSIFICATION_MODEL: Optional[YOLO] = None
_YOLO_CLASSIFICATION_MODEL_NAME: Optional[str] = None
_YOLO_DEVICE: Optional[torch.device] = None
_YOLO_USE_HALF_PRECISION: bool = False


class _NumpyEncoder(json.JSONEncoder):
    """Internal JSON encoder for numpy types."""

    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _ensure_yolo_classification_model(
    model_name: str,
    *,
    debug: bool = False,
) -> Tuple[Optional[YOLO], Optional[torch.device], bool]:
    """Load and cache the YOLO classification model.

    Args:
        model_name: YOLO classification model name (e.g., 'yolo11s-cls.pt').
        debug: Enable debug output.

    Returns:
        Tuple of (model, device, use_half_precision).
    """
    global _YOLO_CLASSIFICATION_MODEL, _YOLO_CLASSIFICATION_MODEL_NAME, _YOLO_DEVICE, _YOLO_USE_HALF_PRECISION

    # Return cached model if same model is requested
    if (
        _YOLO_CLASSIFICATION_MODEL is not None
        and _YOLO_CLASSIFICATION_MODEL_NAME == model_name
        and _YOLO_DEVICE is not None
    ):
        return _YOLO_CLASSIFICATION_MODEL, _YOLO_DEVICE, _YOLO_USE_HALF_PRECISION

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_half = True
    else:
        device = torch.device("cpu")
        use_half = False

    debug_print(f"Loading YOLO classification model: {model_name} on {device}", debug=debug)

    try:
        with gray_debug_output(debug):
            model = YOLO(model_name)
            model.to(device)
    except Exception as exc:
        print(f"ERROR: Failed to load YOLO classification model '{model_name}': {exc}")
        return None, None, False

    _YOLO_CLASSIFICATION_MODEL = model
    _YOLO_CLASSIFICATION_MODEL_NAME = model_name
    _YOLO_DEVICE = device
    _YOLO_USE_HALF_PRECISION = use_half

    return model, device, use_half


def _iterate_selected_frames(
    video_path: str,
    indices: Sequence[int],
    *,
    debug: bool = False,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Iterate through specific frame indices in a video.

    Args:
        video_path: Path to the video file.
        indices: Sequence of frame indices to extract.
        debug: Enable debug output.

    Yields:
        Tuple of (frame_index, frame_bgr_image).
    """
    # Normalize indices
    normalized: List[int] = []
    for value in indices:
        if value is None:
            continue
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            continue

    ordered = sorted(set(normalized))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    max_index: Optional[int] = None
    try:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    except Exception:
        frame_count = 0.0

    if isinstance(frame_count, (int, float)) and frame_count > 0:
        try:
            max_index = int(frame_count) - 1
        except Exception:
            max_index = None

    if max_index is not None:
        in_range: List[int] = []
        clipped: List[int] = []
        for idx in ordered:
            if 0 <= idx <= max_index:
                in_range.append(idx)
            else:
                clipped.append(idx)
        if clipped:
            sample = ", ".join(str(item) for item in clipped[:5])
            debug_print(
                f"Skipping {len(clipped)} frame(s) outside valid range [0, {max_index}] (examples: {sample})",
                debug=debug,
            )
        ordered = in_range
    else:
        ordered = [idx for idx in ordered if idx >= 0]

    if not ordered:
        cap.release()
        return

    try:
        target_iter = iter(ordered)
        try:
            next_index = next(target_iter)
        except StopIteration:
            return

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                remaining = len(ordered) - processed
                if remaining > 0 and debug:
                    missing_preview = ", ".join(
                        str(item) for item in ordered[processed : processed + 3]
                    )
                    debug_print(
                        f"Decoder hit end-of-stream after {processed} frame(s); skipping {remaining} pending index(es) (next: {missing_preview}).",
                        debug=debug,
                    )
                break

            if frame_idx == next_index:
                yield next_index, frame
                processed += 1

                try:
                    next_index = next(target_iter)
                except StopIteration:
                    break

            frame_idx += 1
    finally:
        cap.release()


def _probe_video_fps(video_path: str) -> float:
    """Get the FPS of a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0
    finally:
        cap.release()


def _classify_frame(
    frame: np.ndarray,
    model: YOLO,
    *,
    use_half: bool = False,
    top_k: int = 5,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Classify a single frame using the YOLO model.

    Args:
        frame: BGR image as numpy array.
        model: Loaded YOLO classification model.
        use_half: Use half precision for inference.
        top_k: Number of top predictions to return.
        debug: Enable debug output.

    Returns:
        List of prediction dictionaries with class_id, label, and confidence.
    """
    try:
        with gray_debug_output(debug):
            results = model(frame, verbose=False, half=use_half)
    except Exception as exc:
        debug_print(f"WARNING: Classification inference failed: {exc}", debug=debug)
        return []

    predictions: List[Dict[str, Any]] = []

    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs

            # Get top-k indices and scores
            if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
                # Use built-in top5 if available
                top_indices = probs.top5[:top_k] if len(probs.top5) >= top_k else probs.top5
                top_scores = probs.top5conf[:top_k] if len(probs.top5conf) >= top_k else probs.top5conf

                for idx, score in zip(top_indices, top_scores):
                    class_name = result.names.get(int(idx), f"class_{idx}")
                    predictions.append({
                        "class_id": int(idx),
                        "label": class_name,
                        "confidence": float(score),
                    })
            elif hasattr(probs, "data"):
                # Fallback: manually get top-k from probability tensor
                data = probs.data
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()

                top_indices = np.argsort(data)[::-1][:top_k]
                for idx in top_indices:
                    score = float(data[idx])
                    class_name = result.names.get(int(idx), f"class_{idx}")
                    predictions.append({
                        "class_id": int(idx),
                        "label": class_name,
                        "confidence": score,
                    })

    return predictions


def _save_annotated_frame(
    frame: np.ndarray,
    predictions: List[Dict[str, Any]],
    output_path: str,
    *,
    debug: bool = False,
) -> Optional[str]:
    """Save a frame with classification annotations overlaid.

    Args:
        frame: BGR image as numpy array.
        predictions: List of prediction dictionaries.
        output_path: Path to save the annotated image.
        debug: Enable debug output.

    Returns:
        Path to saved image, or None on failure.
    """
    try:
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        # Draw semi-transparent background for text
        overlay = annotated.copy()
        box_height = min(40 + len(predictions) * 30, height // 2)
        cv2.rectangle(overlay, (10, 10), (width - 10, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Draw predictions
        y_offset = 35
        for pred in predictions:
            label = pred["label"]
            conf = pred["confidence"]
            text = f"{label}: {conf:.2%}"
            cv2.putText(
                annotated,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30

        cv2.imwrite(output_path, annotated)
        return output_path

    except Exception as exc:
        debug_print(f"WARNING: Failed to save annotated frame: {exc}", debug=debug)
        return None


def _extract_config_settings(config: Optional["ClassificationConfig"]) -> _ClassificationSettings:
    """Extract all config values upfront using the project's `ClassificationConfig` defaults.

    If `config` is None, instantiate `ClassificationConfig` from the project's
    `config.py` so module defaults come from the central configuration.
    """
    if config is None:
        try:
            from config import ClassificationConfig

            cfg = ClassificationConfig()
        except Exception:
            # As a last-resort fallback (shouldn't happen in normal execution),
            # provide conservative defaults to avoid crashing.
            cfg = None
    else:
        cfg = config

    if cfg is None:
        # conservative hard-coded fallback (edge-case only)
        return _ClassificationSettings(model="yolo26s-cls.pt", conf_threshold=0.25, top_k=5)

    return _ClassificationSettings(model=cfg.model, conf_threshold=cfg.conf_threshold, top_k=cfg.top_k)


# =============================================================================
# Public Entry Point
# =============================================================================


def handle(
    input_file: str,
    output_folder: str,
    config: "ClassificationConfig | None" = None,
    *,
    frame_indices: Optional[List[int]] = None,
    save_annotations: bool = False,
    debug: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Classify video frames using YOLO classification models.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: ClassificationConfig instance or None for defaults.
        frame_indices: List of frame indices to process.
        save_annotations: Whether to save annotated frames with predictions.
        debug: Enable verbose debug output.

    Returns:
        Tuple of (classification_output_folder, results_list).
    """
    print("INFO: Classifying video frames")

    # Extract config values upfront
    settings = _extract_config_settings(config)

    debug_print(f"Classification settings: model={settings.model}, conf_threshold={settings.conf_threshold}, top_k={settings.top_k}", debug=debug)

    # Setup output directories
    classification_folder = os.path.join(output_folder, "classification")
    os.makedirs(classification_folder, exist_ok=True)

    if save_annotations:
        annotations_folder = os.path.join(classification_folder, "annotated")
        os.makedirs(annotations_folder, exist_ok=True)

    # Load classification model
    model, device, use_half = _ensure_yolo_classification_model(
        settings.model,
        debug=debug,
    )

    if model is None:
        print("ERROR: Failed to load classification model")
        return classification_folder, []

    # Get video metadata
    fps = _probe_video_fps(input_file)

    # Process frames
    indices_to_process = frame_indices or []
    if not indices_to_process:
        print("WARNING: No frame indices provided for classification")
        return classification_folder, []

    debug_print(f"Processing {len(indices_to_process)} frames for classification", debug=debug)

    results_list: List[Dict[str, Any]] = []

    for frame_number, frame_img in _iterate_selected_frames(
        input_file,
        indices_to_process,
        debug=debug,
    ):
        # Classify the frame
        predictions = _classify_frame(
            frame_img,
            model,
            use_half=use_half,
            top_k=settings.top_k,
            debug=debug,
        )

        # Filter predictions by confidence threshold
        filtered_predictions = [
            pred for pred in predictions
            if pred["confidence"] >= settings.conf_threshold
        ]

        # Calculate timestamp
        pts_time = frame_number / fps if fps > 0 else 0.0

        # Get frame resolution
        height, width = frame_img.shape[:2]

        # Build frame result entry
        frame_result: Dict[str, Any] = {
            "frame_number": frame_number,
            "frame_path": f"{os.path.abspath(input_file)}#frame_{frame_number:08d}",
            "resolution": {"width": width, "height": height},
            "pts_time": round(pts_time, 4),
            "predictions": filtered_predictions,
        }

        # Save annotated frame if requested
        if save_annotations and filtered_predictions:
            annotation_path = os.path.join(
                annotations_folder,
                f"frame_{frame_number:08d}.png",
            )
            saved_path = _save_annotated_frame(
                frame_img,
                filtered_predictions,
                annotation_path,
                debug=debug,
            )
            if saved_path:
                frame_result["annotated_path"] = saved_path

        results_list.append(frame_result)

        if debug:
            top_pred = filtered_predictions[0] if filtered_predictions else None
            if top_pred:
                debug_print(
                    f"Frame {frame_number}: {top_pred['label']} ({top_pred['confidence']:.2%})",
                    debug=debug,
                )

    # Save JSON output
    output_data = {
        "source_file": os.path.abspath(input_file),
        "model": settings.model,
        "conf_threshold": settings.conf_threshold,
        "top_k": settings.top_k,
        "total_frames_processed": len(results_list),
        "frames": results_list,
    }

    output_file = os.path.join(classification_folder, "classification.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, cls=_NumpyEncoder)

    print(f"INFO: Classification complete. Processed {len(results_list)} frames.")

    return classification_folder, results_list
