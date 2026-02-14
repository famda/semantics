"""Face detection and embedding extraction module.

Uses DeepFace with RetinaFace backend for face detection and Facenet512
for identity-preserving embeddings.  TensorFlow is forced to CPU-only
mode to avoid GPU contention with YOLO models.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import FacesConfig

__all__ = ["handle", "strip_for_results"]

# Suppress TF noise at import time
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DISABLE_XLA", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", module="tensorflow")

# ---------------------------------------------------------------------------
# Cached model loading (no global state mutation — dict-based caching)
# ---------------------------------------------------------------------------

_cache: Dict[str, Any] = {}


def _get_tensorflow(debug: bool) -> Optional[Any]:
    """Load TensorFlow in CPU-only mode (cached after first call)."""
    if "tf" in _cache:
        return _cache["tf"]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if debug else "3"
    try:
        with gray_debug_output(debug):
            tf = importlib.import_module("tensorflow")
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        _cache["tf"] = tf
        return tf
    except Exception:
        _cache["tf"] = None
        return None


def _get_deepface(debug: bool) -> Optional[Any]:
    """Load DeepFace with Keras 3 monkey-patch (cached after first call)."""
    if "deepface" in _cache:
        return _cache["deepface"]
    # Monkey-patch Keras 3 validation
    for mod_path in (
        "deepface.commons.package_utils",
        "retinaface.commons.package_utils",
    ):
        try:
            pkg = importlib.import_module(mod_path)
            pkg.validate_for_keras3 = lambda: None
        except Exception:
            pass
    try:
        with gray_debug_output(debug):
            from deepface import DeepFace
        _cache["deepface"] = DeepFace
        return DeepFace
    except Exception:
        _cache["deepface"] = None
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def handle(
    frames: List[np.ndarray],
    frame_numbers: List[int],
    output_folder: str,
    config: "FacesConfig | None" = None,
    *,
    writer: Optional[Any] = None,
    debug: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    """Detect faces in the given frames and extract embeddings.

    Args:
        frames: List of BGR numpy arrays.
        frame_numbers: Corresponding frame indices.
        output_folder: Root objects output folder (``faces/`` is created inside).
        config: FacesConfig instance or None for defaults.
        writer: Optional ThreadedImageWriter for non-blocking saves.
        debug: Enable verbose output.

    Returns:
        Dict mapping frame_number → list of face detection dicts.
        Each dict contains class_id (100), bounding_box, confidence,
        image_path, embedding (numpy), and embedding_model (str).
    """
    # Extract config values with inline defaults
    conf_threshold = config.face_conf_threshold if config else 0.9
    embedding_model = config.embedding_model_name if config else "Facenet512"
    min_side = config.face_detect_min_side if config else 720
    max_scale = config.face_detect_max_scale if config else 2.0
    backend = config.detector_backend if config else "retinaface"

    # Ensure TF + DeepFace are loaded
    _get_tensorflow(debug)
    deepface = _get_deepface(debug)

    faces_dir = os.path.join(output_folder, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    result: Dict[int, List[Dict[str, Any]]] = {}
    for frame_img, frame_number in zip(frames, frame_numbers):
        faces = _detect_single_frame(
            frame_img,
            frame_number,
            faces_dir,
            deepface=deepface,
            conf_threshold=conf_threshold,
            embedding_model=embedding_model,
            min_side=min_side,
            max_scale=max_scale,
            backend=backend,
            writer=writer,
            debug=debug,
        )
        result[frame_number] = faces

    return result


def strip_for_results(face: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal fields (embedding) before serializing to JSON."""
    public = dict(face)
    public.pop("embedding", None)
    public.pop("embedding_model", None)
    return public


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_single_frame(
    frame_img: np.ndarray,
    frame_number: int,
    faces_dir: str,
    *,
    deepface: Optional[Any],
    conf_threshold: float,
    embedding_model: str,
    min_side: int,
    max_scale: float,
    backend: str,
    writer: Optional[Any],
    debug: bool,
) -> List[Dict[str, Any]]:
    """Detect faces in a single frame, save crops, and extract embeddings."""
    if frame_img is None:
        return []

    h, w = frame_img.shape[:2]

    # Optional upscale for small frames
    scale = max(1.0, min_side / float(min(h, w)))
    scale = min(scale, max_scale)
    det_img = frame_img if scale == 1.0 else cv2.resize(frame_img, None, fx=scale, fy=scale)

    # Run face detection
    raw_faces: list = []
    if deepface is not None:
        try:
            with gray_debug_output(debug):
                raw_faces = deepface.extract_faces(
                    det_img,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                )
        except Exception:
            try:
                raw_faces = deepface.extract_faces(
                    det_img,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                )
            except Exception:
                pass

    if not raw_faces:
        return []

    faces: List[Dict[str, Any]] = []
    for i, det in enumerate(raw_faces):
        conf = det.get("confidence", 0.0)
        if conf < conf_threshold:
            continue

        area = det.get("facial_area", {})
        fx = int(round(area.get("x", 0) / scale))
        fy = int(round(area.get("y", 0) / scale))
        fw = int(round(area.get("w", 0) / scale))
        fh = int(round(area.get("h", 0) / scale))

        x1, y1 = max(0, fx), max(0, fy)
        x2, y2 = min(w, fx + fw), min(h, fy + fh)
        if x2 <= x1 or y2 <= y1:
            continue

        # Save face crop
        crop = frame_img[y1:y2, x1:x2]
        crop_path = os.path.join(faces_dir, f"{frame_number:08d}_{i}.png")
        if writer is not None:
            writer.write(crop_path, crop)
        else:
            cv2.imwrite(crop_path, crop)

        # Extract face embedding
        embedding = None
        face_array = det.get("face")
        if deepface is not None and face_array is not None:
            try:
                with gray_debug_output(debug):
                    result = deepface.represent(
                        face_array,
                        model_name=embedding_model,
                        enforce_detection=False,
                        detector_backend="skip",
                        align=False,
                    )
                if result and isinstance(result, list):
                    embedding = np.array(result[0]["embedding"], dtype=np.float32)
            except Exception:
                pass

        faces.append({
            "class_id": 100,
            "class_name": "face",
            "confidence": float(conf),
            "image_path": os.path.abspath(crop_path),
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "embedding": embedding,
            "embedding_model": embedding_model,
        })

    return faces
