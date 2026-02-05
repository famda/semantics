from __future__ import annotations

import importlib
import json
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

import clip
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from ultralytics import YOLO

from global_helpers import (
    VIDEO_OBJECT_DETECTION_CATEGORY_MAP,
    VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING,
    estimate_dbscan_eps,
    l2_normalize_rows,
)
from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import ObjectsConfig

__all__ = ["handle"]

os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TF_DISABLE_XLA", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# =============================================================================
# Module-level Constants (filtering, not configuration)
# =============================================================================

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
EXCLUDED_CLUSTER_MARKERS = ("_ann", "_mask", "_background", "_polygon", "_msk")


# =============================================================================
# Internal Settings Dataclass
# =============================================================================

from dataclasses import dataclass, field


def _get_objects_defaults() -> dict:
    """Get default values from ObjectsConfig to avoid circular imports."""
    try:
        from config import ObjectsConfig
        cfg = ObjectsConfig()
        return {
            "detection_model": cfg.detection_model,
            "segmentation_model": cfg.segmentation_model,
            "pose_model": cfg.pose_model,
            "object_conf_threshold": cfg.object_conf_threshold,
            "iou_match_threshold": cfg.iou_match_threshold,
            "keypoint_conf_threshold": cfg.keypoint_conf_threshold,
            "face_conf_threshold": cfg.face_conf_threshold,
            "embedding_model_name": cfg.embedding_model_name,
            "face_detect_min_side": cfg.face_detect_min_side,
            "face_detect_max_scale": cfg.face_detect_max_scale,
            "detector_backend": cfg.detector_backend,
            "clip_model_name": cfg.clip_model_name,
            "cluster_base_eps": cfg.cluster_base_eps,
            "cluster_min_samples": cfg.cluster_min_samples,
            "cluster_min_attempts": cfg.cluster_min_attempts,
            "keyframe_eps": cfg.keyframe_eps,
            "keyframe_min_samples": cfg.keyframe_min_samples,
            "keyframe_hamming_frac": cfg.keyframe_hamming_frac,
            "keyframe_require_both": cfg.keyframe_require_both,
        }
    except Exception:
        # Fallback defaults if config import fails
        return {
            "detection_model": "yolo11s.pt",
            "segmentation_model": "yolo11s-seg.pt",
            "pose_model": "yolo11s-pose.pt",
            "object_conf_threshold": 0.80,
            "iou_match_threshold": 0.5,
            "keypoint_conf_threshold": 0.6,
            "face_conf_threshold": 0.9,
            "embedding_model_name": "Facenet512",
            "face_detect_min_side": 720,
            "face_detect_max_scale": 2.0,
            "detector_backend": "retinaface",
            "clip_model_name": "ViT-B/32",
            "cluster_base_eps": 0.35,
            "cluster_min_samples": 1,
            "cluster_min_attempts": 4,
            "keyframe_eps": 0.12,
            "keyframe_min_samples": 1,
            "keyframe_hamming_frac": 0.30,
            "keyframe_require_both": True,
        }


@dataclass
class _ObjectsSettings:
    """Internal settings container for objects module.
    
    Defaults are sourced from ObjectsConfig in config.py.
    """

    detection_model: str = field(default_factory=lambda: _get_objects_defaults()["detection_model"])
    segmentation_model: str = field(default_factory=lambda: _get_objects_defaults()["segmentation_model"])
    pose_model: str = field(default_factory=lambda: _get_objects_defaults()["pose_model"])
    object_conf_threshold: float = field(default_factory=lambda: _get_objects_defaults()["object_conf_threshold"])
    iou_match_threshold: float = field(default_factory=lambda: _get_objects_defaults()["iou_match_threshold"])
    keypoint_conf_threshold: float = field(default_factory=lambda: _get_objects_defaults()["keypoint_conf_threshold"])
    face_conf_threshold: float = field(default_factory=lambda: _get_objects_defaults()["face_conf_threshold"])
    embedding_model_name: str = field(default_factory=lambda: _get_objects_defaults()["embedding_model_name"])
    face_detect_min_side: int = field(default_factory=lambda: _get_objects_defaults()["face_detect_min_side"])
    face_detect_max_scale: float = field(default_factory=lambda: _get_objects_defaults()["face_detect_max_scale"])
    detector_backend: str = field(default_factory=lambda: _get_objects_defaults()["detector_backend"])
    clip_model_name: str = field(default_factory=lambda: _get_objects_defaults()["clip_model_name"])
    cluster_base_eps: float = field(default_factory=lambda: _get_objects_defaults()["cluster_base_eps"])
    cluster_min_samples: int = field(default_factory=lambda: _get_objects_defaults()["cluster_min_samples"])
    cluster_min_attempts: int = field(default_factory=lambda: _get_objects_defaults()["cluster_min_attempts"])
    keyframe_eps: float = field(default_factory=lambda: _get_objects_defaults()["keyframe_eps"])
    keyframe_min_samples: int = field(default_factory=lambda: _get_objects_defaults()["keyframe_min_samples"])
    keyframe_hamming_frac: float = field(default_factory=lambda: _get_objects_defaults()["keyframe_hamming_frac"])
    keyframe_require_both: bool = field(default_factory=lambda: _get_objects_defaults()["keyframe_require_both"])


# =============================================================================
# Cached Model Singletons (acceptable pattern - not mutated during execution)
# =============================================================================

_TF_MODULE: Optional[Any] = None

_DEEPFACE_CLASS: Optional[Any] = None
_CLIP_MODEL: Optional[Any] = None
_CLIP_PREPROCESS: Optional[Any] = None
_CLIP_DEVICE: Optional[torch.device] = None

_YOLO_DEVICE: Optional[torch.device] = None
_YOLO_OBJECT_MODEL: Optional[YOLO] = None
_YOLO_SEGMENTATION_MODEL: Optional[YOLO] = None
_YOLO_POSE_MODEL: Optional[YOLO] = None
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



# ---------------------------------------------------------------------------
# TensorFlow / DeepFace helpers
# ---------------------------------------------------------------------------

def _ensure_tensorflow(debug: bool) -> Optional[Any]:
    global _TF_MODULE
    if _TF_MODULE is not None:
        return _TF_MODULE
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if debug else "3"
    try:
        with gray_debug_output(debug):
            tf_module = importlib.import_module("tensorflow")
    except Exception:
        return None
    try:
        tf_module.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    _TF_MODULE = tf_module
    return _TF_MODULE


def _ensure_deepface(debug: bool) -> Optional[Any]:
    global _DEEPFACE_CLASS
    if _DEEPFACE_CLASS is not None:
        return _DEEPFACE_CLASS
    with gray_debug_output(debug):
        from deepface import DeepFace as _DeepFace  # type: ignore

    _DEEPFACE_CLASS = _DeepFace
    return _DEEPFACE_CLASS



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _strip_face_for_results(face: Dict[str, Any]) -> Dict[str, Any]:
    public_face = dict(face)
    public_face.pop("embedding", None)
    public_face.pop("embedding_model", None)
    return public_face


def _iterate_selected_frames(
    video_path: str,
    indices: Sequence[int],
    *,
    debug: bool = False,
) -> Iterator[Tuple[int, np.ndarray]]:
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
                (
                    "INFO: Skipping %d frame(s) outside valid range [0, %d] for '%s'"
                    " (examples: %s)"
                )
                % (len(clipped), max_index, os.path.basename(video_path), sample),
                debug=debug,
            )
        ordered = in_range
    else:
        ordered = [idx for idx in ordered if idx >= 0]

    if ordered:
        probe_cap = cv2.VideoCapture(video_path)
        if probe_cap.isOpened():
            probe_index = ordered[-1]
            attempts = 0
            real_max: Optional[int] = None
            while probe_index >= 0 and attempts < 512:
                seek_ok = probe_cap.set(cv2.CAP_PROP_POS_FRAMES, probe_index)
                ret = False
                if seek_ok:
                    ret, _ = probe_cap.read()
                if ret:
                    real_max = probe_index
                    break
                probe_index -= 1
                attempts += 1
            probe_cap.release()

            if real_max is not None and real_max < ordered[-1]:
                trimmed = [idx for idx in ordered if idx > real_max]
                if trimmed:
                    sample = ", ".join(str(item) for item in trimmed[:5])
                    debug_print(
                        (
                            "INFO: Dropping %d frame(s) beyond decode range (max=%d) for '%s'"
                            " (examples: %s)"
                        )
                        % (len(trimmed), real_max, os.path.basename(video_path), sample),
                        debug=debug,
                    )
                ordered = [idx for idx in ordered if idx <= real_max]

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
                        (
                            "INFO: Decoder hit end-of-stream after %d frame(s); skipping %d "
                            "pending index(es) (next: %s)."
                        )
                        % (processed, remaining, missing_preview),
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0
    finally:
        cap.release()


def _prepare_class_dir(cache: Dict[str, str], base_dir: str, class_name: str) -> str:
    if class_name not in cache:
        safe_name = class_name.replace(" ", "_")
        class_path = os.path.join(base_dir, safe_name)
        os.makedirs(class_path, exist_ok=True)
        cache[class_name] = class_path
    return cache[class_name]


def _list_valid_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    valid: List[str] = []
    for name in os.listdir(folder):
        lower = name.lower()
        if not lower.endswith(VALID_IMAGE_EXTENSIONS):
            continue
        if lower.startswith("_") or any(marker in lower for marker in EXCLUDED_CLUSTER_MARKERS):
            continue
        abs_path = os.path.join(folder, name)
        if os.path.isfile(abs_path):
            valid.append(abs_path)
    return sorted(valid)


def _ensure_clip_resources(
    debug: bool,
    clip_model_name: str,
) -> Tuple[Optional[Any], Optional[Any], Optional[torch.device]]:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None and _CLIP_DEVICE is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with gray_debug_output(debug):
            model, preprocess = clip.load(clip_model_name, device=device)
        model.eval()
    except Exception as exc:
        print(f"ERROR: Unable to load CLIP model '{clip_model_name}': {exc}")
        return None, None, None

    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_DEVICE = device
    return model, preprocess, device


def _extract_clip_features(
    image_paths: Sequence[str],
    *,
    debug: bool,
    clip_model_name: str,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Extract CLIP features from images with batch processing for performance.
    
    Args:
        image_paths: Sequence of image file paths.
        debug: Enable debug output.
        clip_model_name: CLIP model variant to use.
        batch_size: Number of images to process in each batch (default 32).
        
    Returns:
        Dictionary mapping image path to feature vector.
    """
    model, preprocess, device = _ensure_clip_resources(debug, clip_model_name)
    if model is None or preprocess is None or device is None:
        return {}

    feature_map: Dict[str, np.ndarray] = {}
    
    # Process images in batches for better GPU utilization
    valid_paths: List[str] = []
    valid_tensors: List[torch.Tensor] = []
    
    for path in image_paths:
        try:
            with Image.open(path) as img:
                tensor = preprocess(img.convert("RGB"))
            valid_paths.append(path)
            valid_tensors.append(tensor)
        except FileNotFoundError:
            debug_print(f"WARNING: Image for clustering not found: {path}", debug=debug)
        except Exception as exc:
            debug_print(f"WARNING: Failed to process image '{path}' for clustering: {exc}", debug=debug)
    
    if not valid_tensors:
        return feature_map
    
    # Process in batches
    for i in range(0, len(valid_tensors), batch_size):
        batch_paths = valid_paths[i:i + batch_size]
        batch_tensors = torch.stack(valid_tensors[i:i + batch_size]).to(device)
        
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensors)
        
        batch_features_np = batch_features.detach().cpu().numpy().astype(np.float32, copy=True)
        
        for idx, path in enumerate(batch_paths):
            feature_map[path] = batch_features_np[idx]

    return feature_map


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    n_ = np.arange(n)[None, :]
    mat = np.cos(np.pi * (n_ + 0.5) * k / n)
    mat[0, :] = mat[0, :] / np.sqrt(n)
    mat[1:, :] = mat[1:, :] * np.sqrt(2 / n)
    return mat.astype(np.float32)


def _dct_2d(a: np.ndarray) -> np.ndarray:
    n, m = a.shape
    return _dct_matrix(n) @ a @ _dct_matrix(m).T


def _phash64(path: str) -> int:
    try:
        with Image.open(path) as img:
            img_l = img.convert("L")
            hash_size = 8
            highfreq_factor = 4
            size = hash_size * highfreq_factor
            resized = img_l.resize((size, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(resized, dtype=np.float32)
            dct = _dct_2d(pixels)
            low = dct[:hash_size, :hash_size].flatten()
            if low.size <= 1:
                threshold = low[0] if low.size == 1 else 0.0
            else:
                threshold = float(np.median(low[1:]))
            bits = (low > threshold).astype(np.uint8)
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            return int(value)
    except Exception:
        return 0


def _copy_with_unique_name(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    stem, ext = os.path.splitext(base)
    candidate = os.path.join(dst_dir, base)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dst_dir, f"{stem}_{counter}{ext}")
        counter += 1
    shutil.copy2(src, candidate)
    return os.path.abspath(candidate)


def _cleanup_annotation_outputs(base_dir: str) -> None:
    if not os.path.isdir(base_dir):
        return

    for entry in list(os.listdir(base_dir)):
        if entry == "clusters":
            continue
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except Exception:
                pass

    try:
        remaining = os.listdir(base_dir)
    except OSError:
        return

    if not remaining:
        shutil.rmtree(base_dir, ignore_errors=True)


def _select_keyframes(
    source_paths: Sequence[str],
    feature_map: Dict[str, np.ndarray],
    *,
    eps: float,
    min_samples: int,
    hamming_frac: float,
    require_both: bool,
    debug: bool,
) -> List[str]:
    if not source_paths:
        return []

    feats = [feature_map.get(path) for path in source_paths]
    feats = [f for f in feats if f is not None]
    if not feats:
        return []

    feature_matrix = l2_normalize_rows(np.stack(feats, axis=0))
    try:
        sub_dbscan = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
            metric="cosine",
            n_jobs=-1,
        )
        labels = sub_dbscan.fit_predict(feature_matrix)
    except Exception as exc:
        debug_print(f"WARNING: Secondary clustering for keyframes failed: {exc}", debug=debug)
        labels = np.zeros((len(source_paths),), dtype=int)

    labels = labels.copy()
    unique_labels = sorted(set(int(lbl) for lbl in labels if lbl >= 0))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for idx, lbl in enumerate(labels):
        if lbl >= 0:
            labels[idx] = label_map[int(lbl)]

    # Compute pHash values in parallel for better performance on large image sets
    with ThreadPoolExecutor(max_workers=min(8, len(source_paths))) as executor:
        phashes = list(executor.map(_phash64, source_paths))
    
    selected_indices: List[int] = []
    considered_clusters = sorted(set(int(lbl) for lbl in labels if lbl >= 0))

    for cluster_idx in considered_clusters:
        member_indices = [i for i, lbl in enumerate(labels) if int(lbl) == cluster_idx]
        chosen: List[int] = []
        for member in member_indices:
            if not chosen:
                chosen.append(member)
                continue
            cos_distances: List[float] = []
            hamm_distances: List[float] = []
            feat_member = feature_matrix[member] / (np.linalg.norm(feature_matrix[member]) + 1e-8)
            for existing in chosen:
                feat_existing = feature_matrix[existing] / (np.linalg.norm(feature_matrix[existing]) + 1e-8)
                cos_distances.append(1.0 - float(np.dot(feat_member, feat_existing)))
                xor_val = (phashes[member] ^ phashes[existing]) & ((1 << 64) - 1)
                hamm_distances.append(xor_val.bit_count() / 64.0)
            min_cos = min(cos_distances) if cos_distances else 1.0
            min_hamm = min(hamm_distances) if hamm_distances else 1.0
            different = (min_cos >= eps) or (min_hamm >= hamming_frac)
            if require_both:
                different = (min_cos >= eps) and (min_hamm >= hamming_frac)
            if different:
                chosen.append(member)
        selected_indices.extend(chosen)

    if not selected_indices:
        selected_indices = [0]

    unique_selected = sorted(set(selected_indices))
    return [source_paths[idx] for idx in unique_selected if idx < len(source_paths)]


def _determine_half_precision(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 6


def _prepare_yolo_model(model: YOLO, device: torch.device, *, use_half: bool, debug: bool) -> YOLO:
    try:
        model.to(device)
    except Exception:
        pass
    try:
        model.fuse()
    except Exception:
        pass

    if use_half:
        try:
            model.model.half()  # type: ignore[attr-defined]
        except Exception:
            debug_print("WARNING: Failed to enable half precision for YOLO model; using float32 instead.", debug=debug)
            use_half = False
            try:
                model.model.float()  # type: ignore[attr-defined]
            except Exception:
                pass
    return model


def _ensure_yolo_models(
    *,
    debug: bool,
    require_segmentation: bool,
    require_pose: bool,
    detection_model_name: str = "yolo11s.pt",
    segmentation_model_name: str = "yolo11s-seg.pt",
    pose_model_name: str = "yolo11s-pose.pt",
) -> Tuple[YOLO, Optional[YOLO], Optional[YOLO], torch.device]:
    global _YOLO_OBJECT_MODEL, _YOLO_SEGMENTATION_MODEL, _YOLO_POSE_MODEL, _YOLO_DEVICE, _YOLO_USE_HALF_PRECISION

    if _YOLO_DEVICE is None:
        _YOLO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _YOLO_USE_HALF_PRECISION = _determine_half_precision(_YOLO_DEVICE)

    device = _YOLO_DEVICE

    if _YOLO_OBJECT_MODEL is None:
        with gray_debug_output(debug):
            old_cwd = os.getcwd()
            try:
                os.chdir("/platform")
                object_model = YOLO(detection_model_name)
            finally:
                os.chdir(old_cwd)
        _YOLO_OBJECT_MODEL = _prepare_yolo_model(object_model, device, use_half=_YOLO_USE_HALF_PRECISION, debug=debug)
        try:
            _YOLO_OBJECT_MODEL.model.eval()  # type: ignore[attr-defined]
        except Exception:
            pass

    if require_segmentation and _YOLO_SEGMENTATION_MODEL is None:
        with gray_debug_output(debug):
            old_cwd = os.getcwd()
            try:
                os.chdir("/platform")
                segmentation_model = YOLO(segmentation_model_name)
            finally:
                os.chdir(old_cwd)
        _YOLO_SEGMENTATION_MODEL = _prepare_yolo_model(segmentation_model, device, use_half=_YOLO_USE_HALF_PRECISION, debug=debug)
        try:
            _YOLO_SEGMENTATION_MODEL.model.eval()  # type: ignore[attr-defined]
        except Exception:
            pass

    if require_pose and _YOLO_POSE_MODEL is None:
        with gray_debug_output(debug):
            old_cwd = os.getcwd()
            try:
                os.chdir("/platform")
                pose_model = YOLO(pose_model_name)
            finally:
                os.chdir(old_cwd)
        _YOLO_POSE_MODEL = _prepare_yolo_model(pose_model, device, use_half=_YOLO_USE_HALF_PRECISION, debug=debug)
        try:
            _YOLO_POSE_MODEL.model.eval()  # type: ignore[attr-defined]
        except Exception:
            pass

    return (
        _YOLO_OBJECT_MODEL,
        _YOLO_SEGMENTATION_MODEL if require_segmentation else None,
        _YOLO_POSE_MODEL if require_pose else None,
        device,
    )


def _cluster_class_directory(
    class_name: str,
    image_folder: str,
    *,
    debug: bool,
    base_eps: float,
    min_samples: int,
    key_eps: float,
    key_min_samples: int,
    key_hamming_frac: float,
    key_require_both: bool,
    cluster_min_attempts: int,
    clip_model_name: str,
) -> Optional[Dict[str, Any]]:
    images = _list_valid_images(image_folder)
    if not images:
        print(f"INFO: Skipping clustering for '{class_name}' (no valid images).")
        return None

    feature_map = _extract_clip_features(images, debug=debug, clip_model_name=clip_model_name)
    ordered_paths = [path for path in images if path in feature_map]
    if len(ordered_paths) < min_samples:
        print(
            f"WARNING: Not enough valid images with features for '{class_name}' clustering (found {len(ordered_paths)})."
        )
        return None

    feature_matrix = l2_normalize_rows(np.stack([feature_map[path] for path in ordered_paths], axis=0))

    attempt_eps = float(base_eps)
    labels: Optional[np.ndarray] = None
    effective_eps = float(base_eps)
    for attempt in range(cluster_min_attempts):
        eps_candidate = estimate_dbscan_eps(feature_matrix, attempt_eps, min_samples)
        if not np.isfinite(eps_candidate) or eps_candidate <= 0:
            eps_candidate = attempt_eps
        eps_candidate = max(0.05, min(float(eps_candidate), 0.95))
        try:
            dbscan = DBSCAN(
                eps=eps_candidate,
                min_samples=int(min_samples),
                metric="cosine",
                n_jobs=-1,
            )
            candidate_labels = dbscan.fit_predict(feature_matrix)
        except Exception as exc:
            debug_print(f"WARNING: Primary clustering failed on attempt {attempt + 1}: {exc}", debug=debug)
            candidate_labels = np.full((feature_matrix.shape[0],), -1, dtype=int)

        candidate_labels = candidate_labels.copy()
        unique_positive = sorted(set(int(lbl) for lbl in candidate_labels if lbl >= 0))
        mapping = {old: new for new, old in enumerate(unique_positive)}
        for idx, lbl in enumerate(candidate_labels):
            if lbl >= 0:
                candidate_labels[idx] = mapping[int(lbl)]

        cluster_count = len(unique_positive)
        if cluster_count > 1 or attempt == cluster_min_attempts - 1:
            labels = candidate_labels
            effective_eps = eps_candidate
            if cluster_count <= 1:
                debug_print(
                    f"INFO: '{class_name}' clustering yielded a single cluster even after tuning (attempt eps={eps_candidate:.3f}).",
                    debug=debug,
                )
            break
        attempt_eps *= 0.85

    if labels is None:
        print(f"WARNING: Clustering failed for '{class_name}'.")
        return None

    clusters_root = os.path.join(image_folder, "clusters")
    if os.path.isdir(clusters_root):
        shutil.rmtree(clusters_root, ignore_errors=True)
    os.makedirs(clusters_root, exist_ok=True)

    cluster_details: List[Dict[str, Any]] = []
    noise_paths: List[str] = []

    unique_labels = sorted(set(int(lbl) for lbl in labels if lbl >= 0))
    for cluster_idx in unique_labels:
        cluster_dir = os.path.join(clusters_root, f"cluster_{cluster_idx:03d}")
        os.makedirs(cluster_dir, exist_ok=True)
        member_sources = [path for path, lbl in zip(ordered_paths, labels) if int(lbl) == cluster_idx]
        copied_paths: List[str] = []
        for src in member_sources:
            copied_paths.append(_copy_with_unique_name(src, cluster_dir))

        keyframe_sources = _select_keyframes(
            member_sources,
            feature_map,
            eps=key_eps,
            min_samples=key_min_samples,
            hamming_frac=key_hamming_frac,
            require_both=key_require_both,
            debug=debug,
        )

        keyframe_paths: List[str] = []
        if keyframe_sources:
            keyframe_dir = os.path.join(cluster_dir, "keyframes")
            for src in keyframe_sources:
                keyframe_paths.append(_copy_with_unique_name(src, keyframe_dir))

        cluster_details.append(
            {
                "cluster_id": int(cluster_idx),
                "cluster_folder": os.path.abspath(cluster_dir),
                "image_paths": copied_paths,
                "image_count": len(copied_paths),
                "keyframes": keyframe_paths,
                "keyframe_count": len(keyframe_paths),
                "effective_eps": effective_eps,
            }
        )

    noise_dir = None
    if any(lbl < 0 for lbl in labels):
        noise_dir = os.path.join(clusters_root, "noise")
        os.makedirs(noise_dir, exist_ok=True)
        for src, lbl in zip(ordered_paths, labels):
            if int(lbl) == -1:
                noise_paths.append(_copy_with_unique_name(src, noise_dir))

    summary = {
        "class_name": class_name,
        "image_folder": os.path.abspath(image_folder),
        "clusters_root": os.path.abspath(clusters_root),
        "cluster_count": len(cluster_details),
        "noise_count": len(noise_paths),
        "total_images": len(ordered_paths),
        "clusters": cluster_details,
        "noise": {
            "folder": os.path.abspath(noise_dir) if noise_dir else None,
            "image_paths": noise_paths,
            "count": len(noise_paths),
        },
    }

    report_path = os.path.join(clusters_root, "clusters.json")
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=4)
        summary["clusters_json"] = os.path.abspath(report_path)
    except Exception as exc:
        print(f"WARNING: Failed to write cluster summary for '{class_name}': {exc}")

    return summary


def _match_segmentation(
    bbox: np.ndarray,
    seg_detections: Optional[sv.Detections],
    *,
    iou_match_threshold: float,
) -> Optional[Tuple[int, np.ndarray]]:
    if (
        seg_detections is None
        or seg_detections.xyxy is None
        or len(seg_detections) == 0
    ):
        return None
    seg_boxes = np.asarray(seg_detections.xyxy, dtype=np.float32)
    if seg_boxes.size == 0:
        return None
    ious = sv.box_iou_batch(np.asarray([bbox], dtype=np.float32), seg_boxes)
    if ious.size == 0:
        return None
    best_idx = int(np.argmax(ious[0]))
    best_iou = float(ious[0, best_idx])
    if best_iou < iou_match_threshold:
        return None
    if seg_detections.mask is None or len(seg_detections.mask) <= best_idx:
        return None
    return best_idx, seg_detections.mask[best_idx]


def _save_segmentation_artifacts(
    frame_img: np.ndarray,
    seg_detections: sv.Detections,
    seg_index: int,
    mask: np.ndarray,
    class_dir: str,
    frame_number: int,
    file_stem: str,
    mask_annotator: sv.MaskAnnotator,
    polygon_annotator: sv.PolygonAnnotator,
) -> Dict[str, Optional[str]]:
    mask_folder = os.path.join(class_dir, "masks")
    os.makedirs(mask_folder, exist_ok=True)

    mask_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}.png")
    binary_mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, binary_mask)

    try:
        detection = sv.Detections(
            xyxy=np.asarray([seg_detections.xyxy[seg_index]], dtype=np.float32),
            mask=np.asarray([mask.astype(bool)], dtype=bool),
            class_id=np.zeros(1, dtype=np.int32),
        )
    except Exception:
        detection = None

    contour_path = None
    if detection is not None:
        try:
            contour_img = polygon_annotator.annotate(scene=frame_img.copy(), detections=detection)
            contour_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}_polygon.png")
            cv2.imwrite(contour_path, contour_img)
        except Exception:
            contour_path = None

    background_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}_background.png")
    frame_bgra = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
    frame_bgra[:, :, 3] = binary_mask
    cv2.imwrite(background_path, frame_bgra)

    overlay_path = None
    if detection is not None:
        try:
            overlay = mask_annotator.annotate(scene=frame_img.copy(), detections=detection)
            overlay_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}_mask.png")
            cv2.imwrite(overlay_path, overlay)
        except Exception:
            overlay_path = None

    polygon_path = contour_path

    return {
        "image_path": mask_path,
        "contour_image_path": contour_path,
        "background_image_path": background_path,
        "mask_overlay_path": overlay_path,
        "polygon_overlay_path": polygon_path,
    }


def _match_keypoints(
    bbox: np.ndarray,
    keypoint_detections: Optional[sv.Detections],
    keypoints: Optional[sv.KeyPoints],
    class_name: str,
    *,
    keypoint_conf_threshold: float,
    iou_match_threshold: float,
) -> List[Dict[str, Any]]:
    if (
        keypoint_detections is None
        or keypoint_detections.xyxy is None
        or len(keypoint_detections) == 0
        or keypoints is None
    ):
        return []

    kp_xy_attr = getattr(keypoints, "xy", None)
    kp_conf_attr = getattr(keypoints, "confidence", None)
    if kp_conf_attr is None:
        kp_conf_attr = getattr(keypoints, "conf", None)
    if kp_xy_attr is None or kp_conf_attr is None:
        return []

    pose_boxes = np.asarray(keypoint_detections.xyxy, dtype=np.float32)
    if pose_boxes.size == 0:
        return []

    ious = sv.box_iou_batch(np.asarray([bbox], dtype=np.float32), pose_boxes)
    if ious.size == 0:
        return []

    best_idx = int(np.argmax(ious[0]))
    best_iou = float(ious[0, best_idx])
    if best_iou < iou_match_threshold:
        return []

    def _to_np(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    kp_xy = _to_np(kp_xy_attr)
    kp_conf = _to_np(kp_conf_attr)

    if best_idx >= kp_xy.shape[0]:
        return []

    kp_names = VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.get(class_name, [])
    if not kp_names:
        return []

    kp_xy = kp_xy[best_idx]
    kp_conf = kp_conf[best_idx]

    results: List[Dict[str, Any]] = []
    for idx, kp_name in enumerate(kp_names):
        if idx >= kp_xy.shape[0]:
            break
        if kp_conf[idx] <= keypoint_conf_threshold:
            continue
        results.append(
            {
                "class_id": idx,
                "class_name": kp_name,
                "point": {"x": float(kp_xy[idx][0]), "y": float(kp_xy[idx][1])},
                "confidence": float(kp_conf[idx]),
            }
        )
    return results


def _progress_iter(
    iterable: Iterable[Tuple[int, np.ndarray]],
    *,
    total: Optional[int],
    debug: bool,
) -> Tuple[Iterable[Tuple[int, np.ndarray]], contextmanager]:
    if not debug:
        return iterable, nullcontext()

    try:
        progress = tqdm(iterable, desc="Objects", unit="frame", colour="#888888", total=total)
    except TypeError:
        progress = tqdm(iterable, desc="Objects", unit="frame", total=total)

    @contextmanager
    def _ctx():
        with gray_debug_output(True):
            try:
                yield
            finally:
                close_fn = getattr(progress, "close", None)
                if callable(close_fn):
                    close_fn()

    return progress, _ctx()


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "ObjectsConfig | None" = None,
    *,
    object_classes: Optional[List[str]] = None,
    frame_indices: Optional[List[int]] = None,
    perform_clustering: bool = False,
    save_annotations: bool = False,
    debug: bool = False,
):
    """Main entry point for object detection.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: ObjectsConfig instance or None for defaults.
        object_classes: List of object class names to detect.
        frame_indices: List of frame indices to process.
        perform_clustering: Whether to cluster detected objects.
        save_annotations: Whether to save annotated images.
        debug: Enable verbose debug output.

    Returns:
        Tuple of (output_folder, results_list).
    """
    print("INFO: Detecting objects present in the frames")

    # Extract ALL config values upfront - use config values or ObjectsSettings defaults
    if config:
        settings = _ObjectsSettings(
            detection_model=config.detection_model,
            segmentation_model=config.segmentation_model,
            pose_model=config.pose_model,
            object_conf_threshold=config.object_conf_threshold,
            iou_match_threshold=config.iou_match_threshold,
            keypoint_conf_threshold=config.keypoint_conf_threshold,
            face_conf_threshold=config.face_conf_threshold,
            embedding_model_name=config.embedding_model_name,
            face_detect_min_side=config.face_detect_min_side,
            face_detect_max_scale=config.face_detect_max_scale,
            detector_backend=config.detector_backend,
            clip_model_name=config.clip_model_name,
            cluster_base_eps=config.cluster_base_eps,
            cluster_min_samples=config.cluster_min_samples,
            cluster_min_attempts=config.cluster_min_attempts,
            keyframe_eps=config.keyframe_eps,
            keyframe_min_samples=config.keyframe_min_samples,
            keyframe_hamming_frac=config.keyframe_hamming_frac,
            keyframe_require_both=config.keyframe_require_both,
        )
    else:
        settings = _ObjectsSettings()

    return _detect(
        input_file,
        output_folder,
        object_classes,
        frame_indices=frame_indices,
        debug=debug,
        perform_clustering=perform_clustering,
        save_annotations=save_annotations,
        settings=settings,
    )


def _detect(
    video_file: str,
    output_folder: str,
    classes_to_detect: Optional[List[str]],
    frame_indices: Optional[List[int]] = None,
    debug: bool = False,
    *,
    perform_clustering: bool = True,
    save_annotations: bool = False,
    settings: Optional[_ObjectsSettings] = None,
):
    """Internal detection implementation."""
    if settings is None:
        settings = _ObjectsSettings()

    _ensure_tensorflow(debug)

    output_folder = os.path.join(output_folder, "objects")

    save_annotations = bool(save_annotations)
    should_save_detection_images = save_annotations or perform_clustering

    if classes_to_detect is not None:
        classes_to_detect = {
            VIDEO_OBJECT_DETECTION_CATEGORY_MAP[class_name]
            for class_name in classes_to_detect
            if class_name in VIDEO_OBJECT_DETECTION_CATEGORY_MAP
        }

    keypoint_classes = set(VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.keys())
    keypoint_class_ids = {
        VIDEO_OBJECT_DETECTION_CATEGORY_MAP[name]
        for name in keypoint_classes
        if name in VIDEO_OBJECT_DETECTION_CATEGORY_MAP
    }
    pose_required = classes_to_detect is None or bool(keypoint_class_ids.intersection(set(classes_to_detect or [])))

    object_model, segmentation_model, pose_model, inference_device = _ensure_yolo_models(
        debug=debug,
        require_segmentation=True,
        require_pose=pose_required,
        detection_model_name=settings.detection_model,
        segmentation_model_name=settings.segmentation_model,
        pose_model_name=settings.pose_model,
    )
    debug_print(f"INFO: Using device: {inference_device}", debug=debug)

    if frame_indices is None or not isinstance(frame_indices, list) or not frame_indices:
        print("No frame indexes provided for object detection")
        return output_folder, []

    coerced_indices: List[int] = []
    for idx in frame_indices:
        if isinstance(idx, (int, np.integer)):
            coerced_indices.append(int(idx))
            continue
        if isinstance(idx, float) and idx.is_integer():
            coerced_indices.append(int(idx))
            continue
        if isinstance(idx, str):
            stripped = idx.strip()
            if stripped.isdigit():
                coerced_indices.append(int(stripped))
    selected_indices = sorted(set(i for i in coerced_indices if i >= 0))
    debug_print(f"Selected {len(selected_indices)} frames for object detection", debug=debug)

    if not selected_indices:
        print("No frames available for object detection")
        return output_folder, []

    os.makedirs(output_folder, exist_ok=True)
    video_abs_path = os.path.abspath(video_file)
    fps = _probe_video_fps(video_file)

    tracker_obj = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    polygon_annotator = sv.PolygonAnnotator()

    class_dir_cache: Dict[str, str] = {}
    frame_iterator = _iterate_selected_frames(video_file, selected_indices, debug=debug)

    iterable, progress_ctx = _progress_iter(frame_iterator, total=len(selected_indices), debug=debug)
    if not debug:
        print("INFO: Processing frames for object detection")

    results_list: List[Dict[str, Any]] = []
    all_faces_data: List[Dict[str, Any]] = []
    processed_frames = 0

    with progress_ctx:
        for frame_number, frame_img in iterable:
            if frame_img is None:
                continue

            processed_frames += 1
            frame_reference = f"{video_abs_path}#frame_{frame_number:08d}"
            height, width = frame_img.shape[:2]

            frame_results: Dict[str, Any] = {
                "frame_number": frame_number,
                "frame_path": frame_reference,
                "resolution": {"width": width, "height": height},
                "detections": [],
            }

            time_value = (float(frame_number) / fps) if fps > 0 else None
            if time_value is not None:
                try:
                    frame_results["pts_time"] = float(time_value)
                except (TypeError, ValueError):
                    pass

            faces = _detect_faces(
                frame_number,
                output_folder,
                save_faces=should_save_detection_images,
                frame_img=frame_img,
                debug=debug,
                settings=settings,
            )
            all_faces_data.extend(faces)
            frame_results["detections"].extend(_strip_face_for_results(face) for face in faces)

            with torch.inference_mode():
                obj_predictions = object_model(
                    frame_img,
                    conf=settings.object_conf_threshold,
                    verbose=False,
                )
            if not obj_predictions:
                results_list.append(frame_results)
                continue

            obj_result = obj_predictions[0]
            names = obj_result.names or {}
            detections = sv.Detections.from_ultralytics(obj_result)
            if len(detections) == 0:
                results_list.append(frame_results)
                continue

            tracked_detections = tracker_obj.update_with_detections(detections)

            seg_detections = None
            if segmentation_model is not None:
                with torch.inference_mode():
                    seg_predictions = segmentation_model(
                        frame_img,
                        conf=settings.object_conf_threshold,
                        verbose=False,
                    )
                seg_detections = sv.Detections.from_ultralytics(seg_predictions[0]) if seg_predictions else None

            keypoint_detections = None
            keypoints = None
            frame_requires_pose = False
            if pose_model is not None:
                detected_class_names = {
                    names.get(int(cls_id), "")
                    for cls_id in tracked_detections.class_id
                    if cls_id is not None
                }
                frame_requires_pose = bool(detected_class_names & keypoint_classes)

            if frame_requires_pose and pose_model is not None:
                with torch.inference_mode():
                    pose_predictions = pose_model(
                        frame_img,
                        conf=settings.object_conf_threshold,
                        verbose=False,
                    )
                keypoint_detections = sv.Detections.from_ultralytics(pose_predictions[0]) if pose_predictions else None
                keypoints = pose_predictions[0].keypoints if pose_predictions else None

            detection_index_counter = 0
            for idx in range(len(tracked_detections)):
                class_id = tracked_detections.class_id[idx]
                tracker_id = tracked_detections.tracker_id[idx]
                confidence = tracked_detections.confidence[idx]
                if confidence is None or confidence < settings.object_conf_threshold:
                    continue
                if classes_to_detect is not None and class_id not in classes_to_detect:
                    continue

                bbox = np.asarray(tracked_detections.xyxy[idx], dtype=np.float32)
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cropped = frame_img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                class_name = names.get(class_id, "N/A") if class_id is not None else "N/A"
                detection_stem = f"{frame_number:08d}_{detection_index_counter}"

                class_dir: Optional[str] = None
                cropped_path: Optional[str] = None
                if should_save_detection_images:
                    class_dir = _prepare_class_dir(class_dir_cache, output_folder, class_name)
                    cropped_path = os.path.join(class_dir, f"{detection_stem}.png")
                    cv2.imwrite(cropped_path, cropped)

                if save_annotations and class_name.lower() == "person" and class_dir is not None:
                    dets = sv.Detections(
                        xyxy=np.asarray([[x1, y1, x2, y2]], dtype=np.float32),
                        confidence=np.asarray([confidence], dtype=np.float32),
                        class_id=np.asarray([class_id if class_id is not None else 0], dtype=np.int32),
                    )
                    annotated = box_annotator.annotate(scene=frame_img.copy(), detections=dets)
                    annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=[class_name])
                    ann_path = os.path.join(class_dir, f"{detection_stem}_ann.png")
                    cv2.imwrite(ann_path, annotated)

                segmentation_info: Dict[str, Optional[str]] = {}
                if save_annotations and class_dir is not None:
                    seg_match = _match_segmentation(
                        bbox, seg_detections,
                        iou_match_threshold=settings.iou_match_threshold,
                    )
                    if seg_match is not None and seg_detections is not None:
                        seg_idx, mask = seg_match
                        mask_payload = _save_segmentation_artifacts(
                            frame_img,
                            seg_detections,
                            seg_idx,
                            mask,
                            class_dir,
                            frame_number,
                            f"{tracker_id if tracker_id is not None else detection_index_counter}",
                            mask_annotator,
                            polygon_annotator,
                        )
                        segmentation_info = {
                            "image_path": mask_payload.get("image_path"),
                            "background_image_path": mask_payload.get("background_image_path"),
                            "mask_overlay_path": mask_payload.get("mask_overlay_path"),
                            "polygon_image_path": mask_payload.get("polygon_overlay_path") or mask_payload.get("contour_image_path"),
                        }

                keypoints_payload = _match_keypoints(
                    bbox, keypoint_detections, keypoints, class_name,
                    keypoint_conf_threshold=settings.keypoint_conf_threshold,
                    iou_match_threshold=settings.iou_match_threshold,
                )

                detection_dict: Dict[str, Any] = {
                    "class_id": int(class_id) if class_id is not None else -1,
                    "class_name": class_name,
                    "tracker_id": int(tracker_id) if tracker_id is not None else -1,
                    "confidence": float(confidence),
                    "image_path": cropped_path if save_annotations and cropped_path else None,
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "mask": segmentation_info,
                    "keypoints": keypoints_payload,
                }

                frame_results["detections"].append(detection_dict)
                detection_index_counter += 1

            results_list.append(frame_results)

    if processed_frames == 0:
        print("WARNING: No frames were processed for object detection")
        return output_folder, []

    json_payload = {"frames": results_list}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    faces_dir = os.path.join(output_folder, "faces")
    face_clusters_summary: Optional[Dict[str, Any]] = None
    object_clusters_summary: Optional[Dict[str, Any]] = None

    if perform_clustering:
        face_clusters_summary = _cluster_faces(all_faces_data, faces_dir, debug=debug, settings=settings)
        object_clusters_summary = _cluster_objects(class_dir_cache, debug=debug, settings=settings)
    else:
        print("INFO: Skipping clustering step (cluster_objects flag disabled).")

    if face_clusters_summary:
        json_payload["face_clusters"] = face_clusters_summary
    if object_clusters_summary:
        json_payload["object_clusters"] = object_clusters_summary

    output_file = os.path.join(output_folder, "objects.json")
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(json_payload, json_file, indent=4, cls=_NumpyEncoder)

    if not save_annotations:
        _cleanup_annotation_outputs(faces_dir)
        for class_dir in set(class_dir_cache.values()):
            _cleanup_annotation_outputs(class_dir)

    return output_folder, results_list


# ---------------------------------------------------------------------------
# Face detection and clustering
# ---------------------------------------------------------------------------

def _detect_faces(
    frame_identifier: Any,
    output_folder: str,
    save_faces: bool = True,
    frame_img: Optional[np.ndarray] = None,
    debug: bool = False,
    *,
    settings: Optional[_ObjectsSettings] = None,
) -> List[Dict[str, Any]]:
    """Internal face detection implementation."""
    if settings is None:
        settings = _ObjectsSettings()

    frame_path = None
    if isinstance(frame_identifier, int):
        frame_number = frame_identifier
        frame_name = f"{frame_number:08d}.png"
    else:
        frame_path = frame_identifier
        frame_name = os.path.basename(frame_path)
        try:
            frame_number = int(frame_name.split("_")[-1].split(".")[0])
        except ValueError:
            frame_number = 0

    if frame_img is None and frame_path:
        frame_img = cv2.imread(frame_path)
    if frame_img is None:
        return []

    height, width = frame_img.shape[:2]
    scale_up = max(1.0, settings.face_detect_min_side / float(min(height, width)))
    scale_up = min(scale_up, settings.face_detect_max_scale)
    det_img = frame_img if scale_up == 1.0 else cv2.resize(
        frame_img,
        None,
        fx=scale_up,
        fy=scale_up,
        interpolation=cv2.INTER_LINEAR,
    )

    dp_class = _ensure_deepface(debug)
    faces_raw: Optional[List[Dict[str, Any]]] = [] if dp_class is not None else []
    if dp_class is not None:
        try:
            with gray_debug_output(debug):
                faces_raw = dp_class.extract_faces(
                    img_path=det_img,
                    detector_backend=settings.detector_backend,
                    enforce_detection=False,
                    align=True,
                )
        except Exception:
            try:
                with gray_debug_output(debug):
                    faces_raw = dp_class.extract_faces(
                        img_path=det_img,
                        detector_backend="opencv",
                        enforce_detection=False,
                        align=True,
                    )
            except Exception:
                faces_raw = []

    faces: List[Dict[str, Any]] = []
    if not faces_raw:
        return faces

    faces_dir = os.path.join(output_folder, "faces")
    if save_faces:
        os.makedirs(faces_dir, exist_ok=True)

    for idx, face_data in enumerate(faces_raw):
        confidence = face_data.get("confidence", 0.0)
        if confidence < settings.face_conf_threshold:
            continue

        facial_area = face_data.get("facial_area", {})
        x = int(round(facial_area.get("x", 0) / scale_up))
        y = int(round(facial_area.get("y", 0) / scale_up))
        w = int(round(facial_area.get("w", 0) / scale_up))
        h = int(round(facial_area.get("h", 0) / scale_up))

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        cropped_face = frame_img[y1:y2, x1:x2]
        cropped_path = os.path.join(faces_dir, f"{frame_number:08d}_{idx}.png") if save_faces else None
        if save_faces and cropped_path:
            cv2.imwrite(cropped_path, cropped_face)

        embedding = None
        if dp_class is not None:
            try:
                with gray_debug_output(debug):
                    embeddings = dp_class.represent(
                        img_path=face_data.get("face"),
                        model_name=settings.embedding_model_name,
                        enforce_detection=False,
                        detector_backend="skip",
                        align=False,
                    )
                if embeddings and isinstance(embeddings, list):
                    rep = embeddings[0]
                    if isinstance(rep, dict) and "embedding" in rep:
                        embedding = np.array(rep["embedding"], dtype=np.float32)
            except Exception:
                embedding = None

        faces.append(
            {
                "class_id": 100,
                "class_name": "face",
                "confidence": float(confidence),
                "image_path": os.path.abspath(cropped_path) if save_faces and cropped_path else None,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "embedding": embedding,
                "embedding_model": settings.embedding_model_name,
            }
        )

    return faces


def _cluster_faces(
    all_faces_data: Sequence[Dict[str, Any]],
    faces_folder: str,
    *,
    debug: bool = False,
    settings: Optional[_ObjectsSettings] = None,
) -> Optional[Dict[str, Any]]:
    """Internal face clustering implementation."""
    if settings is None:
        settings = _ObjectsSettings()

    detected_faces = len(all_faces_data)
    print(f"INFO: Starting face clustering for {detected_faces} detected faces.")

    if not os.path.isdir(faces_folder):
        print(f"WARNING: Faces folder '{faces_folder}' not found; skipping clustering.")
        return None

    summary = _cluster_class_directory(
        class_name="faces",
        image_folder=faces_folder,
        debug=debug,
        base_eps=0.20,
        min_samples=settings.cluster_min_samples,
        key_eps=settings.keyframe_eps,
        key_min_samples=settings.keyframe_min_samples,
        key_hamming_frac=settings.keyframe_hamming_frac,
        key_require_both=settings.keyframe_require_both,
        cluster_min_attempts=settings.cluster_min_attempts,
        clip_model_name=settings.clip_model_name,
    )

    if summary is None:
        print("WARNING: Face clustering did not produce any results.")
        return None

    summary["detected_faces"] = detected_faces
    return summary


def _cluster_objects(
    class_dir_cache: Dict[str, str],
    *,
    debug: bool = False,
    settings: Optional[_ObjectsSettings] = None,
) -> Dict[str, Dict[str, Any]]:
    """Internal object clustering implementation."""
    if settings is None:
        settings = _ObjectsSettings()

    if not class_dir_cache:
        print("INFO: No object detections available for clustering.")
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    for class_name, image_folder in class_dir_cache.items():
        summary = _cluster_class_directory(
            class_name=class_name,
            image_folder=image_folder,
            debug=debug,
            base_eps=settings.cluster_base_eps,
            min_samples=settings.cluster_min_samples,
            key_eps=settings.keyframe_eps,
            key_min_samples=settings.keyframe_min_samples,
            key_hamming_frac=settings.keyframe_hamming_frac,
            key_require_both=settings.keyframe_require_both,
            cluster_min_attempts=settings.cluster_min_attempts,
            clip_model_name=settings.clip_model_name,
        )
        if summary is None:
            continue
        summary["class_name"] = class_name
        results[class_name] = summary

    if not results:
        print("INFO: Object clustering skipped (no eligible crops).")

    return results
