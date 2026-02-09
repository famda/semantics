from __future__ import annotations

import importlib
import json
import os
import shutil
import warnings
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

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
from .utils.logging import debug_print, gray_debug_output, info_print, update_sub_progress

if TYPE_CHECKING:
    from config import ObjectsConfig

__all__ = ["handle"]

# Environment optimizations
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("TF_DISABLE_XLA", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Constants
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
EXCLUDED_CLUSTER_MARKERS = ("_ann", "_mask", "_background", "_polygon", "_msk")
SMART_SEEK_THRESHOLD = 32
INFERENCE_BATCH_SIZE = 8  # Batch size for YOLO models

from dataclasses import dataclass, field
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_objects_defaults() -> dict:
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
            "face_cluster_base_eps": cfg.face_cluster_base_eps,
            "cluster_dedup_threshold": cfg.cluster_dedup_threshold,
            "cluster_noise_max_distance": cfg.cluster_noise_max_distance,
            "cluster_min_samples": cfg.cluster_min_samples,
            "cluster_min_attempts": cfg.cluster_min_attempts,
            "keyframe_eps": cfg.keyframe_eps,
            "keyframe_min_samples": cfg.keyframe_min_samples,
            "keyframe_hamming_frac": cfg.keyframe_hamming_frac,
            "keyframe_require_both": cfg.keyframe_require_both,
        }
    except Exception:
        return {
            "detection_model": "yolo26x.pt",
            "segmentation_model": "yolo26x-seg.pt",
            "pose_model": "yolo26x-pose.pt",
            "object_conf_threshold": 0.80,
            "iou_match_threshold": 0.5,
            "keypoint_conf_threshold": 0.6,
            "face_conf_threshold": 0.9,
            "embedding_model_name": "Facenet512",
            "face_detect_min_side": 720,
            "face_detect_max_scale": 2.0,
            "detector_backend": "retinaface",
            "clip_model_name": "openai/clip-vit-base-patch32",
            "cluster_base_eps": 0.35,
            "face_cluster_base_eps": 0.20,
            "cluster_dedup_threshold": 0.95,
            "cluster_noise_max_distance": 0.60,
            "cluster_min_samples": 1,
            "cluster_min_attempts": 4,
            "keyframe_eps": 0.12,
            "keyframe_min_samples": 1,
            "keyframe_hamming_frac": 0.30,
            "keyframe_require_both": True,
        }

@dataclass
class _ObjectsSettings:
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
    face_cluster_base_eps: float = field(default_factory=lambda: _get_objects_defaults()["face_cluster_base_eps"])
    cluster_dedup_threshold: float = field(default_factory=lambda: _get_objects_defaults()["cluster_dedup_threshold"])
    cluster_noise_max_distance: float = field(default_factory=lambda: _get_objects_defaults()["cluster_noise_max_distance"])
    cluster_min_samples: int = field(default_factory=lambda: _get_objects_defaults()["cluster_min_samples"])
    cluster_min_attempts: int = field(default_factory=lambda: _get_objects_defaults()["cluster_min_attempts"])
    keyframe_eps: float = field(default_factory=lambda: _get_objects_defaults()["keyframe_eps"])
    keyframe_min_samples: int = field(default_factory=lambda: _get_objects_defaults()["keyframe_min_samples"])
    keyframe_hamming_frac: float = field(default_factory=lambda: _get_objects_defaults()["keyframe_hamming_frac"])
    keyframe_require_both: bool = field(default_factory=lambda: _get_objects_defaults()["keyframe_require_both"])

# =============================================================================
# Async I/O Helpers
# =============================================================================

class ThreadedImageWriter:
    """Non-blocking image writer to keep GPU fed."""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        # Limit queue size to prevent memory explosion if disk is slow
        self.max_queue = max_workers * 10
        
    def write(self, path: str, img: np.ndarray):
        # Prune finished futures
        self.futures = [f for f in self.futures if not f.done()]
        
        # Backpressure: Wait if queue is full
        if len(self.futures) > self.max_queue:
            _, self.futures = wait(self.futures, return_when="FIRST_COMPLETED")
            
        self.futures.append(self.executor.submit(cv2.imwrite, path, img))
        
    def shutdown(self):
        wait(self.futures)
        self.executor.shutdown(wait=True)

class AsyncVideoReader:
    """Reads frames in background thread with smart seeking."""
    def __init__(self, path: str, indices: List[int], queue_size=32):
        self.path = path
        self.indices = sorted(list(set(indices)))
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.total = len(self.indices)
        
    def start(self):
        self.thread.start()
        
    def _worker(self):
        cap = cv2.VideoCapture(self.path)
        current_pos = -1
        
        for idx in self.indices:
            if self.stop_event.is_set():
                break
            
            # Smart Seek
            gap = idx - current_pos - 1
            if gap == 0:
                pass
            elif 0 < gap < SMART_SEEK_THRESHOLD:
                for _ in range(gap): cap.grab()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            
            ret, frame = cap.read()
            current_pos = idx
            
            if not ret:
                break
                
            self.queue.put((idx, frame))
            
        cap.release()
        self.queue.put(None)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item
        
    def stop(self):
        self.stop_event.set()
        # Drain queue
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Exception: pass

# =============================================================================
# Cached Models
# =============================================================================

_TF_MODULE: Optional[Any] = None
_DEEPFACE_CLASS: Optional[Any] = None
_CLIP_MODEL: Optional[Any] = None
_CLIP_PROCESSOR: Optional[Any] = None
_CLIP_DEVICE: Optional[torch.device] = None

_YOLO_DEVICE: Optional[torch.device] = None
_YOLO_OBJECT_MODEL: Optional[YOLO] = None
_YOLO_SEGMENTATION_MODEL: Optional[YOLO] = None
_YOLO_POSE_MODEL: Optional[YOLO] = None
_YOLO_USE_HALF_PRECISION: bool = False

class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def _ensure_tensorflow(debug: bool) -> Optional[Any]:
    global _TF_MODULE
    if _TF_MODULE is not None: return _TF_MODULE
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if debug else "3"
    try:
        with gray_debug_output(debug):
            tf_module = importlib.import_module("tensorflow")
        try: tf_module.config.set_visible_devices([], "GPU")
        except Exception: pass
        _TF_MODULE = tf_module
        return _TF_MODULE
    except Exception: return None

def _ensure_deepface(debug: bool) -> Optional[Any]:
    global _DEEPFACE_CLASS
    if _DEEPFACE_CLASS is not None: return _DEEPFACE_CLASS
    # Monkey-patch Keras 3 validation in both deepface and retinaface
    # (tf-keras is not installed; we use Keras 3 directly via tensorflow)
    for mod_path in (
        "deepface.commons.package_utils",
        "retinaface.commons.package_utils",
    ):
        try:
            _pkg = importlib.import_module(mod_path)
            _pkg.validate_for_keras3 = lambda: None
        except Exception:
            pass
    with gray_debug_output(debug):
        from deepface import DeepFace as _DeepFace
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

def _probe_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0
    finally: cap.release()

def _prepare_class_dir(cache: Dict[str, str], base_dir: str, class_name: str) -> str:
    if class_name not in cache:
        safe_name = class_name.replace(" ", "_")
        class_path = os.path.join(base_dir, safe_name)
        os.makedirs(class_path, exist_ok=True)
        cache[class_name] = class_path
    return cache[class_name]

def _list_valid_images(folder: str) -> List[str]:
    if not os.path.isdir(folder): return []
    valid = []
    for name in os.listdir(folder):
        lower = name.lower()
        if not lower.endswith(VALID_IMAGE_EXTENSIONS): continue
        if lower.startswith("_") or any(marker in lower for marker in EXCLUDED_CLUSTER_MARKERS): continue
        abs_path = os.path.join(folder, name)
        if os.path.isfile(abs_path): valid.append(abs_path)
    return sorted(valid)

def _ensure_clip_resources(debug: bool, clip_model_name: str):
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    if _CLIP_MODEL and _CLIP_PROCESSOR and _CLIP_DEVICE:
        return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from transformers import CLIPModel, CLIPProcessor
        with gray_debug_output(debug):
            model = CLIPModel.from_pretrained(clip_model_name).to(device)
            processor = CLIPProcessor.from_pretrained(clip_model_name)
        model.eval()
    except Exception as exc:
        print(f"ERROR: CLIP load failed: {exc}")
        return None, None, None
    _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE = model, processor, device
    return model, processor, device

def _extract_clip_features(
    image_paths: Sequence[str],
    *,
    debug: bool,
    clip_model_name: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Extract CLIP features using ThreadPool for preprocessing to maximize GPU."""
    model, processor, device = _ensure_clip_resources(debug, clip_model_name)
    if not model: return {}

    feature_map = {}

    # Load images in parallel with ThreadPool
    def _load_img(p):
        try:
            with Image.open(p) as img:
                return p, img.convert("RGB").copy()
        except Exception: return p, None

    # Load and extract in batches
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(image_paths), batch_size):
            batch_files = image_paths[i:i+batch_size]
            results = list(pool.map(_load_img, batch_files))

            valid_images = []
            valid_paths = []
            for path, pil_img in results:
                if pil_img is not None:
                    valid_paths.append(path)
                    valid_images.append(pil_img)

            if not valid_images: continue

            inputs = processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = model.get_image_features(**inputs)

            features_np = features.detach().cpu().numpy().astype(np.float32)
            for idx, p in enumerate(valid_paths):
                feature_map[p] = features_np[idx]

    return feature_map

# Helpers for pHash
@lru_cache(maxsize=4)
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

def _phash64_pil(path: str) -> int:
    """Standard PIL pHash to ensure exact precision/results match with original."""
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

def _cleanup_annotation_outputs(base_dir: str):
    if not os.path.isdir(base_dir): return
    for entry in list(os.listdir(base_dir)):
        if entry == "clusters": continue
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path): shutil.rmtree(path, ignore_errors=True)
        else:
            try: os.remove(path)
            except Exception: pass
    if not os.listdir(base_dir): shutil.rmtree(base_dir, ignore_errors=True)

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
    if not source_paths: return []
    feats = [feature_map.get(p) for p in source_paths]
    valid_idxs = [i for i, f in enumerate(feats) if f is not None]
    if not valid_idxs: return []
    
    feats = [feats[i] for i in valid_idxs]
    source_paths_valid = [source_paths[i] for i in valid_idxs]
    
    feature_matrix = l2_normalize_rows(np.stack(feats, axis=0))
    try:
        labels = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="cosine", n_jobs=-1).fit_predict(feature_matrix)
    except Exception:
        labels = np.zeros(len(feats), dtype=int)
        
    # Parallel pHash using PIL logic to preserve precision
    if len(source_paths_valid) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(source_paths_valid))) as pool:
            phashes = list(pool.map(_phash64_pil, source_paths_valid))
    else:
        phashes = [_phash64_pil(p) for p in source_paths_valid]
        
    selected = []
    unique_labels = sorted(set(labels))
    if -1 in unique_labels: unique_labels.remove(-1)
    
    for lbl in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == lbl]
        chosen = []
        for member in indices:
            if not chosen:
                chosen.append(member); continue
            
            f_mem = feature_matrix[member]
            h_mem = phashes[member]
            
            cos_dists = []
            hamm_dists = []
            
            for prev in chosen:
                f_ex = feature_matrix[prev]
                cos_dists.append(1.0 - float(np.dot(f_mem, f_ex)))
                xor = (h_mem ^ phashes[prev]) & ((1 << 64) - 1)
                hamm_dists.append(bin(xor).count('1') / 64.0)
                
            min_cos = min(cos_dists) if cos_dists else 1.0
            min_hamm = min(hamm_dists) if hamm_dists else 1.0
            
            diff = (min_cos >= eps) and (min_hamm >= hamming_frac) if require_both else (min_cos >= eps) or (min_hamm >= hamming_frac)
            if diff: chosen.append(member)
        selected.extend([source_paths_valid[i] for i in chosen])
        
    if not selected: selected = [source_paths_valid[0]]
    return list(sorted(set(selected)))

def _determine_half_precision(device: torch.device) -> bool:
    if device.type != "cuda": return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 6
    except Exception: return False

def _prepare_yolo_model(model: YOLO, device: torch.device, *, use_half: bool, debug: bool) -> YOLO:
    try: model.to(device)
    except Exception: pass
    try: model.fuse()
    except Exception: pass
    if use_half:
        try: model.model.half()
        except Exception:
            use_half = False
            try: model.model.float()
            except Exception: pass
    return model

def _ensure_yolo_models(debug: bool, require_segmentation: bool, require_pose: bool, detection_model_name: str, segmentation_model_name: str, pose_model_name: str):
    global _YOLO_OBJECT_MODEL, _YOLO_SEGMENTATION_MODEL, _YOLO_POSE_MODEL, _YOLO_DEVICE, _YOLO_USE_HALF_PRECISION
    if _YOLO_DEVICE is None:
        _YOLO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _YOLO_USE_HALF_PRECISION = _determine_half_precision(_YOLO_DEVICE)
    device = _YOLO_DEVICE
    
    def _load_model(name):
        # Strict adherence to /platform requirement
        cwd = os.getcwd()
        try:
            if os.path.isdir("/platform"):
                os.chdir("/platform")
            with gray_debug_output(debug):
                m = YOLO(name)
            return m
        finally:
            os.chdir(cwd)

    if _YOLO_OBJECT_MODEL is None:
        _YOLO_OBJECT_MODEL = _load_model(detection_model_name)
        _YOLO_OBJECT_MODEL = _prepare_yolo_model(_YOLO_OBJECT_MODEL, device, use_half=_YOLO_USE_HALF_PRECISION, debug=debug)
        
    if require_segmentation and _YOLO_SEGMENTATION_MODEL is None:
        _YOLO_SEGMENTATION_MODEL = _load_model(segmentation_model_name)
        _YOLO_SEGMENTATION_MODEL = _prepare_yolo_model(_YOLO_SEGMENTATION_MODEL, device, use_half=_YOLO_USE_HALF_PRECISION, debug=debug)
        
    if require_pose and _YOLO_POSE_MODEL is None:
        _YOLO_POSE_MODEL = _load_model(pose_model_name)
        _YOLO_POSE_MODEL = _prepare_yolo_model(_YOLO_POSE_MODEL, device, use_half=_YOLO_USE_HALF_PRECISION, debug=debug)
        
    return _YOLO_OBJECT_MODEL, (_YOLO_SEGMENTATION_MODEL if require_segmentation else None), (_YOLO_POSE_MODEL if require_pose else None), device

def _cluster_class_directory(
    class_name: str,
    image_folder: str,
    *,
    debug: bool,
    base_eps: float,
    min_samples: int,
    dedup_threshold: float,
    noise_max_distance: float,
    key_eps: float,
    key_min_samples: int,
    key_hamming_frac: float,
    key_require_both: bool,
    cluster_min_attempts: int,
    clip_model_name: str,
    precomputed_features: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, Any]]:
    """Cluster cropped object images using a two-passage approach.

    Passage 1 — Near-duplicate pre-filter:
        Sequential scan keeps only crops whose cosine similarity to the
        previously kept crop is below ``dedup_threshold``.

    Passage 2 — Adaptive DBSCAN on candidates:
        Runs the adaptive eps retry loop on the deduplicated candidate set.

    Passage 3 — Centroid assignment:
        Computes cluster centroids from candidates and assigns *every*
        original crop to the nearest centroid.  Crops whose cosine distance
        exceeds ``noise_max_distance`` are relegated to the noise folder.
    """
    images = _list_valid_images(image_folder)
    if not images:
        return None

    # Use segmentation-masked crops (background removed) for feature
    # extraction when available.  This forces CLIP to focus on the subject
    # rather than the shared background, dramatically improving accuracy
    # for small datasets.  Cluster folders still use the original crops.
    if precomputed_features:
        # Use caller-supplied embeddings (e.g. Facenet512 for faces).
        # Only include images that exist in the precomputed dict.
        feature_map: Dict[str, np.ndarray] = {
            p: precomputed_features[p]
            for p in images
            if p in precomputed_features
        }
        if debug:
            debug_print(
                f"[cluster:{class_name}] Using {len(feature_map)}/{len(images)} "
                f"precomputed embeddings (skipping CLIP extraction)",
                debug=debug,
            )
    else:
        seg_dir = os.path.join(image_folder, "_seg")
        if os.path.isdir(seg_dir):
            feature_source_paths: List[str] = []
            for img_path in images:
                seg_path = os.path.join(seg_dir, os.path.basename(img_path))
                feature_source_paths.append(seg_path if os.path.isfile(seg_path) else img_path)
            raw_feature_map = _extract_clip_features(
                feature_source_paths, debug=debug, clip_model_name=clip_model_name,
            )
            # Remap keys: seg path -> original path so downstream uses originals
            feature_map = {}
            for img_path, src_path in zip(images, feature_source_paths):
                if src_path in raw_feature_map:
                    feature_map[img_path] = raw_feature_map[src_path]
        else:
            feature_map = _extract_clip_features(
                images, debug=debug, clip_model_name=clip_model_name,
            )

    ordered_paths = [p for p in images if p in feature_map]
    if len(ordered_paths) < min_samples:
        return None

    feature_matrix = l2_normalize_rows(
        np.stack([feature_map[p] for p in ordered_paths], axis=0)
    )

    if debug:
        sim_mtx = feature_matrix @ feature_matrix.T
        names_short = [os.path.basename(p)[:15] for p in ordered_paths]
        debug_print(f"[cluster:{class_name}] {len(ordered_paths)} images, pairwise cosine similarity:", debug=debug)
        for i, n in enumerate(names_short):
            row = " ".join(f"{sim_mtx[i, j]:.3f}" for j in range(len(names_short)))
            debug_print(f"  {n:>15s} | {row}", debug=debug)
        off = sim_mtx[np.triu_indices(len(names_short), k=1)]
        debug_print(f"  off-diag: min={off.min():.4f} max={off.max():.4f} mean={off.mean():.4f} std={off.std():.4f}", debug=debug)
        used_seg = os.path.isdir(os.path.join(image_folder, "_seg"))
        debug_print(f"  feature_source={'_seg/ (masked)' if used_seg else 'original crops'}", debug=debug)

    # ------------------------------------------------------------------
    # Passage 1 — Near-duplicate pre-filter (linear scan)
    # ------------------------------------------------------------------
    cand_indices: List[int] = []
    last_feat: Optional[np.ndarray] = None
    for j, feat in enumerate(feature_matrix):
        if last_feat is None:
            cand_indices.append(j)
            last_feat = feat
        else:
            sim = float(np.dot(last_feat, feat))
            if sim < float(dedup_threshold):
                cand_indices.append(j)
                last_feat = feat

    cand_matrix = feature_matrix[cand_indices]

    # ------------------------------------------------------------------
    # Passage 2 — Adaptive DBSCAN on candidate subset
    # ------------------------------------------------------------------
    effective_eps = float(base_eps)
    cand_labels: Optional[np.ndarray] = None

    if len(cand_indices) < 2:
        # Too few candidates — everything goes into a single cluster
        cand_labels = np.zeros(len(cand_indices), dtype=int)
    else:
        attempt_eps = float(base_eps)
        for attempt in range(cluster_min_attempts):
            eps_candidate = estimate_dbscan_eps(cand_matrix, attempt_eps, min_samples)
            if not np.isfinite(eps_candidate) or eps_candidate <= 0:
                eps_candidate = attempt_eps
            eps_candidate = max(0.05, min(float(eps_candidate), 0.95))

            try:
                trial_labels = DBSCAN(
                    eps=eps_candidate,
                    min_samples=int(min_samples),
                    metric="cosine",
                    n_jobs=-1,
                ).fit_predict(cand_matrix)
            except Exception:
                trial_labels = np.full(cand_matrix.shape[0], -1, dtype=int)

            u_lbl = sorted(set(int(l) for l in trial_labels if l >= 0))
            mapping = {o: n for n, o in enumerate(u_lbl)}
            trial_labels = np.array([mapping.get(int(l), -1) for l in trial_labels])

            if debug:
                noise_cnt = int(np.sum(trial_labels == -1))
                debug_print(
                    f"[cluster:{class_name}] attempt {attempt}: "
                    f"eps={eps_candidate:.4f} -> {len(u_lbl)} cluster(s), "
                    f"{noise_cnt} noise",
                    debug=debug,
                )

            if len(u_lbl) > 1 or attempt == cluster_min_attempts - 1:
                cand_labels = trial_labels
                effective_eps = eps_candidate
                break
            attempt_eps *= 0.85

    if cand_labels is None:
        return None

    # ------------------------------------------------------------------
    # Passage 3 — Centroid computation & full assignment
    # ------------------------------------------------------------------
    unique_cand_labels = sorted(set(int(l) for l in cand_labels if l >= 0))
    if not unique_cand_labels:
        # All candidates are noise — fall back to a single cluster
        unique_cand_labels = [0]
        cand_labels = np.zeros(len(cand_indices), dtype=int)

    num_clusters = len(unique_cand_labels)
    centroids = np.zeros((num_clusters, cand_matrix.shape[1]), dtype=np.float32)
    for new_id, cid in enumerate(unique_cand_labels):
        members = cand_matrix[cand_labels == cid]
        centroids[new_id] = members.mean(axis=0)
    # Re-normalize centroids
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8

    # Assign every crop to nearest centroid via dot-product similarity
    similarities = np.dot(feature_matrix, centroids.T)  # (N, K)
    labels = np.argmax(similarities, axis=1)
    max_sims = similarities[np.arange(len(labels)), labels]

    # Mark as noise if too far from any centroid
    noise_mask = max_sims < (1.0 - float(noise_max_distance))
    labels[noise_mask] = -1

    # Reindex labels contiguously (0, 1, 2, …)
    final_unique = sorted(set(int(l) for l in labels if l >= 0))
    final_mapping = {old: new for new, old in enumerate(final_unique)}
    labels = np.array([final_mapping.get(int(l), -1) for l in labels])

    # ------------------------------------------------------------------
    # Build output folders and summary
    # ------------------------------------------------------------------
    clusters_root = os.path.join(image_folder, "clusters")
    if os.path.isdir(clusters_root):
        shutil.rmtree(clusters_root)
    os.makedirs(clusters_root, exist_ok=True)

    cluster_details: List[Dict[str, Any]] = []
    noise_paths: List[str] = []
    unique_labels = sorted(set(int(l) for l in labels if l >= 0))

    for c_idx in unique_labels:
        c_dir = os.path.join(clusters_root, f"cluster_{c_idx:03d}")
        members = [p for p, l in zip(ordered_paths, labels) if l == c_idx]
        copied = [_copy_with_unique_name(src, c_dir) for src in members]

        keyframes = _select_keyframes(
            members, feature_map,
            eps=key_eps, min_samples=key_min_samples,
            hamming_frac=key_hamming_frac, require_both=key_require_both,
            debug=debug,
        )
        kf_paths: List[str] = []
        if keyframes:
            kf_dir = os.path.join(c_dir, "keyframes")
            kf_paths = [_copy_with_unique_name(src, kf_dir) for src in keyframes]

        cluster_details.append({
            "cluster_id": int(c_idx),
            "cluster_folder": os.path.abspath(c_dir),
            "image_paths": copied,
            "image_count": len(copied),
            "keyframes": kf_paths,
            "keyframe_count": len(kf_paths),
            "effective_eps": effective_eps,
        })

    n_dir = os.path.join(clusters_root, "noise")
    if any(l < 0 for l in labels):
        for src, lbl in zip(ordered_paths, labels):
            if lbl == -1:
                noise_paths.append(_copy_with_unique_name(src, n_dir))

    summary = {
        "class_name": class_name,
        "image_folder": os.path.abspath(image_folder),
        "clusters_root": os.path.abspath(clusters_root),
        "cluster_count": len(cluster_details),
        "noise_count": len(noise_paths),
        "total_images": len(ordered_paths),
        "candidates_after_dedup": len(cand_indices),
        "clusters": cluster_details,
        "noise": {
            "folder": os.path.abspath(n_dir) if noise_paths else None,
            "image_paths": noise_paths,
            "count": len(noise_paths),
        },
    }

    with open(os.path.join(clusters_root, "clusters.json"), "w") as f:
        json.dump(summary, f, indent=4)
    return summary

def _match_segmentation(bbox, seg_detections, *, iou_match_threshold):
    if not seg_detections or not hasattr(seg_detections, 'xyxy') or len(seg_detections) == 0: return None
    seg_boxes = np.asarray(seg_detections.xyxy, dtype=np.float32)
    if seg_boxes.size == 0: return None
    ious = sv.box_iou_batch(np.asarray([bbox], dtype=np.float32), seg_boxes)
    if ious.size == 0: return None
    best_idx = int(np.argmax(ious[0]))
    if float(ious[0, best_idx]) < iou_match_threshold: return None
    if seg_detections.mask is None or len(seg_detections.mask) <= best_idx: return None
    return best_idx, seg_detections.mask[best_idx]

def _save_segmentation_artifacts(writer: ThreadedImageWriter, frame_img, seg_detections, seg_index, mask, class_dir, frame_number, file_stem, mask_annotator, polygon_annotator):
    mask_folder = os.path.join(class_dir, "masks")
    os.makedirs(mask_folder, exist_ok=True)
    mask_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}.png")
    binary_mask = (mask * 255).astype(np.uint8)
    writer.write(mask_path, binary_mask)
    
    detection = None
    try:
        detection = sv.Detections(
            xyxy=np.asarray([seg_detections.xyxy[seg_index]], dtype=np.float32),
            mask=np.asarray([mask.astype(bool)], dtype=bool),
            class_id=np.zeros(1, dtype=np.int32)
        )
    except Exception: pass
    
    contour_path = None
    if detection:
        try:
            contour_img = polygon_annotator.annotate(scene=frame_img.copy(), detections=detection)
            contour_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}_polygon.png")
            writer.write(contour_path, contour_img)
        except Exception: pass
        
    bg_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}_background.png")
    f_bgra = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
    f_bgra[:, :, 3] = binary_mask
    writer.write(bg_path, f_bgra)
    
    overlay_path = None
    if detection:
        try:
            ov = mask_annotator.annotate(scene=frame_img.copy(), detections=detection)
            overlay_path = os.path.join(mask_folder, f"{frame_number:08d}_{file_stem}_mask.png")
            writer.write(overlay_path, ov)
        except Exception: pass
        
    return {
        "image_path": mask_path,
        "contour_image_path": contour_path,
        "background_image_path": bg_path,
        "mask_overlay_path": overlay_path,
        "polygon_overlay_path": contour_path
    }

def _match_keypoints(bbox, keypoint_detections, keypoints, class_name, *, keypoint_conf_threshold, iou_match_threshold):
    if not keypoint_detections or not keypoints: return []
    kp_xy = getattr(keypoints, "xy", None)
    kp_conf = getattr(keypoints, "confidence", getattr(keypoints, "conf", None))
    if kp_xy is None or kp_conf is None: return []
    pose_boxes = np.asarray(keypoint_detections.xyxy, dtype=np.float32)
    if pose_boxes.size == 0: return []
    ious = sv.box_iou_batch(np.asarray([bbox], dtype=np.float32), pose_boxes)
    if ious.size == 0: return []
    best_idx = int(np.argmax(ious[0]))
    if float(ious[0, best_idx]) < iou_match_threshold: return []
    
    def _np(v): return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    kp_xy = _np(kp_xy)
    kp_conf = _np(kp_conf)
    if best_idx >= kp_xy.shape[0]: return []
    names = VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.get(class_name, [])
    if not names: return []
    kp_xy = kp_xy[best_idx]
    kp_conf = kp_conf[best_idx]
    results = []
    for idx, name in enumerate(names):
        if idx >= kp_xy.shape[0]: break
        if kp_conf[idx] <= keypoint_conf_threshold: continue
        results.append({
            "class_id": idx, "class_name": name,
            "point": {"x": float(kp_xy[idx][0]), "y": float(kp_xy[idx][1])},
            "confidence": float(kp_conf[idx])
        })
    return results

def handle(input_file: str, output_folder: str, config: "ObjectsConfig | None" = None, *, object_classes: Optional[List[str]] = None, frame_indices: Optional[List[int]] = None, perform_clustering: bool = False, save_annotations: bool = False, debug: bool = False):
    info_print("Detecting objects present in the frames")
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
            face_cluster_base_eps=config.face_cluster_base_eps,
            cluster_dedup_threshold=config.cluster_dedup_threshold,
            cluster_noise_max_distance=config.cluster_noise_max_distance,
            cluster_min_samples=config.cluster_min_samples,
            cluster_min_attempts=config.cluster_min_attempts,
            keyframe_eps=config.keyframe_eps,
            keyframe_min_samples=config.keyframe_min_samples,
            keyframe_hamming_frac=config.keyframe_hamming_frac,
            keyframe_require_both=config.keyframe_require_both,
        )
    else:
        settings = _ObjectsSettings()

    return _detect(input_file, output_folder, object_classes, frame_indices=frame_indices, debug=debug, perform_clustering=perform_clustering, save_annotations=save_annotations, settings=settings)

def _detect(video_file, output_folder, classes_to_detect, frame_indices=None, debug=False, *, perform_clustering=True, save_annotations=False, settings=None):
    if settings is None: settings = _ObjectsSettings()
    _ensure_tensorflow(debug)
    output_folder = os.path.join(output_folder, "objects")
    save_annotations = bool(save_annotations)
    # Always save crop images — the module is only invoked when -eo is
    # explicitly requested, so face/object crops should always be persisted.
    should_save_images = True
    
    if classes_to_detect:
        classes_to_detect = {VIDEO_OBJECT_DETECTION_CATEGORY_MAP[c] for c in classes_to_detect if c in VIDEO_OBJECT_DETECTION_CATEGORY_MAP}
    
    kp_cls = set(VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.keys())
    kp_ids = {VIDEO_OBJECT_DETECTION_CATEGORY_MAP[n] for n in kp_cls if n in VIDEO_OBJECT_DETECTION_CATEGORY_MAP}
    pose_req = classes_to_detect is None or bool(kp_ids.intersection(set(classes_to_detect or [])))
    
    obj_model, seg_model, pose_model, device = _ensure_yolo_models(debug, True, pose_req, settings.detection_model, settings.segmentation_model, settings.pose_model)
    
    if not frame_indices: return output_folder, []
    s_idx = []
    for i in frame_indices:
        try: s_idx.append(int(float(i)))
        except Exception: pass
    selected_indices = sorted(list(set(i for i in s_idx if i >= 0)))
    if not selected_indices: return output_folder, []
    
    os.makedirs(output_folder, exist_ok=True)
    fps = _probe_video_fps(video_file)
    tracker = sv.ByteTrack()
    box_ann = sv.BoxAnnotator()
    lbl_ann = sv.LabelAnnotator()
    mask_ann = sv.MaskAnnotator()
    poly_ann = sv.PolygonAnnotator()
    class_dir_cache = {}
    
    writer = ThreadedImageWriter(max_workers=4)
    reader = AsyncVideoReader(video_file, selected_indices)
    reader.start()
    
    total_frames = len(selected_indices)
    pbar_ctx = tqdm(total=total_frames, desc="Objects", unit="frame", colour="#888888") if debug else nullcontext()
    results_list = []
    all_faces = []
    
    # BATCHING LOGIC
    batch_frames = []
    batch_nums = []
    processed = 0
    
    with pbar_ctx as pbar:
        for f_num, f_img in reader:
            batch_frames.append(f_img)
            batch_nums.append(f_num)
            
            if len(batch_frames) >= INFERENCE_BATCH_SIZE:
                _process_batch(
                    batch_frames, batch_nums, video_file, fps, 
                    obj_model, seg_model, pose_model, tracker,
                    settings, classes_to_detect, kp_cls,
                    output_folder, should_save_images, save_annotations,
                    perform_clustering,
                    class_dir_cache, writer, 
                    box_ann, lbl_ann, mask_ann, poly_ann,
                    results_list, all_faces, debug
                )
                processed += len(batch_frames)
                if pbar: pbar.update(len(batch_frames))
                update_sub_progress(processed, total_frames, "frames")
                batch_frames = []
                batch_nums = []
                
        # Remainder
        if batch_frames:
            _process_batch(
                batch_frames, batch_nums, video_file, fps, 
                obj_model, seg_model, pose_model, tracker,
                settings, classes_to_detect, kp_cls,
                output_folder, should_save_images, save_annotations,
                perform_clustering,
                class_dir_cache, writer, 
                box_ann, lbl_ann, mask_ann, poly_ann,
                results_list, all_faces, debug
            )
            processed += len(batch_frames)
            if pbar: pbar.update(len(batch_frames))
            update_sub_progress(processed, total_frames, "frames")
            
    reader.stop()
    writer.shutdown()
    
    json_pl = {"frames": results_list}
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    f_summ = None
    o_summ = None
    if perform_clustering:
        f_dir = os.path.join(output_folder, "faces")
        f_summ = _cluster_faces(all_faces, f_dir, debug=debug, settings=settings)
        o_summ = _cluster_objects(class_dir_cache, debug=debug, settings=settings)
        
    if f_summ: json_pl["face_clusters"] = f_summ
    if o_summ: json_pl["object_clusters"] = o_summ
    
    with open(os.path.join(output_folder, "objects.json"), "w") as f:
        json.dump(json_pl, f, indent=4, cls=_NumpyEncoder)
    if not save_annotations:
        # Only clean up annotated overlay images (_ann/*_ann.png etc.),
        # never delete face crops — they are the primary extraction output.
        for d in set(class_dir_cache.values()): _cleanup_annotation_outputs(d)
        
    return output_folder, results_list

def _process_batch(frames, f_nums, v_file, fps, o_model, s_model, p_model, tracker, settings, cls_detect, kp_cls, out_dir, save_imgs, save_ann, perform_clustering, dir_cache, writer, box_ann, lbl_ann, mask_ann, poly_ann, res_list, all_faces, debug):
    """Internal helper to handle a batch of frames."""
    # 1. Faces (Sequential per frame because DeepFace is heavy/complex)
    # Note: We do this first to keep it simple, though parallelizing DeepFace is hard.
    # The optimization here is we are not blocking I/O for saving face crops.
    for i, f_img in enumerate(frames):
        faces = _detect_faces(f_nums[i], out_dir, save_imgs, f_img, debug, settings, writer)
        all_faces.extend(faces)
        
        h, w = f_img.shape[:2]
        r = {
            "frame_number": f_nums[i],
            "frame_path": f"{os.path.abspath(v_file)}#frame_{f_nums[i]:08d}",
            "resolution": {"width": w, "height": h},
            "detections": [_strip_face_for_results(f) for f in faces]
        }
        if fps > 0: r["pts_time"] = float(f_nums[i]) / fps
        res_list.append(r)

    # 2. YOLO Inference (Batch)
    with torch.inference_mode():
        o_preds = o_model(frames, conf=settings.object_conf_threshold, verbose=False)
        s_preds = s_model(frames, conf=settings.object_conf_threshold, verbose=False) if s_model else [None]*len(frames)
        # Pose needs class checks, usually runs if person/animal detected.
        # Running it on the batch is usually faster than checking per frame unless sparse.
        p_preds = p_model(frames, conf=settings.object_conf_threshold, verbose=False) if p_model else [None]*len(frames)

    # 3. Process Results Frame-by-Frame (Tracking must be sequential)
    for i, f_img in enumerate(frames):
        r_entry = res_list[-(len(frames) - i)] # Get the correct entry we created above
        
        o_res = o_preds[i]
        if not o_res or not o_res.boxes: continue
        
        names = o_res.names
        dets = sv.Detections.from_ultralytics(o_res)
        
        # Track
        tracked = tracker.update_with_detections(dets)
        
        s_res = s_preds[i]
        seg_dets = sv.Detections.from_ultralytics(s_res) if s_res else None
        
        p_res = p_preds[i]
        kp_dets = sv.Detections.from_ultralytics(p_res) if p_res else None
        kps = p_res.keypoints if p_res else None
        
        cnt = 0
        h, w = f_img.shape[:2]
        
        for k in range(len(tracked)):
            cid = tracked.class_id[k]
            tid = tracked.tracker_id[k]
            conf = tracked.confidence[k]
            
            if conf < settings.object_conf_threshold: continue
            if cls_detect and cid not in cls_detect: continue
            
            bbox = tracked.xyxy[k].astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            if x2<=x1 or y2<=y1: continue
            
            c_name = names.get(cid, "N/A")
            stem = f"{f_nums[i]:08d}_{cnt}"
            
            crop_path = None
            c_dir = None
            seg_match = None
            if save_imgs:
                c_dir = _prepare_class_dir(dir_cache, out_dir, c_name)
                crop_path = os.path.join(c_dir, f"{stem}.png")
                crop_img = f_img[y1:y2, x1:x2]
                writer.write(crop_path, crop_img)

                # Save background-removed crop for clustering features.
                # Uses gray fill (≈ ImageNet mean) so CLIP treats masked
                # pixels as neutral, and tight-crops to the foreground
                # bounding-box so the subject fills the frame.
                if perform_clustering:
                    if seg_dets is not None:
                        seg_match = _match_segmentation(
                            [x1, y1, x2, y2], seg_dets,
                            iou_match_threshold=settings.iou_match_threshold,
                        )
                    seg_crop_dir = os.path.join(c_dir, "_seg")
                    os.makedirs(seg_crop_dir, exist_ok=True)
                    seg_crop_path = os.path.join(seg_crop_dir, f"{stem}.png")
                    if seg_match:
                        _, full_mask = seg_match
                        crop_mask = full_mask[y1:y2, x1:x2]
                        masked = crop_img.copy()
                        # Gray fill ≈ ImageNet mean (BGR order) to minimise
                        # CLIP background bias vs black (which produces
                        # strong negative features after normalisation).
                        masked[crop_mask == 0] = [104, 116, 122]
                        # Tight-crop to foreground bbox so CLIP focuses
                        # on the subject rather than surrounding fill.
                        fg_rows = np.any(crop_mask, axis=1)
                        fg_cols = np.any(crop_mask, axis=0)
                        if fg_rows.any() and fg_cols.any():
                            r0, r1 = np.where(fg_rows)[0][[0, -1]]
                            c0, c1 = np.where(fg_cols)[0][[0, -1]]
                            masked = masked[r0:r1 + 1, c0:c1 + 1]
                        writer.write(seg_crop_path, masked)
                    else:
                        # Fallback: save original crop so clustering still works
                        writer.write(seg_crop_path, crop_img)

            if save_ann and c_name.lower() == "person" and c_dir:
                try:
                    _d = sv.Detections(
                        xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                        confidence=np.array([conf], dtype=np.float32),
                        class_id=np.array([cid], dtype=int)
                    )
                    ann = box_ann.annotate(f_img.copy(), _d)
                    ann = lbl_ann.annotate(ann, _d, labels=[c_name])
                    writer.write(os.path.join(c_dir, f"{stem}_ann.png"), ann)
                except Exception: pass
                
            seg_info = {}
            if save_ann and c_dir:
                ann_seg_match = seg_match if (perform_clustering and seg_match) else _match_segmentation([x1, y1, x2, y2], seg_dets, iou_match_threshold=settings.iou_match_threshold)
                if ann_seg_match:
                    si, mask = ann_seg_match
                    payload = _save_segmentation_artifacts(
                        writer, f_img, seg_dets, si, mask, c_dir, f_nums[i], 
                        str(tid) if tid is not None else str(cnt), mask_ann, poly_ann
                    )
                    seg_info = {
                        "image_path": payload["image_path"],
                        "background_image_path": payload["background_image_path"],
                        "mask_overlay_path": payload["mask_overlay_path"],
                        "polygon_image_path": payload["polygon_overlay_path"]
                    }
            
            kp_info = _match_keypoints([x1, y1, x2, y2], kp_dets, kps, c_name, keypoint_conf_threshold=settings.keypoint_conf_threshold, iou_match_threshold=settings.iou_match_threshold)
            
            r_entry["detections"].append({
                "class_id": int(cid) if cid is not None else -1,
                "class_name": c_name,
                "tracker_id": int(tid) if tid is not None else -1,
                "confidence": float(conf),
                "image_path": crop_path,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "mask": seg_info,
                "keypoints": kp_info
            })
            cnt += 1

def _detect_faces(frame_number, output_folder, save_faces, frame_img, debug, settings, writer=None):
    if frame_img is None: return []
    h, w = frame_img.shape[:2]
    
    # Resize logic
    s = max(1.0, settings.face_detect_min_side / float(min(h, w)))
    s = min(s, settings.face_detect_max_scale)
    det_img = frame_img if s == 1.0 else cv2.resize(frame_img, None, fx=s, fy=s)
    
    dp = _ensure_deepface(debug)
    raw = []
    if dp:
        try:
            with gray_debug_output(debug):
                raw = dp.extract_faces(det_img, detector_backend=settings.detector_backend, enforce_detection=False, align=True)
        except Exception:
            try:
                raw = dp.extract_faces(det_img, detector_backend="opencv", enforce_detection=False, align=True)
            except Exception: pass
            
    faces = []
    if not raw: return faces
    
    f_dir = os.path.join(output_folder, "faces")
    if save_faces: os.makedirs(f_dir, exist_ok=True)
    
    for i, d in enumerate(raw):
        conf = d.get("confidence", 0.0)
        if conf < settings.face_conf_threshold: continue
        
        area = d.get("facial_area", {})
        x = int(round(area.get("x", 0) / s))
        y = int(round(area.get("y", 0) / s))
        fw = int(round(area.get("w", 0) / s))
        fh = int(round(area.get("h", 0) / s))
        
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x+fw), min(h, y+fh)
        if x2<=x1 or y2<=y1: continue
        
        crop_path = None
        if save_faces:
            crop = frame_img[y1:y2, x1:x2]
            crop_path = os.path.join(f_dir, f"{frame_number:08d}_{i}.png")
            if writer: writer.write(crop_path, crop)
            else: cv2.imwrite(crop_path, crop)
            
        emb = None
        if dp:
            try:
                with gray_debug_output(debug):
                    e = dp.represent(d.get("face"), model_name=settings.embedding_model_name, enforce_detection=False, detector_backend="skip", align=False)
                if e and isinstance(e, list): emb = np.array(e[0]["embedding"], dtype=np.float32)
            except Exception: pass
            
        faces.append({
            "class_id": 100,
            "class_name": "face",
            "confidence": float(conf),
            "image_path": os.path.abspath(crop_path) if crop_path else None,
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "embedding": emb,
            "embedding_model": settings.embedding_model_name
        })
        
    return faces

def _cluster_faces(all_faces, faces_dir, debug, settings):
    info_print(f"Clustering {len(all_faces)} faces")
    if not os.path.isdir(faces_dir): return None

    # Build precomputed feature map from Facenet512 embeddings already
    # extracted during detection.  This avoids re-reading face crops from
    # disk and running CLIP inference, while also using a model that is
    # specialised for face identity (much better than generic CLIP).
    precomputed: Dict[str, np.ndarray] = {}
    for face in all_faces:
        emb = face.get("embedding")
        path = face.get("image_path")
        if emb is not None and path:
            precomputed[path] = emb

    if debug:
        debug_print(
            f"[cluster:faces] {len(precomputed)}/{len(all_faces)} faces have "
            f"precomputed {all_faces[0].get('embedding_model', 'unknown')} embeddings",
            debug=debug,
        )

    summ = _cluster_class_directory(
        "faces", faces_dir, debug=debug,
        base_eps=settings.face_cluster_base_eps,
        min_samples=settings.cluster_min_samples,
        dedup_threshold=settings.cluster_dedup_threshold,
        noise_max_distance=settings.cluster_noise_max_distance,
        key_eps=settings.keyframe_eps, key_min_samples=settings.keyframe_min_samples,
        key_hamming_frac=settings.keyframe_hamming_frac, key_require_both=settings.keyframe_require_both,
        cluster_min_attempts=settings.cluster_min_attempts, clip_model_name=settings.clip_model_name,
        precomputed_features=precomputed if precomputed else None,
    )
    if summ: summ["detected_faces"] = len(all_faces)
    return summ

def _cluster_objects(cache, debug, settings):
    if not cache: return {}
    res = {}
    for c_name, f_dir in cache.items():
        summ = _cluster_class_directory(
            c_name, f_dir, debug=debug,
            base_eps=settings.cluster_base_eps,
            min_samples=settings.cluster_min_samples,
            dedup_threshold=settings.cluster_dedup_threshold,
            noise_max_distance=settings.cluster_noise_max_distance,
            key_eps=settings.keyframe_eps, key_min_samples=settings.keyframe_min_samples,
            key_hamming_frac=settings.keyframe_hamming_frac, key_require_both=settings.keyframe_require_both,
            cluster_min_attempts=settings.cluster_min_attempts, clip_model_name=settings.clip_model_name,
        )
        if summ:
            summ["class_name"] = c_name
            res[c_name] = summ
    return res