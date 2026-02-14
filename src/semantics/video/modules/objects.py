"""Object detection, segmentation, pose estimation, tracking, and clustering.

Architecture
------------
The **detection model** serves as the primary detector for maximum recall.
It determines which objects exist in each frame.

The **segmentation model** runs as a secondary model at a lower confidence
threshold and its per-instance masks are matched to detection-model boxes
via IoU.  If the seg model misses an object, the detection still appears
(simply without a mask).

The **pose model** runs as a tertiary model and its keypoints are matched
to detection boxes by IoU.

Face detection is delegated to the dedicated :mod:`faces` module.

Annotation outputs (when ``--save-annotations`` is used) are organized into
dedicated subdirectories per class: ``masks/``, ``polygons/``,
``backgrounds/``, ``overlays/``, ``annotations/``, and ``keypoints/``.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from functools import lru_cache
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
from .utils.logging import (
    debug_print,
    gray_debug_output,
    info_print,
    update_sub_progress,
)

if TYPE_CHECKING:
    from config import FacesConfig, ObjectsConfig

__all__ = ["handle"]

# ---------------------------------------------------------------------------
# Environment optimizations (module-level constants — never mutated)
# ---------------------------------------------------------------------------
os.environ.setdefault("YOLO_VERBOSE", "False")
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
EXCLUDED_CLUSTER_MARKERS = ("_ann", "_mask", "_background", "_polygon", "_msk")
_SMART_SEEK_THRESHOLD = 32
_INFERENCE_BATCH_SIZE = 8
_ANNOTATION_SUBDIRS = ("masks", "polygons", "backgrounds", "overlays", "annotations", "keypoints", "_seg")


# ============================================================================
# Async I/O Helpers
# ============================================================================

class _ThreadedImageWriter:
    """Non-blocking image writer with backpressure."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list = []
        self._max_queue = max_workers * 10

    def write(self, path: str, img: np.ndarray) -> None:
        self._futures = [f for f in self._futures if not f.done()]
        if len(self._futures) > self._max_queue:
            _, not_done = wait(self._futures, return_when="FIRST_COMPLETED")
            self._futures = list(not_done)
        self._futures.append(self._executor.submit(cv2.imwrite, path, img))

    def shutdown(self) -> None:
        wait(self._futures)
        self._executor.shutdown(wait=True)


class _AsyncVideoReader:
    """Background-thread video reader with smart seeking."""

    def __init__(self, path: str, indices: List[int], queue_size: int = 32):
        self._path = path
        self._indices = sorted(set(indices))
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self.total = len(self._indices)

    def start(self) -> None:
        self._thread.start()

    def _worker(self) -> None:
        cap = cv2.VideoCapture(self._path)
        pos = -1
        for idx in self._indices:
            if self._stop.is_set():
                break
            gap = idx - pos - 1
            if gap == 0:
                pass
            elif 0 < gap < _SMART_SEEK_THRESHOLD:
                for _ in range(gap):
                    cap.grab()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            pos = idx
            if ret:
                self._queue.put((idx, frame))
        cap.release()
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item

    def stop(self) -> None:
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                pass


# ============================================================================
# JSON Encoder
# ============================================================================

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ============================================================================
# YOLO Model Loading (dict-cached, no global variable mutation)
# ============================================================================

_yolo_cache: Dict[str, Any] = {}


@lru_cache(maxsize=1)
def _get_yolo_device() -> tuple:
    """Return ``(device, use_half)`` for YOLO models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = False
    if device.type == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability(device)
            use_half = major >= 6
        except Exception:
            pass
    return device, use_half


def _load_yolo_model(
    name: str, device: torch.device, use_half: bool, debug: bool,
) -> YOLO:
    """Load and prepare a single YOLO model."""
    cwd = os.getcwd()
    try:
        if os.path.isdir("/platform"):
            os.chdir("/platform")
        with gray_debug_output(debug):
            model = YOLO(name)
    finally:
        os.chdir(cwd)
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
            model.model.half()
        except Exception:
            try:
                model.model.float()
            except Exception:
                pass
    return model


def _ensure_yolo_model(name: str, debug: bool) -> YOLO:
    """Get a YOLO model, loading and caching if not already loaded."""
    if name in _yolo_cache:
        return _yolo_cache[name]
    device, use_half = _get_yolo_device()
    model = _load_yolo_model(name, device, use_half, debug)
    _yolo_cache[name] = model
    return model


# ============================================================================
# CLIP Embedding Helpers (for clustering)
# ============================================================================

_clip_cache: Dict[str, Any] = {}


def _ensure_clip(clip_model_name: str, debug: bool):
    """Load CLIP model + processor (cached after first call)."""
    if "model" in _clip_cache:
        return _clip_cache["model"], _clip_cache["processor"], _clip_cache["device"]
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
    _clip_cache.update(model=model, processor=processor, device=device)
    return model, processor, device


def _extract_clip_features(
    image_paths: Sequence[str],
    *,
    debug: bool,
    clip_model_name: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Extract CLIP image features for clustering."""
    model, processor, device = _ensure_clip(clip_model_name, debug)
    if model is None:
        return {}

    feature_map: Dict[str, np.ndarray] = {}

    def _load_img(p: str):
        try:
            with Image.open(p) as img:
                return p, img.convert("RGB").copy()
        except Exception:
            return p, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0, len(image_paths), batch_size):
            batch_files = image_paths[i : i + batch_size]
            results = list(pool.map(_load_img, batch_files))
            valid_images, valid_paths = [], []
            for path, pil_img in results:
                if pil_img is not None:
                    valid_paths.append(path)
                    valid_images.append(pil_img)
            if not valid_images:
                continue
            inputs = processor(images=valid_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features_np = features.detach().cpu().numpy().astype(np.float32)
            for idx, p in enumerate(valid_paths):
                feature_map[p] = features_np[idx]

    return feature_map


# ============================================================================
# Perceptual Hashing
# ============================================================================

@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    n_ = np.arange(n)[None, :]
    mat = np.cos(np.pi * (n_ + 0.5) * k / n)
    mat[0, :] /= np.sqrt(n)
    mat[1:, :] *= np.sqrt(2 / n)
    return mat.astype(np.float32)


def _dct_2d(a: np.ndarray) -> np.ndarray:
    n, m = a.shape
    return _dct_matrix(n) @ a @ _dct_matrix(m).T


def _phash64_pil(path: str) -> int:
    """Compute 64-bit perceptual hash via DCT."""
    try:
        with Image.open(path) as img:
            img_l = img.convert("L")
            size = 32  # 8 * highfreq_factor(4)
            resized = img_l.resize((size, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(resized, dtype=np.float32)
            dct = _dct_2d(pixels)
            low = dct[:8, :8].flatten()
            threshold = (
                float(np.median(low[1:]))
                if low.size > 1
                else (low[0] if low.size == 1 else 0.0)
            )
            bits = (low > threshold).astype(np.uint8)
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            return int(value)
    except Exception:
        return 0


# ============================================================================
# File / Directory Helpers
# ============================================================================

def _probe_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else 0.0
    finally:
        cap.release()


def _prepare_class_dir(
    cache: Dict[str, str], base_dir: str, class_name: str,
) -> str:
    if class_name not in cache:
        safe = class_name.replace(" ", "_")
        path = os.path.join(base_dir, safe)
        os.makedirs(path, exist_ok=True)
        cache[class_name] = path
    return cache[class_name]


def _list_valid_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    valid = []
    for name in os.listdir(folder):
        lower = name.lower()
        if not lower.endswith(VALID_IMAGE_EXTENSIONS):
            continue
        if lower.startswith("_") or any(
            m in lower for m in EXCLUDED_CLUSTER_MARKERS
        ):
            continue
        abs_path = os.path.join(folder, name)
        if os.path.isfile(abs_path):
            valid.append(abs_path)
    return sorted(valid)


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


def _cleanup_annotation_dirs(class_dir: str) -> None:
    """Remove annotation-only subdirectories, preserving crops and clusters."""
    if not os.path.isdir(class_dir):
        return
    for subdir in _ANNOTATION_SUBDIRS:
        path = os.path.join(class_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# Keyframe Selection
# ============================================================================

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
    feats = [feature_map.get(p) for p in source_paths]
    valid_idxs = [i for i, f in enumerate(feats) if f is not None]
    if not valid_idxs:
        return []

    feats = [feats[i] for i in valid_idxs]
    paths = [source_paths[i] for i in valid_idxs]
    matrix = l2_normalize_rows(np.stack(feats, axis=0))

    try:
        labels = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
            metric="cosine",
            n_jobs=-1,
        ).fit_predict(matrix)
    except Exception:
        labels = np.zeros(len(feats), dtype=int)

    if len(paths) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
            phashes = list(pool.map(_phash64_pil, paths))
    else:
        phashes = [_phash64_pil(p) for p in paths]

    selected: list[str] = []
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    for lbl in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == lbl]
        chosen: list[int] = []
        for member in indices:
            if not chosen:
                chosen.append(member)
                continue
            f_mem = matrix[member]
            h_mem = phashes[member]
            cos_dists = [
                1.0 - float(np.dot(f_mem, matrix[prev])) for prev in chosen
            ]
            hamm_dists = [
                bin((h_mem ^ phashes[prev]) & ((1 << 64) - 1)).count("1") / 64.0
                for prev in chosen
            ]
            min_cos = min(cos_dists) if cos_dists else 1.0
            min_hamm = min(hamm_dists) if hamm_dists else 1.0
            diff = (
                (min_cos >= eps) and (min_hamm >= hamming_frac)
                if require_both
                else (min_cos >= eps) or (min_hamm >= hamming_frac)
            )
            if diff:
                chosen.append(member)
        selected.extend(paths[i] for i in chosen)

    if not selected:
        selected = [paths[0]]
    return sorted(set(selected))


# ============================================================================
# Clustering
# ============================================================================

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
    """Cluster cropped images with adaptive DBSCAN.

    Three passages:
    1. Near-duplicate pre-filter (sequential cosine scan).
    2. Adaptive DBSCAN on candidate subset.
    3. Centroid computation + full assignment.
    """
    images = _list_valid_images(image_folder)
    if not images:
        return None

    # Feature extraction
    if precomputed_features:
        feature_map: Dict[str, np.ndarray] = {
            p: precomputed_features[p] for p in images if p in precomputed_features
        }
        if debug:
            debug_print(
                f"[cluster:{class_name}] Using {len(feature_map)}/{len(images)} "
                f"precomputed embeddings",
                debug=debug,
            )
    else:
        seg_dir = os.path.join(image_folder, "_seg")
        if os.path.isdir(seg_dir):
            source_paths: List[str] = []
            for img_path in images:
                seg_path = os.path.join(seg_dir, os.path.basename(img_path))
                source_paths.append(
                    seg_path if os.path.isfile(seg_path) else img_path,
                )
            raw_map = _extract_clip_features(
                source_paths, debug=debug, clip_model_name=clip_model_name,
            )
            feature_map = {}
            for img_path, src_path in zip(images, source_paths):
                if src_path in raw_map:
                    feature_map[img_path] = raw_map[src_path]
        else:
            feature_map = _extract_clip_features(
                images, debug=debug, clip_model_name=clip_model_name,
            )

    ordered = [p for p in images if p in feature_map]
    if len(ordered) < min_samples:
        return None

    matrix = l2_normalize_rows(
        np.stack([feature_map[p] for p in ordered], axis=0),
    )

    if debug:
        sim = matrix @ matrix.T
        shorts = [os.path.basename(p)[:15] for p in ordered]
        debug_print(
            f"[cluster:{class_name}] {len(ordered)} images, pairwise cosine similarity:",
            debug=debug,
        )
        for i, n in enumerate(shorts):
            row = " ".join(f"{sim[i, j]:.3f}" for j in range(len(shorts)))
            debug_print(f"  {n:>15s} | {row}", debug=debug)
        off = sim[np.triu_indices(len(shorts), k=1)]
        debug_print(
            f"  off-diag: min={off.min():.4f} max={off.max():.4f} "
            f"mean={off.mean():.4f} std={off.std():.4f}",
            debug=debug,
        )
        used_seg = os.path.isdir(os.path.join(image_folder, "_seg"))
        debug_print(
            f"  feature_source={'_seg/ (masked)' if used_seg else 'original crops'}",
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Passage 1 — Near-duplicate pre-filter (linear scan)
    # ------------------------------------------------------------------
    cand_idx: List[int] = []
    last_feat: Optional[np.ndarray] = None
    for j, feat in enumerate(matrix):
        if last_feat is None or float(np.dot(last_feat, feat)) < dedup_threshold:
            cand_idx.append(j)
            last_feat = feat
    cand = matrix[cand_idx]

    # ------------------------------------------------------------------
    # Passage 2 — Adaptive DBSCAN on candidate subset
    # ------------------------------------------------------------------
    effective_eps = float(base_eps)
    cand_labels: Optional[np.ndarray] = None

    if len(cand_idx) < 2:
        cand_labels = np.zeros(len(cand_idx), dtype=int)
    else:
        attempt_eps = float(base_eps)
        for attempt in range(cluster_min_attempts):
            eps_c = estimate_dbscan_eps(cand, attempt_eps, min_samples)
            if not np.isfinite(eps_c) or eps_c <= 0:
                eps_c = attempt_eps
            eps_c = max(0.05, min(float(eps_c), 0.95))
            try:
                trial = DBSCAN(
                    eps=eps_c,
                    min_samples=int(min_samples),
                    metric="cosine",
                    n_jobs=-1,
                ).fit_predict(cand)
            except Exception:
                trial = np.full(cand.shape[0], -1, dtype=int)
            u = sorted(set(int(l) for l in trial if l >= 0))
            mapping = {o: n for n, o in enumerate(u)}
            trial = np.array([mapping.get(int(l), -1) for l in trial])
            if debug:
                debug_print(
                    f"[cluster:{class_name}] attempt {attempt}: "
                    f"eps={eps_c:.4f} -> {len(u)} cluster(s), "
                    f"{int(np.sum(trial == -1))} noise",
                    debug=debug,
                )
            if len(u) > 1 or attempt == cluster_min_attempts - 1:
                cand_labels = trial
                effective_eps = eps_c
                break
            attempt_eps *= 0.85

    if cand_labels is None:
        return None

    # ------------------------------------------------------------------
    # Passage 3 — Centroid computation & full assignment
    # ------------------------------------------------------------------
    unique_cand = sorted(set(int(l) for l in cand_labels if l >= 0))
    if not unique_cand:
        unique_cand = [0]
        cand_labels = np.zeros(len(cand_idx), dtype=int)

    centroids = np.zeros((len(unique_cand), cand.shape[1]), dtype=np.float32)
    for nid, cid in enumerate(unique_cand):
        centroids[nid] = cand[cand_labels == cid].mean(axis=0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8

    sims = np.dot(matrix, centroids.T)
    labels = np.argmax(sims, axis=1)
    max_sims = sims[np.arange(len(labels)), labels]
    labels[max_sims < (1.0 - noise_max_distance)] = -1

    final_unique = sorted(set(int(l) for l in labels if l >= 0))
    fmap = {old: new for new, old in enumerate(final_unique)}
    labels = np.array([fmap.get(int(l), -1) for l in labels])

    # Build output
    clusters_root = os.path.join(image_folder, "clusters")
    if os.path.isdir(clusters_root):
        shutil.rmtree(clusters_root)
    os.makedirs(clusters_root, exist_ok=True)

    details: List[Dict[str, Any]] = []
    noise_paths: List[str] = []

    for c_idx in sorted(set(int(l) for l in labels if l >= 0)):
        c_dir = os.path.join(clusters_root, f"cluster_{c_idx:03d}")
        members = [p for p, l in zip(ordered, labels) if l == c_idx]
        copied = [_copy_with_unique_name(src, c_dir) for src in members]
        kf = _select_keyframes(
            members,
            feature_map,
            eps=key_eps,
            min_samples=key_min_samples,
            hamming_frac=key_hamming_frac,
            require_both=key_require_both,
            debug=debug,
        )
        kf_paths = (
            [_copy_with_unique_name(src, os.path.join(c_dir, "keyframes")) for src in kf]
            if kf
            else []
        )
        details.append({
            "cluster_id": int(c_idx),
            "cluster_folder": os.path.abspath(c_dir),
            "image_paths": copied,
            "image_count": len(copied),
            "keyframes": kf_paths,
            "keyframe_count": len(kf_paths),
            "effective_eps": effective_eps,
        })

    n_dir = os.path.join(clusters_root, "noise")
    for src, lbl in zip(ordered, labels):
        if lbl == -1:
            noise_paths.append(_copy_with_unique_name(src, n_dir))

    summary = {
        "class_name": class_name,
        "image_folder": os.path.abspath(image_folder),
        "clusters_root": os.path.abspath(clusters_root),
        "cluster_count": len(details),
        "noise_count": len(noise_paths),
        "total_images": len(ordered),
        "candidates_after_dedup": len(cand_idx),
        "clusters": details,
        "noise": {
            "folder": os.path.abspath(n_dir) if noise_paths else None,
            "image_paths": noise_paths,
            "count": len(noise_paths),
        },
    }
    with open(os.path.join(clusters_root, "clusters.json"), "w") as f:
        json.dump(summary, f, indent=4)
    return summary


# ============================================================================
# Pose Keypoint Matching (IoU-based)
# ============================================================================

def _match_keypoints(
    bbox: List[int],
    pose_detections: Optional[sv.Detections],
    keypoints: Any,
    class_name: str,
    *,
    keypoint_conf_threshold: float,
    iou_match_threshold: float,
) -> List[Dict[str, Any]]:
    """Match pose keypoints to a detection box via IoU."""
    if pose_detections is None or keypoints is None:
        return []
    kp_xy = getattr(keypoints, "xy", None)
    kp_conf = getattr(keypoints, "confidence", getattr(keypoints, "conf", None))
    if kp_xy is None or kp_conf is None:
        return []
    pose_boxes = np.asarray(pose_detections.xyxy, dtype=np.float32)
    if pose_boxes.size == 0:
        return []
    ious = sv.box_iou_batch(
        np.asarray([bbox], dtype=np.float32), pose_boxes,
    )
    if ious.size == 0:
        return []
    best_idx = int(np.argmax(ious[0]))
    if float(ious[0, best_idx]) < iou_match_threshold:
        return []

    def _np(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    kp_xy_np = _np(kp_xy)
    kp_conf_np = _np(kp_conf)
    if best_idx >= kp_xy_np.shape[0]:
        return []
    names = VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.get(class_name, [])
    if not names:
        return []
    kp_xy_best = kp_xy_np[best_idx]
    kp_conf_best = kp_conf_np[best_idx]
    results: List[Dict[str, Any]] = []
    for idx, name in enumerate(names):
        if idx >= kp_xy_best.shape[0]:
            break
        if kp_conf_best[idx] <= keypoint_conf_threshold:
            continue
        results.append({
            "class_id": idx,
            "class_name": name,
            "point": {
                "x": float(kp_xy_best[idx][0]),
                "y": float(kp_xy_best[idx][1]),
            },
            "confidence": float(kp_conf_best[idx]),
        })
    return results


# ============================================================================
# Keypoint Annotation Drawing
# ============================================================================

# COCO 17-keypoint skeleton: pairs of keypoint indices to connect with lines.
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12),                           # torso
    (11, 12),                                   # hips
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# BGR colours per limb group for visual clarity.
_SKELETON_COLORS = [
    (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),        # head — blue
    (0, 255, 0),                                                  # shoulders — green
    (0, 255, 255), (0, 255, 255), (0, 200, 200), (0, 200, 200), # arms — yellow/cyan
    (0, 128, 255), (0, 128, 255),                                 # torso — orange
    (255, 0, 255),                                                # hips — magenta
    (255, 128, 0), (255, 128, 0), (200, 128, 0), (200, 128, 0), # legs — teal
]


def _save_keypoints_annotation(
    writer: _ThreadedImageWriter,
    frame_img: np.ndarray,
    keypoints_info: List[Dict[str, Any]],
    class_name: str,
    class_dir: str,
    stem: str,
) -> Optional[str]:
    """Draw keypoint circles and skeleton lines, save to ``keypoints/`` subdir.

    Returns the saved file path, or *None* if nothing was drawn.
    """
    if not keypoints_info:
        return None

    kp_names = VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.get(class_name, [])
    if not kp_names:
        return None

    # Build index: name → (x, y, conf)
    name_to_pt: Dict[str, tuple] = {}
    for kp in keypoints_info:
        pt = kp.get("point", {})
        name_to_pt[kp["class_name"]] = (
            int(round(pt.get("x", 0))),
            int(round(pt.get("y", 0))),
            float(kp.get("confidence", 0)),
        )

    canvas = frame_img.copy()

    # Draw skeleton lines
    for idx, (a, b) in enumerate(_COCO_SKELETON):
        if a >= len(kp_names) or b >= len(kp_names):
            continue
        name_a, name_b = kp_names[a], kp_names[b]
        if name_a not in name_to_pt or name_b not in name_to_pt:
            continue
        xa, ya, _ = name_to_pt[name_a]
        xb, yb, _ = name_to_pt[name_b]
        color = _SKELETON_COLORS[idx] if idx < len(_SKELETON_COLORS) else (200, 200, 200)
        cv2.line(canvas, (xa, ya), (xb, yb), color, thickness=2, lineType=cv2.LINE_AA)

    # Draw keypoint circles
    for kp in keypoints_info:
        pt = kp.get("point", {})
        x, y = int(round(pt.get("x", 0))), int(round(pt.get("y", 0)))
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    kp_dir = os.path.join(class_dir, "keypoints")
    os.makedirs(kp_dir, exist_ok=True)
    kp_path = os.path.join(kp_dir, f"{stem}.png")
    writer.write(kp_path, canvas)
    return kp_path


# ============================================================================
# Segmentation Artifact Saving
# ============================================================================

def _save_segmentation_artifacts(
    writer: _ThreadedImageWriter,
    frame_img: np.ndarray,
    bbox: List[int],
    mask: np.ndarray,
    class_dir: str,
    stem: str,
    mask_annotator: sv.MaskAnnotator,
    polygon_annotator: sv.PolygonAnnotator,
) -> Dict[str, Optional[str]]:
    """Save mask, polygon, background, and overlay to organized subdirectories.

    Args:
        stem: File name stem in the format ``{frame:08d}_{tid}``.
    """
    x1, y1, x2, y2 = bbox

    # 1. Binary mask → masks/
    masks_dir = os.path.join(class_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    mask_path = os.path.join(masks_dir, f"{stem}.png")
    binary = (mask * 255).astype(np.uint8)
    writer.write(mask_path, binary)

    # Build a single-detection object for annotation
    det = None
    try:
        det = sv.Detections(
            xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
            mask=np.array([mask.astype(bool)], dtype=bool),
            class_id=np.zeros(1, dtype=np.int32),
        )
    except Exception:
        pass

    # 2. Polygon contour → polygons/
    polygon_path = None
    if det is not None:
        try:
            polygons_dir = os.path.join(class_dir, "polygons")
            os.makedirs(polygons_dir, exist_ok=True)
            polygon_path = os.path.join(
                polygons_dir, f"{stem}.png",
            )
            contour_img = polygon_annotator.annotate(
                scene=frame_img.copy(), detections=det,
            )
            writer.write(polygon_path, contour_img)
        except Exception:
            polygon_path = None

    # 3. Background-removed BGRA → backgrounds/
    bgs_dir = os.path.join(class_dir, "backgrounds")
    os.makedirs(bgs_dir, exist_ok=True)
    bg_path = os.path.join(bgs_dir, f"{stem}.png")
    bgra = cv2.cvtColor(frame_img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = binary
    writer.write(bg_path, bgra)

    # 4. Mask overlay → overlays/
    overlay_path = None
    if det is not None:
        try:
            overlays_dir = os.path.join(class_dir, "overlays")
            os.makedirs(overlays_dir, exist_ok=True)
            overlay_path = os.path.join(
                overlays_dir, f"{stem}.png",
            )
            ov = mask_annotator.annotate(scene=frame_img.copy(), detections=det)
            writer.write(overlay_path, ov)
        except Exception:
            overlay_path = None

    return {
        "image_path": mask_path,
        "background_image_path": bg_path,
        "mask_overlay_path": overlay_path,
        "polygon_image_path": polygon_path,
    }


# ============================================================================
# Face & Object Clustering
# ============================================================================

def _cluster_faces(
    all_faces: List[Dict[str, Any]],
    faces_dir: str,
    *,
    debug: bool,
    face_cluster_base_eps: float,
    face_cluster_min_samples: int,
    face_cluster_dedup_threshold: float,
    face_cluster_noise_max_distance: float,
    face_cluster_min_attempts: int,
    keyframe_eps: float,
    keyframe_min_samples: int,
    keyframe_hamming_frac: float,
    keyframe_require_both: bool,
    clip_model_name: str,
) -> Optional[Dict[str, Any]]:
    """Cluster faces using precomputed Facenet512 embeddings."""
    info_print(f"Clustering {len(all_faces)} faces")
    if not os.path.isdir(faces_dir):
        return None

    precomputed: Dict[str, np.ndarray] = {}
    for face in all_faces:
        emb = face.get("embedding")
        path = face.get("image_path")
        if emb is not None and path:
            precomputed[path] = emb

    if debug:
        model_name = (
            all_faces[0].get("embedding_model", "unknown") if all_faces else "unknown"
        )
        debug_print(
            f"[cluster:faces] {len(precomputed)}/{len(all_faces)} faces have "
            f"precomputed {model_name} embeddings",
            debug=debug,
        )

    summ = _cluster_class_directory(
        "faces",
        faces_dir,
        debug=debug,
        base_eps=face_cluster_base_eps,
        min_samples=face_cluster_min_samples,
        dedup_threshold=face_cluster_dedup_threshold,
        noise_max_distance=face_cluster_noise_max_distance,
        key_eps=keyframe_eps,
        key_min_samples=keyframe_min_samples,
        key_hamming_frac=keyframe_hamming_frac,
        key_require_both=keyframe_require_both,
        cluster_min_attempts=face_cluster_min_attempts,
        clip_model_name=clip_model_name,
        precomputed_features=precomputed if precomputed else None,
    )
    if summ:
        summ["detected_faces"] = len(all_faces)
    return summ


def _cluster_objects(
    class_dirs: Dict[str, str],
    *,
    debug: bool,
    cluster_base_eps: float,
    cluster_min_samples: int,
    cluster_dedup_threshold: float,
    cluster_noise_max_distance: float,
    keyframe_eps: float,
    keyframe_min_samples: int,
    keyframe_hamming_frac: float,
    keyframe_require_both: bool,
    cluster_min_attempts: int,
    clip_model_name: str,
    person_face_embeddings: Optional[Dict[str, np.ndarray]] = None,
    face_cluster_base_eps: float = 0.20,
    face_cluster_min_samples: int = 1,
    face_cluster_dedup_threshold: float = 0.99,
    face_cluster_noise_max_distance: float = 0.40,
    face_cluster_min_attempts: int = 6,
) -> Dict[str, Any]:
    """Cluster object crops per class."""
    if not class_dirs:
        return {}
    result: Dict[str, Any] = {}
    for c_name, c_dir in class_dirs.items():
        # For 'person' class, use face embeddings when available for
        # instance-level clustering instead of semantic-only CLIP features.
        precomputed = None
        use_face_params = False
        if c_name == "person" and person_face_embeddings:
            precomputed = person_face_embeddings
            use_face_params = True
            if debug:
                debug_print(
                    f"[cluster:person] Using {len(precomputed)} face embeddings "
                    f"for instance-level clustering",
                    debug=debug,
                )
        summ = _cluster_class_directory(
            c_name,
            c_dir,
            debug=debug,
            base_eps=face_cluster_base_eps if use_face_params else cluster_base_eps,
            min_samples=face_cluster_min_samples if use_face_params else cluster_min_samples,
            dedup_threshold=face_cluster_dedup_threshold if use_face_params else cluster_dedup_threshold,
            noise_max_distance=face_cluster_noise_max_distance if use_face_params else cluster_noise_max_distance,
            key_eps=keyframe_eps,
            key_min_samples=keyframe_min_samples,
            key_hamming_frac=keyframe_hamming_frac,
            key_require_both=keyframe_require_both,
            cluster_min_attempts=face_cluster_min_attempts if use_face_params else cluster_min_attempts,
            clip_model_name=clip_model_name,
            precomputed_features=precomputed,
        )
        if summ:
            summ["class_name"] = c_name
            result[c_name] = summ
    return result


# ============================================================================
# Batch Processing
# ============================================================================

def _match_seg_mask(
    bbox: List[int],
    seg_detections: Optional[sv.Detections],
    *,
    iou_match_threshold: float,
) -> Optional[np.ndarray]:
    """Find the best matching segmentation mask for a detection box."""
    if seg_detections is None or seg_detections.mask is None:
        return None
    seg_boxes = np.asarray(seg_detections.xyxy, dtype=np.float32)
    if seg_boxes.size == 0:
        return None
    ious = sv.box_iou_batch(
        np.asarray([bbox], dtype=np.float32), seg_boxes,
    )
    if ious.size == 0:
        return None
    best_idx = int(np.argmax(ious[0]))
    if float(ious[0, best_idx]) < iou_match_threshold:
        return None
    if best_idx < len(seg_detections.mask):
        return seg_detections.mask[best_idx]
    return None


def _process_batch(
    frames: List[np.ndarray],
    frame_numbers: List[int],
    video_path: str,
    fps: float,
    det_model: YOLO,
    seg_model: Optional[YOLO],
    pose_model: Optional[YOLO],
    tracker: sv.ByteTrack,
    *,
    face_map: Dict[int, List[Dict[str, Any]]],
    object_conf_threshold: float,
    segmentation_conf_threshold: float,
    iou_match_threshold: float,
    keypoint_conf_threshold: float,
    classes_to_detect: Optional[set],
    kp_class_names: set,
    output_dir: str,
    save_annotations: bool,
    perform_clustering: bool,
    class_dir_cache: Dict[str, str],
    writer: _ThreadedImageWriter,
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    mask_annotator: sv.MaskAnnotator,
    polygon_annotator: sv.PolygonAnnotator,
    results_list: List[Dict[str, Any]],
    fallback_tid_counter: List[int],
    person_face_embeddings: Dict[str, np.ndarray],
    debug: bool,
) -> None:
    """Process a batch: detection model primary, seg for masks, pose for keypoints."""
    from . import faces as faces_module

    # 1. Create frame result entries with face detections pre-populated
    for i, f_img in enumerate(frames):
        fn = frame_numbers[i]
        frame_faces = face_map.get(fn, [])
        h, w = f_img.shape[:2]
        entry: Dict[str, Any] = {
            "frame_number": fn,
            "frame_path": f"{os.path.abspath(video_path)}#frame_{fn:08d}",
            "resolution": {"width": w, "height": h},
            "detections": [faces_module.strip_for_results(f) for f in frame_faces],
        }
        if fps > 0:
            entry["pts_time"] = float(fn) / fps
        results_list.append(entry)

    # 2. YOLO Inference: Detection (primary) + Seg (masks) + Pose (keypoints)
    with torch.inference_mode():
        det_preds = det_model(
            frames, conf=object_conf_threshold, verbose=False,
        )
        seg_preds = (
            seg_model(
                frames, conf=segmentation_conf_threshold, verbose=False,
            )
            if seg_model is not None
            else [None] * len(frames)
        )
        pose_preds = (
            pose_model(frames, conf=object_conf_threshold, verbose=False)
            if pose_model is not None
            else [None] * len(frames)
        )

    # 3. Per-frame tracking and post-processing
    for i, f_img in enumerate(frames):
        fn = frame_numbers[i]
        r_entry = results_list[-(len(frames) - i)]

        det_res = det_preds[i]
        if det_res is None or det_res.boxes is None or len(det_res.boxes) == 0:
            continue

        names = det_res.names

        # Detection model provides the authoritative set of detections
        dets = sv.Detections.from_ultralytics(det_res)

        # Attempt tracking — used only for ID assignment, never for filtering.
        # ByteTrack may drop detections (especially with sparse keyframes),
        # so we iterate over raw *dets* and look up tracker IDs.
        tracked = tracker.update_with_detections(dets)

        # Build a mapping from raw detection index → tracker ID using IoU
        _tid_map: Dict[int, int] = {}
        if tracked.tracker_id is not None and len(tracked) > 0:
            _iou_matrix = sv.box_iou_batch(
                np.asarray(dets.xyxy, dtype=np.float32),
                np.asarray(tracked.xyxy, dtype=np.float32),
            )
            for raw_k in range(len(dets)):
                if _iou_matrix[raw_k].size > 0:
                    best_t = int(np.argmax(_iou_matrix[raw_k]))
                    if float(_iou_matrix[raw_k, best_t]) > 0.5:
                        _tid_map[raw_k] = int(tracked.tracker_id[best_t])

        # Segmentation detections (for mask lookup via IoU)
        seg_res = seg_preds[i]
        seg_dets = (
            sv.Detections.from_ultralytics(seg_res)
            if seg_res is not None
            and seg_res.boxes is not None
            and len(seg_res.boxes) > 0
            else None
        )

        # Pose data for this frame
        p_res = pose_preds[i]
        pose_dets = (
            sv.Detections.from_ultralytics(p_res) if p_res is not None else None
        )
        kps = p_res.keypoints if p_res is not None else None

        h, w = f_img.shape[:2]
        cnt = 0

        # Iterate over ALL raw detections (never filtered by tracker)
        for k in range(len(dets)):
            cid = dets.class_id[k]
            conf = dets.confidence[k]
            tid = _tid_map.get(k)
            if tid is None:
                tid = fallback_tid_counter[0]
                fallback_tid_counter[0] += 1

            if conf < object_conf_threshold:
                continue
            if classes_to_detect is not None and cid not in classes_to_detect:
                continue

            bbox_raw = dets.xyxy[k].astype(int)
            x1, y1 = max(0, bbox_raw[0]), max(0, bbox_raw[1])
            x2, y2 = min(w, bbox_raw[2]), min(h, bbox_raw[3])
            if x2 <= x1 or y2 <= y1:
                continue

            c_name = names.get(cid, "N/A")
            stem = f"{fn:08d}_{tid}"

            # Match segmentation mask from seg model via IoU
            det_mask = _match_seg_mask(
                [x1, y1, x2, y2],
                seg_dets,
                iou_match_threshold=iou_match_threshold,
            )

            # Save crop
            c_dir = _prepare_class_dir(class_dir_cache, output_dir, c_name)
            crop_img = f_img[y1:y2, x1:x2]
            crop_path = os.path.join(c_dir, f"{stem}.png")
            writer.write(crop_path, crop_img)

            # Associate face embedding with person crop for clustering
            if c_name == "person" and perform_clustering:
                frame_faces = face_map.get(fn, [])
                best_face_iou = 0.0
                best_face_emb = None
                person_box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                for face in frame_faces:
                    fb = face.get("bounding_box")
                    emb = face.get("embedding")
                    if fb is None or emb is None:
                        continue
                    face_box = np.array(
                        [[fb["x1"], fb["y1"], fb["x2"], fb["y2"]]],
                        dtype=np.float32,
                    )
                    iou_val = float(sv.box_iou_batch(person_box, face_box)[0, 0])
                    if iou_val > best_face_iou:
                        best_face_iou = iou_val
                        best_face_emb = emb
                # The face must be inside the person bbox (even partial overlap)
                if best_face_emb is not None and best_face_iou > 0.01:
                    person_face_embeddings[os.path.abspath(crop_path)] = best_face_emb

            # Save seg-masked crop for clustering
            if perform_clustering:
                seg_crop_dir = os.path.join(c_dir, "_seg")
                os.makedirs(seg_crop_dir, exist_ok=True)
                seg_crop_path = os.path.join(seg_crop_dir, f"{stem}.png")
                if det_mask is not None:
                    crop_mask = det_mask[y1:y2, x1:x2]
                    masked = crop_img.copy()
                    # Gray fill ≈ ImageNet mean (BGR) to minimise CLIP
                    # background bias vs black.
                    masked[crop_mask == 0] = [104, 116, 122]
                    # Tight-crop to foreground bbox
                    fg_rows = np.any(crop_mask, axis=1)
                    fg_cols = np.any(crop_mask, axis=0)
                    if fg_rows.any() and fg_cols.any():
                        r0, r1 = np.where(fg_rows)[0][[0, -1]]
                        c0, c1 = np.where(fg_cols)[0][[0, -1]]
                        masked = masked[r0 : r1 + 1, c0 : c1 + 1]
                    writer.write(seg_crop_path, masked)
                else:
                    # Fallback: save original crop so clustering still works
                    writer.write(seg_crop_path, crop_img)

            # Save annotations (box + label) → annotations/
            if save_annotations:
                try:
                    ann_dir = os.path.join(c_dir, "annotations")
                    os.makedirs(ann_dir, exist_ok=True)
                    _d = sv.Detections(
                        xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                        confidence=np.array([conf], dtype=np.float32),
                        class_id=np.array([cid], dtype=int),
                    )
                    ann = box_annotator.annotate(f_img.copy(), _d)
                    ann = label_annotator.annotate(ann, _d, labels=[c_name])
                    writer.write(os.path.join(ann_dir, f"{stem}.png"), ann)
                except Exception:
                    pass

            # Save segmentation artifacts (masks, polygons, backgrounds, overlays)
            seg_info: Dict[str, Optional[str]] = {}
            if save_annotations and det_mask is not None:
                seg_info = _save_segmentation_artifacts(
                    writer,
                    f_img,
                    [x1, y1, x2, y2],
                    det_mask,
                    c_dir,
                    stem,
                    mask_annotator,
                    polygon_annotator,
                )

            # Match pose keypoints via IoU
            kp_info = _match_keypoints(
                [x1, y1, x2, y2],
                pose_dets,
                kps,
                c_name,
                keypoint_conf_threshold=keypoint_conf_threshold,
                iou_match_threshold=iou_match_threshold,
            )

            # Save keypoints annotation → keypoints/
            kp_ann_path: Optional[str] = None
            if save_annotations and kp_info:
                kp_ann_path = _save_keypoints_annotation(
                    writer, f_img, kp_info, c_name, c_dir, stem,
                )

            r_entry["detections"].append({
                "class_id": int(cid) if cid is not None else -1,
                "class_name": c_name,
                "tracker_id": int(tid) if tid is not None else -1,
                "confidence": float(conf),
                "image_path": crop_path,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "mask": seg_info,
                "keypoints": kp_info,
                "keypoints_annotation_path": kp_ann_path,
            })
            cnt += 1


# ============================================================================
# Public Entry Point
# ============================================================================

def handle(
    input_file: str,
    output_folder: str,
    config: "ObjectsConfig | None" = None,
    *,
    faces_config: "FacesConfig | None" = None,
    object_classes: Optional[List[str]] = None,
    frame_indices: Optional[List[int]] = None,
    perform_clustering: bool = False,
    save_annotations: bool = False,
    debug: bool = False,
):
    """Detect, track, and optionally cluster objects and faces in video frames.

    The detection model serves as the primary detector for maximum recall.
    The seg model provides per-instance masks (matched by IoU), and pose
    provides keypoints.  Face detection is delegated to :mod:`faces`.

    Args:
        input_file: Path to the video file.
        output_folder: Root output directory.
        config: ObjectsConfig instance or None for defaults.
        faces_config: FacesConfig instance or None for defaults.
        object_classes: List of COCO class names to detect (None = all).
        frame_indices: List of frame indices to process.
        perform_clustering: Cluster detected objects and faces.
        save_annotations: Save masks, polygons, overlays, annotated images.
        debug: Enable verbose output.

    Returns:
        Tuple of (output_folder, results_list).
    """
    info_print("Detecting objects present in the frames")

    # Extract config values with inline defaults
    det_model_name = config.detection_model if config else "yolo26x.pt"
    seg_model_name = config.segmentation_model if config else "yolo26x-seg.pt"
    pose_model_name = config.pose_model if config else "yolo26x-pose.pt"
    object_conf_threshold = config.object_conf_threshold if config else 0.80
    segmentation_conf_threshold = config.segmentation_conf_threshold if config else 0.25
    iou_match_threshold = config.iou_match_threshold if config else 0.5
    keypoint_conf_threshold = config.keypoint_conf_threshold if config else 0.6
    clip_model_name = config.clip_model_name if config else "openai/clip-vit-base-patch32"
    cluster_base_eps = config.cluster_base_eps if config else 0.35
    cluster_dedup_threshold = config.cluster_dedup_threshold if config else 0.95
    cluster_noise_max_distance = config.cluster_noise_max_distance if config else 0.60
    cluster_min_samples = config.cluster_min_samples if config else 1
    cluster_min_attempts = config.cluster_min_attempts if config else 4
    keyframe_eps = config.keyframe_eps if config else 0.12
    keyframe_min_samples = config.keyframe_min_samples if config else 1
    keyframe_hamming_frac = config.keyframe_hamming_frac if config else 0.30
    keyframe_require_both = config.keyframe_require_both if config else True

    # Face clustering config
    face_cluster_base_eps = (
        faces_config.face_cluster_base_eps if faces_config else 0.40
    )
    face_cluster_dedup_threshold = (
        faces_config.face_cluster_dedup_threshold if faces_config else 0.99
    )
    face_cluster_noise_max_distance = (
        faces_config.face_cluster_noise_max_distance if faces_config else 0.55
    )
    face_cluster_min_samples = (
        faces_config.face_cluster_min_samples if faces_config else 1
    )
    face_cluster_min_attempts = (
        faces_config.face_cluster_min_attempts if faces_config else 6
    )

    output_dir = os.path.join(output_folder, "objects")
    os.makedirs(output_dir, exist_ok=True)

    # Determine which classes to detect
    cls_filter: Optional[set] = None
    if object_classes:
        cls_filter = {
            VIDEO_OBJECT_DETECTION_CATEGORY_MAP[c]
            for c in object_classes
            if c in VIDEO_OBJECT_DETECTION_CATEGORY_MAP
        }

    # Determine if pose model is needed
    kp_cls = set(VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING.keys())
    kp_ids = {
        VIDEO_OBJECT_DETECTION_CATEGORY_MAP[n]
        for n in kp_cls
        if n in VIDEO_OBJECT_DETECTION_CATEGORY_MAP
    }
    pose_required = cls_filter is None or bool(kp_ids & (cls_filter or set()))

    # Load YOLO models: detection (primary) + segmentation (masks) + pose (keypoints)
    det_model = _ensure_yolo_model(det_model_name, debug)
    seg_model = _ensure_yolo_model(seg_model_name, debug)
    pose_model = _ensure_yolo_model(pose_model_name, debug) if pose_required else None

    # Validate and sort frame indices
    if not frame_indices:
        return output_dir, []
    selected_indices: List[int] = []
    for idx in frame_indices:
        try:
            val = int(float(idx))
            if val >= 0:
                selected_indices.append(val)
        except Exception:
            pass
    selected_indices = sorted(set(selected_indices))
    if not selected_indices:
        return output_dir, []

    fps = _probe_video_fps(input_file)
    tracker = sv.ByteTrack()
    box_ann = sv.BoxAnnotator()
    lbl_ann = sv.LabelAnnotator()
    mask_ann = sv.MaskAnnotator()
    poly_ann = sv.PolygonAnnotator()
    class_dir_cache: Dict[str, str] = {}

    writer = _ThreadedImageWriter(max_workers=4)
    reader = _AsyncVideoReader(input_file, selected_indices)
    reader.start()

    total_frames = len(selected_indices)
    pbar_ctx = (
        tqdm(total=total_frames, desc="Objects", unit="frame", colour="#888888")
        if debug
        else nullcontext()
    )
    results_list: List[Dict[str, Any]] = []
    all_faces: List[Dict[str, Any]] = []
    fallback_tid_counter: List[int] = [100_000]  # high start to avoid ByteTrack collisions
    person_face_embeddings: Dict[str, np.ndarray] = {}  # person crop → face embedding

    # Import faces module once
    from . import faces as faces_module

    batch_frames: List[np.ndarray] = []
    batch_nums: List[int] = []
    processed = 0

    with pbar_ctx as pbar:
        for f_num, f_img in reader:
            batch_frames.append(f_img)
            batch_nums.append(f_num)

            if len(batch_frames) >= _INFERENCE_BATCH_SIZE:
                # Detect faces for this batch
                face_map = faces_module.handle(
                    batch_frames,
                    batch_nums,
                    output_dir,
                    config=faces_config,
                    writer=writer,
                    debug=debug,
                )
                for faces in face_map.values():
                    all_faces.extend(faces)

                # Process YOLO objects (det + seg + pose + tracking)
                _process_batch(
                    batch_frames,
                    batch_nums,
                    input_file,
                    fps,
                    det_model,
                    seg_model,
                    pose_model,
                    tracker,
                    face_map=face_map,
                    object_conf_threshold=object_conf_threshold,
                    segmentation_conf_threshold=segmentation_conf_threshold,
                    iou_match_threshold=iou_match_threshold,
                    keypoint_conf_threshold=keypoint_conf_threshold,
                    classes_to_detect=cls_filter,
                    kp_class_names=kp_cls,
                    output_dir=output_dir,
                    save_annotations=save_annotations,
                    perform_clustering=perform_clustering,
                    class_dir_cache=class_dir_cache,
                    writer=writer,
                    box_annotator=box_ann,
                    label_annotator=lbl_ann,
                    mask_annotator=mask_ann,
                    polygon_annotator=poly_ann,
                    results_list=results_list,
                    fallback_tid_counter=fallback_tid_counter,
                    person_face_embeddings=person_face_embeddings,
                    debug=debug,
                )
                processed += len(batch_frames)
                if pbar:
                    pbar.update(len(batch_frames))
                update_sub_progress(processed, total_frames, "frames")
                batch_frames = []
                batch_nums = []

        # Remainder
        if batch_frames:
            face_map = faces_module.handle(
                batch_frames,
                batch_nums,
                output_dir,
                config=faces_config,
                writer=writer,
                debug=debug,
            )
            for faces in face_map.values():
                all_faces.extend(faces)

            _process_batch(
                batch_frames,
                batch_nums,
                input_file,
                fps,
                det_model,
                seg_model,
                pose_model,
                tracker,
                face_map=face_map,
                object_conf_threshold=object_conf_threshold,
                segmentation_conf_threshold=segmentation_conf_threshold,
                iou_match_threshold=iou_match_threshold,
                keypoint_conf_threshold=keypoint_conf_threshold,
                classes_to_detect=cls_filter,
                kp_class_names=kp_cls,
                output_dir=output_dir,
                save_annotations=save_annotations,
                perform_clustering=perform_clustering,
                class_dir_cache=class_dir_cache,
                writer=writer,
                box_annotator=box_ann,
                label_annotator=lbl_ann,
                mask_annotator=mask_ann,
                polygon_annotator=poly_ann,
                results_list=results_list,
                fallback_tid_counter=fallback_tid_counter,
                person_face_embeddings=person_face_embeddings,
                debug=debug,
            )
            processed += len(batch_frames)
            if pbar:
                pbar.update(len(batch_frames))
            update_sub_progress(processed, total_frames, "frames")

    reader.stop()
    writer.shutdown()

    # Build JSON output
    json_payload: Dict[str, Any] = {"frames": results_list}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clustering
    face_summary = None
    object_summary = None
    if perform_clustering:
        faces_dir = os.path.join(output_dir, "faces")
        face_summary = _cluster_faces(
            all_faces,
            faces_dir,
            debug=debug,
            face_cluster_base_eps=face_cluster_base_eps,
            face_cluster_min_samples=face_cluster_min_samples,
            face_cluster_dedup_threshold=face_cluster_dedup_threshold,
            face_cluster_noise_max_distance=face_cluster_noise_max_distance,
            face_cluster_min_attempts=face_cluster_min_attempts,
            keyframe_eps=keyframe_eps,
            keyframe_min_samples=keyframe_min_samples,
            keyframe_hamming_frac=keyframe_hamming_frac,
            keyframe_require_both=keyframe_require_both,
            clip_model_name=clip_model_name,
        )
        object_summary = _cluster_objects(
            class_dir_cache,
            debug=debug,
            cluster_base_eps=cluster_base_eps,
            cluster_min_samples=cluster_min_samples,
            cluster_dedup_threshold=cluster_dedup_threshold,
            cluster_noise_max_distance=cluster_noise_max_distance,
            keyframe_eps=keyframe_eps,
            keyframe_min_samples=keyframe_min_samples,
            keyframe_hamming_frac=keyframe_hamming_frac,
            keyframe_require_both=keyframe_require_both,
            cluster_min_attempts=cluster_min_attempts,
            clip_model_name=clip_model_name,
            person_face_embeddings=person_face_embeddings,
            face_cluster_base_eps=face_cluster_base_eps,
            face_cluster_min_samples=face_cluster_min_samples,
            face_cluster_dedup_threshold=face_cluster_dedup_threshold,
            face_cluster_noise_max_distance=face_cluster_noise_max_distance,
            face_cluster_min_attempts=face_cluster_min_attempts,
        )

    if face_summary:
        json_payload["face_clusters"] = face_summary
    if object_summary:
        json_payload["object_clusters"] = object_summary

    with open(os.path.join(output_dir, "objects.json"), "w") as f:
        json.dump(json_payload, f, indent=4, cls=_NumpyEncoder)

    # Clean up annotation-only directories when annotations are not requested
    if not save_annotations:
        for d in set(class_dir_cache.values()):
            _cleanup_annotation_dirs(d)

    return output_dir, results_list
