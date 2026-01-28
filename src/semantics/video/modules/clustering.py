from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager, nullcontext
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

import cv2 as _cv
import numpy as np
from tqdm import tqdm as _tqdm

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import ClusteringConfig

__all__ = ["handle"]


def _probe_frame_indices(video_file: str, sample_fps: float) -> List[int]:
    capture = _cv.VideoCapture(video_file)
    if not capture.isOpened():
        return []

    total_frames = int(capture.get(_cv.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(_cv.CAP_PROP_FPS) or 0.0)
    capture.release()

    if total_frames <= 0:
        return []

    if fps <= 1e-6 or sample_fps <= 0:
        step = max(int(round(fps)) if fps > 1e-6 else 1, 1)
        return list(range(0, total_frames, step))

    ratio = fps / sample_fps
    step = max(int(round(ratio)), 1)
    return list(range(0, total_frames, step))


def handle(
    input_file: str,
    output_folder: str,
    config: "ClusteringConfig | None" = None,
    *,
    debug: bool = False,
    device=None,
    workers=None,
) -> Tuple[str, List[dict], str]:
    """Main entry point for frame clustering.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: ClusteringConfig instance or None for defaults.
        debug: Enable verbose debug output.
        device: Device for inference (cuda/cpu).
        workers: Number of parallel workers.

    Returns:
        Tuple of (clusters_folder, frame_data, clusters_json_path).
    """
    return _cluster_frames(
        input_file,
        output_folder,
        device=device,
        workers=workers,
        mn_threshold=config.mn_threshold if config else 0.995,
        mn_eps=config.mn_eps if config else 0.2,
        mn_min_samples=config.mn_min_samples if config else 3,
        fps=config.fps if config else 5.0,
        keyframes=config.keyframes if config else True,
        kf_eps=config.kf_eps if config else 0.18,
        kf_min_samples=config.kf_min_samples if config else 1,
        kf_hamming_frac=config.kf_hamming_frac if config else 0.30,
        kf_require_both=config.kf_require_both if config else True,
        save_keyframes=config.save_keyframes if config else True,
        save_clusters=config.save_clusters if config else False,
        debug=debug,
    )


def _cluster_frames(
    video_file,
    output_folder,
    device=None,
    workers=None,
    mn_threshold: float = 0.995,
    mn_eps: float = 0.2,
    mn_min_samples: int = 3,
    fps: float = 5.0,
    keyframes: bool = True,
    kf_eps: float = 0.18,
    kf_min_samples: int = 1,
    kf_hamming_frac: float = 0.30,
    kf_require_both: bool = True,
    save_keyframes: bool = True,
    save_clusters: bool = False,
    debug: bool = False,
) -> Tuple[str, List[dict], str]:
    clusters_root = os.path.join(output_folder, "frames", "clusters")
    report_path = os.path.join(clusters_root, "clusters.json")

    fps_arg = fps if (isinstance(fps, (int, float)) and fps > 0) else None
    print(f"INFO: Clustering frames at {fps_arg if fps_arg is not None else 'all'} fps")

    frame_numbers: List[int] = _probe_frame_indices(video_file, fps_arg or 1.0)
    frame_data: List[dict] = [{"index": int(idx)} for idx in frame_numbers]

    for _pil_logger in ("PIL", "PIL.Image", "PIL.PngImagePlugin"):
        logging.getLogger(_pil_logger).setLevel(logging.WARNING)

    debug_print(f"Collected {len(frame_numbers)} candidate frames", debug=debug)

    if not frame_numbers:
        print("No frames available for clustering")
        return clusters_root, [], report_path

    output_folder = clusters_root

    # Clean existing clusters folder before starting
    if os.path.isdir(output_folder):
        print("INFO: Cleaning existing clusters folder")
        debug_print(f"Removing {output_folder}", debug=debug)
        try:
            shutil.rmtree(output_folder)
        except Exception as e:
            print(f"Warning: Failed to clean clusters folder: {e}")

    os.makedirs(output_folder, exist_ok=True)

    video_abs_path = os.path.abspath(video_file)

    # Open the video and loop only through the selected frame indices (no clustering yet)
    print("INFO: Reading frames & extracting features")
    cap = _cv.VideoCapture(video_file)
    if not cap.isOpened():
        print("Failed to open video for reading")
        return output_folder, frame_data, report_path

    processed = 0
    # Prepare clusters.json inside the 'clusters' output folder
    frames_entries = []
    # Write initial empty report
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"clusters": 0, "frames": []}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to initialize clusters.json: {e}")

    # Load MobileNet to extract features for clustering (only for selected frames) - same as original script
    import torch as _torch
    import timm as _timm

    with gray_debug_output(debug):
        dev = device if device in ("cpu", "cuda") else ("cuda" if _torch.cuda.is_available() else "cpu")
        model = _timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0).to(dev)
        model.eval()
        data_config = _timm.data.resolve_model_data_config(model)
        transform = _timm.data.create_transform(**data_config, is_training=False)

    debug_print(f"Using device '{dev}' for clustering", debug=debug)
    def _feat_from_bgr(frame_bgr):
        from PIL import Image as _Image
        frame_rgb = _cv.cvtColor(frame_bgr, _cv.COLOR_BGR2RGB)
        pil = _Image.fromarray(frame_rgb)
        with _torch.no_grad():
            inp = transform(pil).unsqueeze(0).to(dev)
            feat = model(inp)
        return feat.detach().cpu().numpy().flatten()
    def _feat_from_pil(pil_image):
        with _torch.no_grad():
            inp = transform(pil_image.convert("RGB")).unsqueeze(0).to(dev)
            feat = model(inp)
        return feat.detach().cpu().numpy().flatten()

    # Helper to wrap iterables with tqdm only when debug progress is desired
    def _progress_iter(it: Iterable, desc: Optional[str] = None, unit: Optional[str] = None):
        if _tqdm and debug:
            kwargs = {}
            if desc is not None:
                kwargs["desc"] = desc
            if unit is not None:
                kwargs["unit"] = unit
            try:
                tqdm_iter = _tqdm(it, colour="#888888", **kwargs)
            except TypeError:
                tqdm_iter = _tqdm(it, **kwargs)

            @contextmanager
            def _ctx():
                with gray_debug_output(True):
                    try:
                        yield
                    finally:
                        close_fn = getattr(tqdm_iter, "close", None)
                        if callable(close_fn):
                            close_fn()

            return tqdm_iter, _ctx()
        return it, nullcontext()

    iterable, progress_ctx = _progress_iter(frame_numbers, desc="Reading frames", unit="frame")
    selected_indices: List[int] = []
    selected_paths: List[str] = []
    feats: List[np.ndarray] = []
    with progress_ctx:
        for idx in iterable:
            try:
                cap.set(_cv.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    # Could not read this frame index; skip
                    continue
                # Placeholder: here we'll compute features and cluster later
                processed += 1
                # Collect for clustering and final report (reference back to the source video)
                frame_reference = f"{video_abs_path}#frame_{int(idx):08d}"
                selected_indices.append(int(idx))
                selected_paths.append(frame_reference)
                try:
                    feats.append(_feat_from_bgr(frame))
                except Exception:
                    # If feature extraction fails, append a dummy vector to keep alignment
                    feats.append(np.zeros((1,), dtype=np.float32))
            except Exception:
                # Skip any problematic index
                continue

    cap.release()
    print(f"INFO: Processed {processed} / {len(frame_numbers)} selected frames from video")
    debug_print(f"Generated {len(feats)} feature vectors", debug=debug)
    # Perform clustering on the collected features (only selected frames) using the original logic
    if feats:
        try:
            print("INFO: Clustering features (DBSCAN on candidates)")
            from sklearn.cluster import DBSCAN as _DBSCAN
            feat_mat = np.stack(feats, axis=0)
            # Candidate selection via threshold between consecutive features
            cand_indices_local: list[int] = []
            cand_feats_local: list[np.ndarray] = []
            last_feat = None
            for j, f in enumerate(feat_mat):
                if last_feat is None:
                    cand_indices_local.append(j)
                    cand_feats_local.append(f)
                    last_feat = f
                else:
                    n1 = last_feat / (np.linalg.norm(last_feat) + 1e-8)
                    n2 = f / (np.linalg.norm(f) + 1e-8)
                    sim = float(np.dot(n1, n2))
                    if sim < float(mn_threshold):
                        cand_indices_local.append(j)
                        cand_feats_local.append(f)
                        last_feat = f

            # DBSCAN on candidate features (cosine)
            if len(cand_feats_local) == 0:
                labels = np.zeros((len(feat_mat),), dtype=int)
                centroids = None
            else:
                cand_mat = np.stack(cand_feats_local, axis=0)
                cand_labels = _DBSCAN(eps=float(mn_eps), min_samples=int(mn_min_samples), metric='cosine').fit_predict(cand_mat)
                # Reindex candidate labels
                uniq = sorted(set(int(l) for l in cand_labels if l >= 0))
                mapping = {old: new for new, old in enumerate(uniq)}
                for i_lab, l in enumerate(cand_labels):
                    if l >= 0:
                        cand_labels[i_lab] = mapping[int(l)]
                # Centroids per candidate cluster
                K = int(cand_labels.max() + 1) if cand_labels.size else 0
                centroids = []
                for cid in range(K):
                    centroids.append(cand_mat[cand_labels == cid].mean(axis=0))
                centroids = np.stack(centroids, axis=0) if K > 0 else None
                # Assign labels to all selected frames by nearest centroid
                labels = np.zeros((len(feat_mat),), dtype=int)
                if centroids is not None and centroids.shape[0] > 0:
                    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
                    for j, f in enumerate(feat_mat):
                        ff = f / (np.linalg.norm(f) + 1e-8)
                        dists = 1.0 - np.dot(C, ff)
                        labels[j] = int(np.argmin(dists))
                else:
                    labels[:] = 0

            # Reindex final labels to 0..K-1 keeping -1 (not present here) consistent
            labels = labels.copy()
            uniq_final = sorted(set(int(l) for l in labels if l >= 0))
            mapping_final = {old: new for new, old in enumerate(uniq_final)}
            for i_lab, l in enumerate(labels):
                if l >= 0:
                    labels[i_lab] = mapping_final[int(l)]

            # Build final frames entries including cluster_number and optional keyframe flag
            frames_entries = []
            for k, (idx_val, path_val, lbl) in enumerate(zip(selected_indices, selected_paths, labels)):
                entry = {
                    "index": int(idx_val),
                    "frame_index": int(idx_val),
                    "path": path_val,
                    "cluster_number": int(lbl),
                }
                frames_entries.append(entry)
            clusters_count = len(set(int(l) for l in labels if int(l) >= 0))
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"clusters": clusters_count, "frames": frames_entries}, f, ensure_ascii=False, indent=2)
            frame_data = frames_entries
        except Exception as e:
            print(f"Failed to cluster selected frames: {e}")

    # Final confirmation of report path
    if os.path.exists(report_path):
        debug_print(f"Cluster report contains {len(frames_entries)} entries", debug=debug)

    # Optionally save clustered frames into folders like the original script
    if feats and save_clusters:
        out_dir = output_folder  # already points to .../clusters
        os.makedirs(out_dir, exist_ok=True)
        cap2 = _cv.VideoCapture(video_file)
        if not cap2.isOpened():
            print("Failed to reopen video for writing frames")
        else:
            iterable2, progress_ctx = _progress_iter(range(len(selected_indices)), desc="Writing PNGs", unit="img")
            # Optional threaded writing if workers > 0
            pool = None
            futures = []
            limit = max(64, int(workers) * 4) if (workers and int(workers) > 0) else 0
            if workers and int(workers) > 0:
                import concurrent.futures as _cf
                pool = _cf.ThreadPoolExecutor(max_workers=int(workers))
            def _write_png(path_dst: str, img) -> bool:
                return _cv.imwrite(path_dst, img)
            with progress_ctx:
                for k in iterable2:
                    idx_val = int(selected_indices[k])
                    lbl = int(labels[k]) if 'labels' in locals() else 0
                    cap2.set(_cv.CAP_PROP_POS_FRAMES, idx_val)
                    ret, frame = cap2.read()
                    if not ret:
                        continue
                    cdir = os.path.join(out_dir, str(lbl))
                    os.makedirs(cdir, exist_ok=True)
                    dst = os.path.join(cdir, f"{idx_val:08d}.png")
                    if pool:
                        if len(futures) >= limit:
                            futures.pop(0).result()
                        futures.append(pool.submit(_write_png, dst, frame.copy()))
                    else:
                        _cv.imwrite(dst, frame)
            if futures:
                for f in futures:
                    f.result()
            if pool:
                pool.shutdown(wait=True)
            cap2.release()
            debug_print("Finished writing clustered frame images", debug=debug)

    # Post-clustering keyframes step (mirrors original behavior)
    if save_keyframes and not keyframes:
        # Mirror original behavior: keyframes are only extracted when 'keyframes' is True
        print("Keyframe saving requested but 'keyframes' flag is False; skipping per original logic.")
    if keyframes and save_keyframes:
        print("INFO: Extracting keyframes per cluster")
        # Helper pHash functions
        def _dct_matrix(n: int) -> np.ndarray:
            k = np.arange(n)[:, None]
            n_ = np.arange(n)[None, :]
            mat = np.cos(np.pi * (n_ + 0.5) * k / n)
            mat[0, :] = mat[0, :] / np.sqrt(n)
            mat[1:, :] = mat[1:, :] * np.sqrt(2 / n)
            return mat.astype(np.float32)
        def _dct_2d(a: np.ndarray) -> np.ndarray:
            N, M = a.shape
            Cn = _dct_matrix(N)
            Cm = _dct_matrix(M)
            return Cn @ a @ Cm.T
        def _phash64_pil(im, hash_size: int = 8, highfreq_factor: int = 4) -> int:
            from PIL import Image as _Image
            size = hash_size * highfreq_factor
            img = im.convert("L").resize((size, size), _Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=np.float32)
            dct = _dct_2d(pixels)
            low = dct[:hash_size, :hash_size].flatten()
            med = np.median(low[1:]) if low.size > 1 else low[0]
            bits = (low > med).astype(np.uint8)
            v = 0
            for b in bits:
                v = (v << 1) | int(b)
            return int(v)
        from sklearn.cluster import DBSCAN as _DBSCAN2
        out_dir = output_folder
        os.makedirs(out_dir, exist_ok=True)
        # Build label map for selected frames only if needed (video path without saved clusters)
        if save_clusters:
            # Compute keyframes by scanning images saved in each cluster directory (replicates original logic)
            from PIL import Image as _Image
            try:
                cluster_dirs = [
                    d for d in sorted(
                        os.listdir(out_dir),
                        key=lambda n: (0, int(n)) if n.lstrip('-').isdigit() else (1, n),
                    )
                    if os.path.isdir(os.path.join(out_dir, d)) and d != 'keyframes'
                ]
            except Exception:
                cluster_dirs = []

            iterable_dirs, progress_dirs = _progress_iter(cluster_dirs, desc='Keyframes', unit='dir')
            with progress_dirs:
                for cname in iterable_dirs:
                    cdir = os.path.join(out_dir, cname)
                    files = [
                        os.path.join(cdir, fn)
                        for fn in sorted(os.listdir(cdir))
                        if os.path.isfile(os.path.join(cdir, fn))
                        and os.path.splitext(fn)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
                    ]
                    if not files:
                        continue

                    feats_kf: list[np.ndarray] = []
                    phashes: list[int] = []
                    iterable_f, progress_files = _progress_iter(files, desc=f'Feats {cname}', unit='img')
                    with progress_files:
                        for fp in iterable_f:
                            try:
                                with _Image.open(fp) as im:
                                    feats_kf.append(_feat_from_pil(im))
                            except Exception:
                                feats_kf.append(np.zeros((1,), dtype=np.float32))

                    for fp in files:
                        try:
                            with _Image.open(fp) as im:
                                phashes.append(_phash64_pil(im))
                        except Exception:
                            phashes.append(0)

                    fm = np.stack(feats_kf, axis=0) if feats_kf else None
                    if fm is None:
                        continue

                    labels_sub = _DBSCAN2(
                        eps=float(kf_eps),
                        min_samples=int(kf_min_samples),
                        metric='cosine',
                        n_jobs=-1,
                    ).fit_predict(fm)

                    uniq_sub = sorted(set(int(l) for l in labels_sub if l >= 0))
                    map_sub = {old: new for new, old in enumerate(uniq_sub)}
                    for i2, l2 in enumerate(labels_sub):
                        if l2 >= 0:
                            labels_sub[i2] = map_sub[int(l2)]

                    selected_files: list[str] = []
                    for sub in sorted({int(l) for l in labels_sub if l >= 0}):
                        idxs_local = [i for i, l in enumerate(labels_sub) if int(l) == sub]
                        chosen: list[int] = []
                        for iidx in idxs_local:
                            if not chosen:
                                chosen.append(iidx)
                                continue
                            cos_dists = []
                            hamm_fracs = []
                            for j in chosen:
                                f1 = fm[iidx] / (np.linalg.norm(fm[iidx]) + 1e-8)
                                f2 = fm[j] / (np.linalg.norm(fm[j]) + 1e-8)
                                cos_dists.append(1.0 - float(np.dot(f1, f2)))
                                hamm_fracs.append(
                                    (((phashes[iidx] ^ phashes[j]) & ((1 << 64) - 1)).bit_count()) / 64.0
                                )
                            min_cos = min(cos_dists) if cos_dists else 1.0
                            min_hamm = min(hamm_fracs) if hamm_fracs else 1.0
                            different = (min_cos >= kf_eps) or (min_hamm >= kf_hamming_frac)
                            if kf_require_both:
                                different = (min_cos >= kf_eps) and (min_hamm >= kf_hamming_frac)
                            if different:
                                chosen.append(iidx)
                        selected_files.extend([files[s] for s in chosen])

                    kdir = os.path.join(cdir, 'keyframes')
                    os.makedirs(kdir, exist_ok=True)
                    if not selected_files and files:
                        selected_files = [files[0]]

                    used_names: set[str] = set()
                    for src in selected_files:
                        base = os.path.basename(src)
                        dst = os.path.join(kdir, base)
                        suffix = 1
                        while os.path.basename(dst) in used_names or os.path.exists(dst):
                            stem, ext = os.path.splitext(base)
                            dst = os.path.join(kdir, f'{stem}_{suffix}{ext}')
                            suffix += 1
                        used_names.add(os.path.basename(dst))
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            pass
        else:
            # Compute keyframes directly from the video for each cluster's frame indices
            if 'labels' not in locals():
                print('Keyframes requested without saved clusters, but labels are unavailable; skipping.')
            else:
                # Build label map from labels
                label_map = {}
                for idx_val, lbl in zip(selected_indices, labels):
                    label_map.setdefault(int(lbl), []).append(int(idx_val))
                cap3 = _cv.VideoCapture(video_file)
                if cap3.isOpened():
                    iterable_clusters, progress_clusters = _progress_iter(sorted(label_map.keys()), desc='Keyframes', unit='cluster')
                    with progress_clusters:
                        for cid in iterable_clusters:
                            idxs = sorted(label_map[cid])
                            feats_kf = []
                            phashes = []
                            iterable_idx, progress_idx = _progress_iter(idxs, desc=f'Feats {cid}', unit='frame')
                            with progress_idx:
                                for idx_val in iterable_idx:
                                    cap3.set(_cv.CAP_PROP_POS_FRAMES, int(idx_val))
                                    ret, frame = cap3.read()
                                    if not ret:
                                        continue
                                    feats_kf.append(_feat_from_bgr(frame))
                                    try:
                                        from PIL import Image as _Image
                                        im = _Image.fromarray(_cv.cvtColor(frame, _cv.COLOR_BGR2RGB))
                                        phashes.append(_phash64_pil(im))
                                    except Exception:
                                        phashes.append(0)
                            if not feats_kf:
                                continue
                            fm = np.stack(feats_kf, axis=0)
                            labels_sub = _DBSCAN2(
                                eps=float(kf_eps),
                                min_samples=int(kf_min_samples),
                                metric='cosine',
                            ).fit_predict(fm)
                            labels_sub = labels_sub.copy()
                            uniq_sub = sorted(set(int(l) for l in labels_sub if l >= 0))
                            map_sub = {old: new for new, old in enumerate(uniq_sub)}
                            for i2, l2 in enumerate(labels_sub):
                                if l2 >= 0:
                                    labels_sub[i2] = map_sub[int(l2)]
                            selected_idx = []
                            for sub in sorted(set(int(l) for l in labels_sub if l >= 0)):
                                idxs_local = [i for i, l in enumerate(labels_sub) if int(l) == sub]
                                chosen = []
                                for iidx in idxs_local:
                                    if not chosen:
                                        chosen.append(iidx)
                                        continue
                                    cos_dists = []
                                    hamm_fracs = []
                                    for j in chosen:
                                        f1 = fm[iidx] / (np.linalg.norm(fm[iidx]) + 1e-8)
                                        f2 = fm[j] / (np.linalg.norm(fm[j]) + 1e-8)
                                        cos_dists.append(1.0 - float(np.dot(f1, f2)))
                                        hamm_fracs.append(
                                            (((phashes[iidx] ^ phashes[j]) & ((1 << 64) - 1)).bit_count()) / 64.0
                                        )
                                    min_cos = min(cos_dists) if cos_dists else 1.0
                                    min_hamm = min(hamm_fracs) if hamm_fracs else 1.0
                                    different = (min_cos >= kf_eps) or (min_hamm >= kf_hamming_frac)
                                    if kf_require_both:
                                        different = (min_cos >= kf_eps) and (min_hamm >= kf_hamming_frac)
                                    if different:
                                        chosen.append(iidx)
                                selected_idx.extend([idxs[s] for s in chosen])
                            cdir = os.path.join(out_dir, str(cid))
                            os.makedirs(cdir, exist_ok=True)
                            kdir = os.path.join(cdir, 'keyframes')
                            os.makedirs(kdir, exist_ok=True)
                            if not selected_idx and idxs:
                                selected_idx = [idxs[0]]
                            for idx_sel in selected_idx:
                                cap3.set(_cv.CAP_PROP_POS_FRAMES, int(idx_sel))
                                ret, frame = cap3.read()
                                if not ret:
                                    continue
                                dst = os.path.join(kdir, f'{idx_sel:08d}.png')
                                _cv.imwrite(dst, frame)
                    cap3.release()

        # Update JSON to include keyframe flags (only when labels exist)
        try:
            key_set = set()
            # Collect keyframe names across clusters and map to frame indices
            try:
                cluster_dirs_present = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d)) and d != 'keyframes']
            except Exception:
                cluster_dirs_present = []
            for cname in cluster_dirs_present:
                kdir = os.path.join(out_dir, cname, 'keyframes')
                if not os.path.isdir(kdir):
                    continue
                for name in os.listdir(kdir):
                    stem, ext = os.path.splitext(name)
                    if stem.isdigit():
                        key_set.add((int(cname) if cname.lstrip('-').isdigit() else cname, int(stem)))
            if 'labels' in locals():
                frames_entries2 = []
                for idx_val, path_val, lbl in zip(selected_indices, selected_paths, labels):
                    frames_entries2.append({
                        'index': int(idx_val),
                        'frame_index': int(idx_val),
                        'path': path_val,
                        'cluster_number': int(lbl),
                        'keyframe': (int(lbl), int(idx_val)) in key_set,
                    })
                # Determine cluster count either from labels or current cluster directories
                if 'clusters_count' not in locals():
                    clusters_count_local = len(cluster_dirs_present)
                else:
                    clusters_count_local = clusters_count
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump({'clusters': clusters_count_local, 'frames': frames_entries2}, f, ensure_ascii=False, indent=2)
                frame_data = frames_entries2
            debug_print("Cluster report updated with keyframe annotations", debug=debug)
        except Exception as _e:
            print(f'Failed to update keyframes in JSON: {_e}')

    # For now, just return the list of frame numbers to be processed
    keyframes_only = [entry for entry in frame_data if entry.get("keyframe")]
    if keyframes_only:
        frame_data = keyframes_only
    elif frame_data:
        # Ensure bare entries carry only the index field when keyframes were unavailable
        cleaned = []
        seen = set()
        for entry in frame_data:
            idx_val = entry.get("index") or entry.get("frame_index")
            try:
                idx_int = int(idx_val)
            except Exception:
                continue
            if idx_int in seen:
                continue
            seen.add(idx_int)
            cleaned.append({"index": idx_int})
        frame_data = cleaned
    return output_folder, frame_data, report_path