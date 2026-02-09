from __future__ import annotations

import json
import logging
import os
import shutil
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING, Any

import cv2 as _cv
import numpy as np
from tqdm import tqdm as _tqdm
from PIL import Image as _Image

from .utils.logging import debug_print, gray_debug_output, info_print, update_sub_progress

if TYPE_CHECKING:
    from config import ClusteringConfig

__all__ = ["handle"]

# Tuning Constants
BATCH_SIZE = 96  # Increased for higher GPU saturation
QUEUE_SIZE = 8   # Number of batches to buffer
SMART_SEEK_THRESHOLD = 32

def _probe_frame_indices(video_file: str, sample_fps: float) -> List[int]:
    """Determines which frame indices to sample based on FPS."""
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


class AsyncVideoLoader:
    """
    Reads video frames in a separate thread, applies transforms in parallel workers,
    and feeds a queue for the GPU consumer.
    """
    def __init__(self, video_path: str, indices: List[int], batch_size: int, transform: Any, device: str):
        self.video_path = video_path
        self.indices = sorted(indices)
        self.batch_size = batch_size
        self.transform = transform
        self.device = device
        self.queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.loader_thread = None
        
    def start(self):
        self.loader_thread = threading.Thread(target=self._worker, daemon=True)
        self.loader_thread.start()
        
    def _worker(self):
        import torch
        cap = _cv.VideoCapture(self.video_path)
        current_pos = -1
        
        batch_frames = []
        batch_indices = []
        
        # Helper for transforming a single frame (runs in thread pool)
        def _prep_frame(bgr_img):
            # Convert BGR -> RGB -> PIL -> Tensor
            rgb = _cv.cvtColor(bgr_img, _cv.COLOR_BGR2RGB)
            pil = _Image.fromarray(rgb)
            return self.transform(pil)

        # ThreadPool for CPU-bound transformations
        # Workers = 4 is usually sweet spot for feeding one GPU
        with ThreadPoolExecutor(max_workers=4) as pool:
            for target_idx in self.indices:
                if self.stop_event.is_set():
                    break
                
                # Smart Seek Logic
                gap = target_idx - current_pos - 1
                if gap == 0:
                    pass
                elif 0 < gap < SMART_SEEK_THRESHOLD:
                    for _ in range(gap): cap.grab()
                else:
                    cap.set(_cv.CAP_PROP_POS_FRAMES, float(target_idx))
                
                ret, frame = cap.read()
                current_pos = target_idx
                
                if not ret:
                    break
                
                batch_frames.append(frame)
                batch_indices.append(target_idx)
                
                if len(batch_frames) >= self.batch_size:
                    # Parallel Transform
                    tensors = list(pool.map(_prep_frame, batch_frames))
                    
                    # Stack on CPU, pin memory for faster transfer if CUDA
                    try:
                        batch_tensor = torch.stack(tensors)
                        if self.device == 'cuda':
                            batch_tensor = batch_tensor.pin_memory()
                    except Exception:
                        batch_tensor = None
                        
                    self.queue.put((batch_indices, batch_tensor, [f.copy() for f in batch_frames]))
                    
                    batch_frames = []
                    batch_indices = []

            # Cleanup remaining
            if batch_frames and not self.stop_event.is_set():
                tensors = list(pool.map(_prep_frame, batch_frames))
                try:
                    batch_tensor = torch.stack(tensors)
                except Exception:
                    batch_tensor = None
                self.queue.put((batch_indices, batch_tensor, [f.copy() for f in batch_frames]))

        cap.release()
        self.queue.put(None) # Sentinel

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def stop(self):
        self.stop_event.set()
        # Drain queue to allow thread to exit
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Exception: pass


def handle(
    input_file: str,
    output_folder: str,
    config: "ClusteringConfig | None" = None,
    *,
    save_frames: bool = False,
    debug: bool = False,
    device=None,
    workers=None,
) -> Tuple[str, List[dict], str]:
    """Main entry point for frame clustering."""
    
    # User Request:
    # 1. If save_frames=True, FORCE save both keyframes and clusters.
    # 2. If save_frames=False, FORCE save neither (avoid empty folders).
    if save_frames:
        should_save_keyframes = True
        should_save_clusters = True
    else:
        should_save_keyframes = False
        should_save_clusters = False

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
        save_keyframes=should_save_keyframes,
        save_clusters=should_save_clusters,
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
    info_print(f"Clustering frames at {fps_arg if fps_arg is not None else 'all'} fps")

    frame_numbers: List[int] = _probe_frame_indices(video_file, fps_arg or 1.0)
    frame_data: List[dict] = [{"index": int(idx)} for idx in frame_numbers]

    # Silence PIL
    for _pil_logger in ("PIL", "PIL.Image", "PIL.PngImagePlugin"):
        logging.getLogger(_pil_logger).setLevel(logging.WARNING)

    debug_print(f"Collected {len(frame_numbers)} candidate frames", debug=debug)

    if not frame_numbers:
        print("No frames available for clustering")
        return clusters_root, [], report_path

    if os.path.isdir(clusters_root):
        info_print("Cleaning existing clusters folder")
        try:
            shutil.rmtree(clusters_root)
        except Exception as e:
            print(f"Warning: Failed to clean clusters folder: {e}")

    os.makedirs(clusters_root, exist_ok=True)
    video_abs_path = os.path.abspath(video_file)

    # --- Step 1: Model Setup (With FP16) ---
    import torch as _torch
    import timm as _timm

    info_print("Reading frames & extracting features")
    
    with gray_debug_output(debug):
        dev = device if device in ("cpu", "cuda") else ("cuda" if _torch.cuda.is_available() else "cpu")
        model = _timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0).to(dev)
        model.eval()
        
        # Use FP16 for speed if on CUDA
        if dev == 'cuda':
            model.half()
            
        data_config = _timm.data.resolve_model_data_config(model)
        transform = _timm.data.create_transform(**data_config, is_training=False)

    debug_print(f"Using device '{dev}' (FP16 enabled)" if dev == 'cuda' else f"Using device '{dev}'", debug=debug)

    # --- Step 2: Pipelined Feature Extraction ---
    # The AsyncVideoLoader handles IO and CPU transform in background threads
    loader = AsyncVideoLoader(video_file, frame_numbers, BATCH_SIZE, transform, dev)
    loader.start()

    selected_indices: List[int] = []
    selected_paths: List[str] = []
    feats: List[np.ndarray] = []
    
    # We may need raw frames later for saving, but keeping them all in memory is OOM risk.
    # We will re-read them if needed (SmartSeek makes this tolerable), 
    # OR we write them immediately if save_clusters=True is likely.
    # However, we don't know the cluster labels yet. So we can't save them to cluster folders.
    # We MUST re-read later. 

    processed_count = 0
    total_cluster_frames = len(frame_numbers)
    pbar_ctx = _tqdm(total=total_cluster_frames, desc="Extracting features", unit="frame", colour="#888888") if debug else nullcontext()

    with pbar_ctx as pbar:
        try:
            for b_indices, b_tensor, _ in loader:
                if b_tensor is not None:
                    # Move to GPU
                    # non_blocking=True allows overlap with CPU work
                    inp = b_tensor.to(dev, non_blocking=True)
                    if dev == 'cuda':
                        inp = inp.half()
                    
                    with _torch.no_grad():
                        out = model(inp).cpu().numpy().astype(np.float32)
                    
                    for i, f_vec in enumerate(out):
                        idx_val = b_indices[i]
                        selected_indices.append(idx_val)
                        selected_paths.append(f"{video_abs_path}#frame_{int(idx_val):08d}")
                        feats.append(f_vec.flatten())
                
                cnt = len(b_indices)
                processed_count += cnt
                if pbar: pbar.update(cnt)
                update_sub_progress(processed_count, total_cluster_frames, "frames")
        except Exception as e:
            print(f"Error in feature extraction loop: {e}")
        finally:
            loader.stop()

    info_print(f"Processed {processed_count} frames")

    # --- Step 3: Clustering (DBSCAN) ---
    labels = np.zeros(len(feats), dtype=int)
    
    if feats:
        try:
            info_print("Clustering features (DBSCAN)")
            from sklearn.cluster import DBSCAN as _DBSCAN
            feat_mat = np.stack(feats, axis=0)
            
            # 1. Candidate Generation (Linear Scan)
            cand_indices_local = []
            cand_feats_local = []
            
            # Pre-normalize for dot product speed
            norms = np.linalg.norm(feat_mat, axis=1, keepdims=True) + 1e-8
            normalized_feats = feat_mat / norms
            
            last_feat = None
            
            # Using simple loop is fastest for sequential dependency
            for j, f_norm in enumerate(normalized_feats):
                if last_feat is None:
                    cand_indices_local.append(j)
                    cand_feats_local.append(f_norm)
                    last_feat = f_norm
                else:
                    # dot product of normalized vectors = cosine similarity
                    sim = float(np.dot(last_feat, f_norm))
                    if sim < float(mn_threshold):
                        cand_indices_local.append(j)
                        cand_feats_local.append(f_norm)
                        last_feat = f_norm

            # 2. DBSCAN on Candidates
            if cand_feats_local:
                cand_mat = np.stack(cand_feats_local, axis=0)
                # Compute DBSCAN
                cand_labels = _DBSCAN(eps=float(mn_eps), min_samples=int(mn_min_samples), metric='cosine', n_jobs=-1).fit_predict(cand_mat)
                
                # 3. Assign remaining frames to Centroids
                unique_labels = set(cand_labels)
                if -1 in unique_labels: unique_labels.remove(-1)
                
                K = max(unique_labels) + 1 if unique_labels else 0
                
                if K > 0:
                    centroids = np.zeros((K, cand_mat.shape[1]), dtype=cand_mat.dtype)
                    for cid in range(K):
                        centroids[cid] = cand_mat[cand_labels == cid].mean(axis=0)
                    
                    # Normalize centroids
                    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
                    
                    # Matrix multiplication for all-pairs similarity (Vectorized assignment)
                    # (N, D) @ (K, D).T -> (N, K) cosine similarities
                    sims = np.dot(normalized_feats, centroids.T)
                    # 1 - sim = dist
                    labels = np.argmax(sims, axis=1) # argmax sim == argmin dist
                else:
                    labels[:] = 0

            # Reindex labels
            uniq_final = sorted(set(int(l) for l in labels if l >= 0))
            mapping_final = {old: new for new, old in enumerate(uniq_final)}
            labels = np.array([mapping_final.get(int(l), -1) for l in labels])

        except Exception as e:
            print(f"Clustering failed: {e}")
            labels = np.zeros(len(feats), dtype=int)

    # Build Report
    frames_entries = []
    for idx_val, path_val, lbl in zip(selected_indices, selected_paths, labels):
        frames_entries.append({
            "index": int(idx_val),
            "frame_index": int(idx_val),
            "path": path_val,
            "cluster_number": int(lbl),
        })
    
    clusters_count = len(set(int(l) for l in labels if int(l) >= 0))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"clusters": clusters_count, "frames": frames_entries}, f, ensure_ascii=False, indent=2)

    # --- Step 4: Save Clusters & Keyframes (Optimized I/O) ---
    # Always run keyframe selection if keyframes=True, but only save images if save_keyframes/save_clusters
    
    if keyframes and feats:
        
        # Prepare for Keyframe extraction (Precompute DCT matrices)
        def _get_dct_matrix(n: int) -> np.ndarray:
            k = np.arange(n)[:, None]
            n_ = np.arange(n)[None, :]
            mat = np.cos(np.pi * (n_ + 0.5) * k / n)
            mat[0, :] = mat[0, :] / np.sqrt(n)
            mat[1:, :] = mat[1:, :] * np.sqrt(2 / n)
            return mat.astype(np.float32)

        _DCT_SIZE = 32
        _DCT_MAT = _get_dct_matrix(_DCT_SIZE)
        _DCT_MAT_T = _DCT_MAT.T

        # Optimized pHash using OpenCV Resize
        def _batch_phash64_cv2(images_bgr: List[np.ndarray], hash_size=8) -> List[int]:
            if not images_bgr: return []
            
            # Batch prep: Resize and Color Convert
            # OpenCV resize is faster than PIL
            arrs = []
            for img in images_bgr:
                gray = _cv.cvtColor(img, _cv.COLOR_BGR2GRAY)
                # INTER_LANCZOS4 matches PIL LANCZOS closely
                small = _cv.resize(gray, (_DCT_SIZE, _DCT_SIZE), interpolation=_cv.INTER_LANCZOS4)
                arrs.append(small.astype(np.float32))
            
            batch = np.stack(arrs) 
            dct_batch = _DCT_MAT @ batch @ _DCT_MAT_T
            
            low_freq = dct_batch[:, :hash_size, :hash_size]
            flat = low_freq.reshape(len(images_bgr), -1)
            dc_removed = flat[:, 1:]
            medians = np.median(dc_removed, axis=1, keepdims=True)
            bits = (flat > medians)
            
            hashes = []
            powers = (1 << np.arange(64 - 1, -1, -1)).astype(np.uint64)
            for row in bits:
                val = np.sum(row * powers, dtype=np.uint64)
                hashes.append(int(val))
            return hashes

        # Map indices to labels
        idx_to_label = {idx: lbl for idx, lbl in zip(selected_indices, labels)}
        cluster_map = {}
        for idx, lbl in idx_to_label.items():
            cluster_map.setdefault(int(lbl), []).append(int(idx))
            
        key_set = set()

        # Re-open video for reading (Random access required)
        # To optimize, we read linearly and route frames to destinations
        
        if save_clusters or save_keyframes:
            info_print("Saving results and extracting keyframes")
        else:
            info_print("Selecting keyframes")
        
        # ThreadPool for writing images
        max_w = int(workers) if (workers and int(workers) > 0) else 4
        writer_pool = ThreadPoolExecutor(max_workers=max_w)
        writer_futures = []

        # We need another reader loop. 
        # To avoid re-reading the whole file, we iterate ONLY the selected indices again.
        # Smart Video Reader Logic Inline
        cap = _cv.VideoCapture(video_file)
        curr_pos = -1
        
        sorted_indices = sorted(selected_indices)
        
        # Batch accumulator for keyframe processing
        # We process keyframes cluster-by-cluster usually, but we are reading frame-by-frame.
        # Strategy: Accumulate frames in memory per cluster. 
        # When a cluster is "complete" (all frames read), process keyframes.
        # Problem: Large clusters = OOM.
        # Fallback: Write all frames first (if save_clusters). Then read for keyframes?
        # Better: Accumulate small batches per cluster and compute partials? No, DBSCAN needs all.
        
        # Hybrid Approach:
        # 1. Read frame.
        # 2. If save_clusters: submit write job.
        # 3. If save_keyframes: Store features/phash for this frame in memory dict {cluster_id: [data]}. 
        #    Wait, we need the image for pHash. Storing all images is OOM.
        #    Optimization: Compute feature & pHash IMMEDIATELY upon read. Store tiny vector. Discard image.
        
        # Pre-compute features map to avoid re-inference
        idx_to_feat = {idx: f for idx, f in zip(selected_indices, feats)}
        
        # Storage for Keyframe Logic: {cluster_id: {'idxs': [], 'feats': [], 'phashes': []}}
        kf_data = {cid: {'idxs': [], 'feats': [], 'phashes': []} for cid in cluster_map}

        total_final_pass = len(sorted_indices)
        pbar = _tqdm(total=total_final_pass, desc="Final Pass", unit="frame", colour="#888888") if debug else nullcontext()
        final_pass_count = 0
        
        with pbar as pb:
            for idx in sorted_indices:
                # Seek/Grab
                gap = idx - curr_pos - 1
                if 0 < gap < SMART_SEEK_THRESHOLD:
                    for _ in range(gap): cap.grab()
                elif gap != 0:
                    cap.set(_cv.CAP_PROP_POS_FRAMES, float(idx))
                
                ret, frame = cap.read()
                curr_pos = idx
                if not ret: break

                lbl = idx_to_label.get(idx, 0)
                
                # 1. Save Cluster Image (Async)
                if save_clusters:
                    cdir = os.path.join(clusters_root, str(lbl))
                    if not os.path.exists(cdir): os.makedirs(cdir, exist_ok=True)
                    dst = os.path.join(cdir, f"{idx:08d}.png")
                    
                    # Copy frame to avoid thread race on buffer
                    f_copy = frame.copy()
                    
                    # Clean finished futures
                    if len(writer_futures) > max_w * 4:
                        done, not_done = wait(writer_futures, return_when="FIRST_COMPLETED")
                        writer_futures = list(not_done)
                    writer_futures.append(writer_pool.submit(_cv.imwrite, dst, f_copy))

                # 2. Keyframe Data Collection (Immediate Compute)
                if keyframes:
                    # pHash (Fast OpenCV version)
                    # We pass a list of 1 for the batch function, overhead is minimal compared to I/O
                    ph = _batch_phash64_cv2([frame])[0]
                    
                    kf_data[lbl]['idxs'].append(idx)
                    kf_data[lbl]['phashes'].append(ph)
                    kf_data[lbl]['feats'].append(idx_to_feat[idx]) # Reuse from Step 1

                if pb: pb.update(1)
                final_pass_count += 1
                update_sub_progress(final_pass_count, total_final_pass, "frames")

        cap.release()
        wait(writer_futures) # Finish writing
        writer_pool.shutdown()

        # --- Process Keyframes (Memory-Only, No Re-Read) ---
        if keyframes:
            from sklearn.cluster import DBSCAN as _DBSCAN2
            
            for cid, data in kf_data.items():
                if not data['idxs']: continue
                
                idxs = data['idxs']
                fm = np.stack(data['feats'])
                phashes = data['phashes']
                
                # DBSCAN on subset
                labels_sub = _DBSCAN2(
                    eps=float(kf_eps), min_samples=int(kf_min_samples), metric='cosine', n_jobs=-1
                ).fit_predict(fm)
                
                # Select Representative
                selected_kf_indices = []
                unique_subs = set(labels_sub)
                if -1 in unique_subs: unique_subs.remove(-1)
                
                for sub in sorted(unique_subs):
                    sub_indices = [i for i, x in enumerate(labels_sub) if x == sub]
                    chosen = []
                    
                    for iidx in sub_indices:
                        if not chosen:
                            chosen.append(iidx); continue
                        
                        cos_dists = []
                        hamm_fracs = []
                        
                        for j in chosen:
                            f1 = fm[iidx] / (np.linalg.norm(fm[iidx]) + 1e-8)
                            f2 = fm[j] / (np.linalg.norm(fm[j]) + 1e-8)
                            cos_dists.append(1.0 - float(np.dot(f1, f2)))
                            
                            xor = (phashes[iidx] ^ phashes[j]) & ((1 << 64) - 1)
                            hamm_fracs.append(bin(xor).count('1') / 64.0)
                        
                        min_cos = min(cos_dists) if cos_dists else 1.0
                        min_hamm = min(hamm_fracs) if hamm_fracs else 1.0
                        
                        different = (min_cos >= kf_eps) or (min_hamm >= kf_hamming_frac)
                        if kf_require_both:
                            different = (min_cos >= kf_eps) and (min_hamm >= kf_hamming_frac)
                        if different:
                            chosen.append(iidx)
                    
                    for ch in chosen:
                        selected_kf_indices.append(idxs[ch])
                
                # If nothing selected (noise only or empty), pick first
                if not selected_kf_indices and idxs:
                    selected_kf_indices.append(idxs[0])
                    
                # Mark in JSON and Copy Files
                kdir = os.path.join(clusters_root, str(cid), 'keyframes')
                if save_keyframes:
                    os.makedirs(kdir, exist_ok=True)
                
                for kf_idx in selected_kf_indices:
                    key_set.add((cid, kf_idx))
                    
                    if save_keyframes:
                        # If we saved clusters, copy from there. Else we need to re-read.
                        # Assumption: save_clusters is usually True if we want output. 
                        # If save_clusters is False, we unfortunately have to re-read just these keyframes.
                        # But usually saving keyframes implies having the images.
                        
                        if save_clusters:
                            src = os.path.join(clusters_root, str(cid), f"{kf_idx:08d}.png")
                            dst = os.path.join(kdir, f"{kf_idx:08d}.png")
                            try:
                                if os.path.exists(src):
                                    shutil.copy2(src, dst)
                            except: pass
                        else:
                            # Fallback: Read specific frame (rare case)
                            try:
                                # Re-open capture to fetch specific frame
                                tmp_cap = _cv.VideoCapture(video_file)
                                # Check if opened
                                if tmp_cap.isOpened():
                                    tmp_cap.set(_cv.CAP_PROP_POS_FRAMES, float(kf_idx))
                                    ret, kf_img = tmp_cap.read()
                                    if ret:
                                        _cv.imwrite(dst, kf_img)
                                    tmp_cap.release()
                            except Exception as e:
                                print(f"Warning: Failed to extract keyframe {kf_idx}: {e}") 

        # Final JSON Update
        frames_entries = []
        for idx_val, path_val, lbl in zip(selected_indices, selected_paths, labels):
            frames_entries.append({
                "index": int(idx_val),
                "frame_index": int(idx_val),
                "path": path_val,
                "cluster_number": int(lbl),
                "keyframe": (int(lbl), int(idx_val)) in key_set
            })

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"clusters": clusters_count, "frames": frames_entries}, f, ensure_ascii=False, indent=2)
            
        frame_data = frames_entries

    # Final cleanup of return data
    keyframes_only = [e for e in frame_data if e.get("keyframe")]
    if keyframes_only:
        frame_data = keyframes_only
    elif frame_data:
        clean = []
        seen = set()
        for e in frame_data:
            i = e.get('index', e.get('frame_index'))
            if i is not None and i not in seen:
                seen.add(i)
                clean.append({'index': int(i)})
        frame_data = clean

    return clusters_root, frame_data, report_path