"""Action recognition module for video using transformer-based models.

Recognizes human actions in video clips using models like VideoMAE or TimeSformer.
Uses motion-based activity scanning to intelligently select clips for analysis.

Key Features:
- Motion-based pre-scan to identify active video segments (fast, CPU-only)
- Adaptive clip selection focusing on areas with actual activity
- Clip export for detected actions
- Self-contained (no external frame selection dependencies)
"""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .utils.logging import debug_print, gray_debug_output, info_print, update_sub_progress

# Import PyAV globally to ensure it's available for the persistent container
try:
    import av
except ImportError:
    raise ImportError("PyAV is required. Install it via: pip install av")

if TYPE_CHECKING:
    from config import ActionsConfig

__all__ = ["handle"]

# Hardcoded constants (rarely need configuration)
MIN_CLIPS = 3  # Minimum clips to process even if no activity detected
MAX_CLIP_EXPORT_WORKERS = 4  # Parallel FFmpeg processes for clip export

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Config Defaults Helper
# =============================================================================


def _get_actions_defaults() -> dict:
    """Get default values from ActionsConfig to avoid circular imports."""
    try:
        from config import ActionsConfig
        cfg = ActionsConfig()
        return {
            "model": cfg.model,
            "num_frames": cfg.num_frames,
            "frame_sample_rate": cfg.frame_sample_rate,
            "conf_threshold": cfg.conf_threshold,
            "top_k": cfg.top_k,
            "batch_size": cfg.batch_size,
            "save_clips": cfg.save_clips,
            "padding": cfg.padding,
            "clip_overlap": cfg.clip_overlap,
            "scan_fps": cfg.scan_fps,
            "motion_threshold": cfg.motion_threshold,
            "min_activity_duration": cfg.min_activity_duration,
            "merge_gap": cfg.merge_gap,
        }
    except Exception:
        # Fallback defaults if config import fails
        return {
            "model": "MCG-NJU/videomae-base-finetuned-kinetics",
            "num_frames": 16,
            "frame_sample_rate": 8,
            "conf_threshold": 0.2,
            "top_k": 3,
            "batch_size": 8,
            "save_clips": True,
            "padding": 1.0,
            "clip_overlap": 0.25, # Reduced from 0.5 to reduce redundancy
            "scan_fps": 2.0,
            "motion_threshold": 0.02,
            "min_activity_duration": 0.5,
            "merge_gap": 1.0,
        }


# =============================================================================
# Internal Settings Dataclass
# =============================================================================


@dataclass
class _ActionsSettings:
    """Internal settings container for actions module."""

    model: str = field(default_factory=lambda: _get_actions_defaults()["model"])
    num_frames: int = field(default_factory=lambda: _get_actions_defaults()["num_frames"])
    frame_sample_rate: int = field(default_factory=lambda: _get_actions_defaults()["frame_sample_rate"])
    conf_threshold: float = field(default_factory=lambda: _get_actions_defaults()["conf_threshold"])
    top_k: int = field(default_factory=lambda: _get_actions_defaults()["top_k"])
    batch_size: int = field(default_factory=lambda: _get_actions_defaults()["batch_size"])
    save_clips: bool = field(default_factory=lambda: _get_actions_defaults()["save_clips"])
    padding: float = field(default_factory=lambda: _get_actions_defaults()["padding"])
    clip_overlap: float = field(default_factory=lambda: _get_actions_defaults()["clip_overlap"])
    scan_fps: float = field(default_factory=lambda: _get_actions_defaults()["scan_fps"])
    motion_threshold: float = field(default_factory=lambda: _get_actions_defaults()["motion_threshold"])
    min_activity_duration: float = field(default_factory=lambda: _get_actions_defaults()["min_activity_duration"])
    merge_gap: float = field(default_factory=lambda: _get_actions_defaults()["merge_gap"])


@dataclass
class _ActivitySegment:
    """Represents a detected activity region in the video."""
    start_frame: int
    end_frame: int
    peak_frame: int  # Frame with highest activity
    motion_score: float


# =============================================================================
# Model Cache (Singleton Pattern)
# =============================================================================


_ACTION_MODEL: Optional[Any] = None
_ACTION_PROCESSOR: Optional[Any] = None
_ACTION_MODEL_NAME: Optional[str] = None
_ACTION_DEVICE: Optional[torch.device] = None


class _NumpyEncoder(json.JSONEncoder):
    """Internal JSON encoder for numpy types."""

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


# =============================================================================
# Motion Detection Functions
# =============================================================================


def _compute_motion_score(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    threshold: int = 25,
) -> float:
    """Fast frame differencing for motion detection."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    motion_ratio = np.count_nonzero(thresh) / thresh.size
    return float(motion_ratio)


def _fast_activity_scan(
    video_path: str,
    sample_fps: float = 2.0,
    motion_threshold: float = 0.02,
    min_duration: float = 0.5,
    merge_gap: float = 1.0,
    segment_padding: float = 0.0,
    *,
    debug: bool = False,
) -> Tuple[List[_ActivitySegment], Dict[str, Any]]:
    """Fast first-pass scan to identify activity regions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate sample interval
    sample_interval = max(1, int(fps / sample_fps))
    
    # Resize for speed (width=320 is sufficient for motion detection)
    target_width = 320
    
    prev_gray = None
    frame_scores: List[Tuple[int, float]] = []  # (frame_idx, motion_score)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process sampled frames
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        
        # Resize for speed
        h, w = frame.shape[:2]
        scale = target_width / w
        small_frame = cv2.resize(frame, (target_width, int(h * scale)))
        
        # Convert to grayscale and blur
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is not None:
            motion_score = _compute_motion_score(prev_gray, gray)
            frame_scores.append((frame_idx, motion_score))
        
        prev_gray = gray
        frame_idx += 1
    
    cap.release()
    
    # Extract activity segments with padding
    segments = _extract_activity_segments(
        frame_scores=frame_scores,
        fps=fps,
        motion_threshold=motion_threshold,
        min_duration=min_duration,
        merge_gap=merge_gap,
        segment_padding=segment_padding,
        total_frames=total_frames,
    )
    
    # Calculate coverage stats
    total_activity_frames = sum(s.end_frame - s.start_frame for s in segments)
    coverage_ratio = total_activity_frames / total_frames if total_frames > 0 else 0
    
    metadata = {
        "video_fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "frames_scanned": len(frame_scores),
        "sample_interval": sample_interval,
        "activity_segments": len(segments),
        "coverage_ratio": round(coverage_ratio, 3),
    }
    
    if debug:
        debug_print(
            f"Activity scan: {len(segments)} segments, "
            f"{coverage_ratio:.1%} coverage",
            debug=debug,
        )
    
    return segments, metadata


def _extract_activity_segments(
    frame_scores: List[Tuple[int, float]],
    fps: float,
    motion_threshold: float,
    min_duration: float,
    merge_gap: float,
    segment_padding: float = 0.0,
    total_frames: int = 0,
) -> List[_ActivitySegment]:
    """Extract continuous activity segments from frame scores."""
    if not frame_scores:
        return []
    
    # Mark frames as active
    active_frames: List[Tuple[int, float]] = [
        (frame_idx, motion)
        for frame_idx, motion in frame_scores
        if motion >= motion_threshold
    ]
    
    if not active_frames:
        return []
    
    # Group into continuous segments
    gap_frames = int(merge_gap * fps)
    min_frames = int(min_duration * fps)
    padding_frames = int(segment_padding * fps)
    
    segments: List[_ActivitySegment] = []
    current_start = active_frames[0][0]
    current_end = active_frames[0][0]
    peak_frame = active_frames[0][0]
    peak_motion = active_frames[0][1]
    
    for frame_idx, motion in active_frames[1:]:
        if frame_idx - current_end <= gap_frames:
            # Extend current segment
            current_end = frame_idx
            if motion > peak_motion:
                peak_frame = frame_idx
                peak_motion = motion
        else:
            # Start new segment (save current if long enough)
            if current_end - current_start >= min_frames:
                segments.append(_ActivitySegment(
                    start_frame=current_start,
                    end_frame=current_end,
                    peak_frame=peak_frame,
                    motion_score=peak_motion,
                ))
            current_start = frame_idx
            current_end = frame_idx
            peak_frame = frame_idx
            peak_motion = motion
    
    # Don't forget the last segment
    if current_end - current_start >= min_frames:
        segments.append(_ActivitySegment(
            start_frame=current_start,
            end_frame=current_end,
            peak_frame=peak_frame,
            motion_score=peak_motion,
        ))
    
    # Apply padding to capture full action context
    if padding_frames > 0 and total_frames > 0:
        for segment in segments:
            segment.start_frame = max(0, segment.start_frame - padding_frames)
            segment.end_frame = min(total_frames, segment.end_frame + padding_frames)
    
    return segments


def _generate_activity_clips(
    segments: List[_ActivitySegment],
    total_frames: int,
    clip_span: int,
    clip_overlap: float = 0.25,
) -> List[int]:
    """Generate clip start positions from activity segments.
    
    Optimized to reduce redundant clips:
    1. Short segments (< 2 * clip_span): Gets 1 clip centered on the peak action.
    2. Long segments: Uses striding, but with strictly controlled overlap.
    """
    clip_starts: List[int] = []
    
    # Calculate stride: If overlap is 0.25, stride is 75% of span
    stride = max(1, int(clip_span * (1.0 - clip_overlap)))
    
    if segments:
        for segment in segments:
            seg_start = segment.start_frame
            seg_end = segment.end_frame
            segment_duration = seg_end - seg_start
            
            # OPTIMIZATION: If the segment is small/moderate, prioritize the PEAK action
            # instead of blindly sliding from the start.
            if segment_duration <= (clip_span * 1.5):
                # Center the clip on the peak of the motion (if available), 
                # otherwise center of segment
                center = segment.peak_frame if segment.peak_frame >= seg_start else (seg_start + seg_end) // 2
                
                clip_start = max(0, center - clip_span // 2)
                clip_start = min(clip_start, max(0, total_frames - clip_span))
                clip_starts.append(clip_start)
            else:
                # Long segment: Scan it
                # Ensure we cover the peak frame
                pos = seg_start
                while pos + clip_span <= seg_end:
                    clip_start = min(pos, max(0, total_frames - clip_span))
                    clip_starts.append(clip_start)
                    pos += stride
                
                # Ensure the end is covered
                end_clip = max(0, seg_end - clip_span)
                end_clip = min(end_clip, max(0, total_frames - clip_span))
                clip_starts.append(end_clip)
    
    # Ensure minimum clips via uniform sampling if nothing detected
    if len(clip_starts) < MIN_CLIPS:
        if total_frames <= clip_span:
            clip_starts = [0]
        else:
            available_space = total_frames - clip_span
            for i in range(MIN_CLIPS):
                pos = int(i * available_space / (MIN_CLIPS - 1)) if MIN_CLIPS > 1 else 0
                clip_starts.append(pos)
    
    # Always include start and end of video for context
    clip_starts.append(0)
    if total_frames > clip_span:
        clip_starts.append(total_frames - clip_span)
    
    # Deduplicate and sort
    return sorted(set(clip_starts))


# =============================================================================
# Model and Video Processing Functions
# =============================================================================


def _ensure_action_model(
    model_name: str,
    *,
    debug: bool = False,
) -> Tuple[Optional[Any], Optional[Any], Optional[torch.device], int]:
    """Load and cache the action recognition model."""
    global _ACTION_MODEL, _ACTION_PROCESSOR, _ACTION_MODEL_NAME, _ACTION_DEVICE

    if (
        _ACTION_MODEL is not None
        and _ACTION_PROCESSOR is not None
        and _ACTION_MODEL_NAME == model_name
        and _ACTION_DEVICE is not None
    ):
        expected_frames = getattr(_ACTION_MODEL.config, "num_frames", 8)
        return _ACTION_MODEL, _ACTION_PROCESSOR, _ACTION_DEVICE, expected_frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        debug_print(f"INFO: Using GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)", debug=debug)
    else:
        debug_print("WARNING: CUDA not available, using CPU (inference will be slow)", debug=debug)
    
    debug_print(f"Loading action recognition model: {model_name} on {device}", debug=debug)

    try:
        with gray_debug_output(debug):
            from transformers import AutoImageProcessor, AutoModelForVideoClassification

            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForVideoClassification.from_pretrained(model_name)
            model.to(device)
            model.eval()
    except Exception as exc:
        print(f"ERROR: Failed to load action recognition model '{model_name}': {exc}")
        return None, None, None, 8

    _ACTION_MODEL = model
    _ACTION_PROCESSOR = processor
    _ACTION_MODEL_NAME = model_name
    _ACTION_DEVICE = device

    expected_frames = getattr(model.config, "num_frames", 8)
    debug_print(f"Model expects {expected_frames} frames per clip", debug=debug)

    return model, processor, device, expected_frames


def _get_video_info_pyav(video_path: str) -> Tuple[float, int, int, int]:
    """Get video info using PyAV."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frame_count = stream.frames if stream.frames > 0 else 0
        
        if frame_count == 0 and stream.duration:
            frame_count = int(stream.duration * stream.time_base * fps)
        
        width = stream.width
        height = stream.height
        
        # If frame count still 0, count manually
        if frame_count == 0:
            frame_count = sum(1 for _ in container.decode(video=0))
        
        container.close()
        return fps, frame_count, width, height
    except Exception as e:
        print(f"ERROR: PyAV info error: {e}")
        return 30.0, 0, 0, 0


def _sample_frame_indices(
    num_frames: int,
    frame_sample_rate: int,
    total_frames: int,
    start_idx: int = 0,
) -> List[int]:
    """Sample frame indices using the official HuggingFace strategy."""
    clip_span = num_frames * frame_sample_rate
    
    if start_idx + clip_span > total_frames:
        end_idx = total_frames
        start_idx = max(0, end_idx - clip_span)
    else:
        end_idx = start_idx + clip_span
    
    if end_idx - start_idx < num_frames:
        indices = np.linspace(start_idx, max(start_idx, end_idx - 1), num=num_frames)
    else:
        indices = np.linspace(start_idx, end_idx - 1, num=num_frames)
    
    indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
    return indices.tolist()


def _extract_frames_pyav_fast(
    container: av.container.InputContainer,
    indices: List[int],
) -> List[np.ndarray]:
    """Extract frames using an EXISTING PyAV container (no reopen)."""
    frames_dict: Dict[int, np.ndarray] = {}
    
    if not indices:
        return []
    
    indices_set = set(indices)
    start_index = min(indices)
    end_index = max(indices)
    
    # Seek to the closest keyframe before the start index
    # Note: Seeking can be imprecise, so we seek a bit earlier
    seek_target = max(0, start_index - 30) 
    
    # Get the time base to convert frame number to time
    stream = container.streams.video[0]
    time_base = stream.time_base
    fps = stream.average_rate
    
    # Calculate timestamp
    if fps and time_base:
        pts = int(seek_target / fps / time_base)
        container.seek(pts, stream=stream)
    else:
        # Fallback to pure frame seek if possible (less reliable in some containers)
        container.seek(0)

    # Decode until we get what we need
    # This is much faster than reopening the file, but still requires decoding
    # intermediate frames.
    
    # Optimization: If the container is already near the target, we don't need to seek.
    # However, PyAV state management is complex, so explicit seek is safer for random access.
    
    current_idx = -1
    
    try:
        for packet in container.demux(video=0):
            for frame in packet.decode():
                # Approximation of frame index if not available directly
                # Ideally calculate from PTS, but sequential counting after seek is safer
                # assuming we seeked roughly correctly. 
                # For robust index matching, we rely on PTS, but for speed here we assume
                # the seek got us close.
                
                # To be precise, we need to calculate exact frame index from PTS
                pts = frame.pts
                if pts is None:
                    continue
                    
                # Calculate approximate index
                idx = int(frame.time * float(fps))
                
                if idx > end_index + 5: # Small buffer
                    return _assemble_frames(frames_dict, indices)
                
                if idx in indices_set:
                    frames_dict[idx] = frame.to_ndarray(format="rgb24")
    except Exception:
        pass # End of file or decode error
        
    return _assemble_frames(frames_dict, indices)

def _assemble_frames(frames_dict: Dict[int, np.ndarray], indices: List[int]) -> List[np.ndarray]:
    """Helper to assemble the final list from the dict."""
    frames = []
    if not frames_dict:
        # Fallback: return black frames if totally failed
        return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in indices]

    for idx in indices:
        if idx in frames_dict:
            frames.append(frames_dict[idx])
        else:
            # Nearest neighbor interpolation for missing frames
            nearest = min(frames_dict.keys(), key=lambda x: abs(x - idx))
            frames.append(frames_dict[nearest].copy())
    return frames


def _process_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    batch_clips: List[List[np.ndarray]],
    *,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Process a batch of video clips through the action recognition model."""
    if not batch_clips:
        return []
    
    results: List[Dict[str, Any]] = []
    
    try:
        inputs = processor(images=batch_clips, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        for idx in range(len(batch_clips)):
            clip_probs = probs[idx].cpu().numpy()
            results.append({
                "probabilities": clip_probs,
                "labels": model.config.id2label,
            })
    except Exception as exc:
        debug_print(f"WARNING: Batch processing failed: {exc}", debug=debug)
        for _ in batch_clips:
            results.append({"probabilities": None, "labels": {}})
    
    return results


# =============================================================================
# Clip Export Functions
# =============================================================================


def _save_video_clip(
    source_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    duration: float,
) -> bool:
    """Save a video clip from source using FFmpeg with fast seeking."""
    import subprocess
    
    try:
        # Clamp times to valid range
        start_time = max(0, start_time)
        end_time = min(end_time, duration)
        clip_duration = end_time - start_time
        
        if clip_duration <= 0:
            return False
        
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_time),  # Seek BEFORE input (fast keyframe seek)
            "-i", source_path,
            "-t", str(clip_duration),  # Duration
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-loglevel", "error",
            output_path,
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        return result.returncode == 0
        
    except Exception as exc:
        print(f"WARNING: Failed to save clip {output_path}: {exc}")
        return False


# =============================================================================
# Action Merging and Post-Processing
# =============================================================================


def _merge_adjacent_actions(
    actions: List[Dict[str, Any]],
    max_gap: float = 0.5,
) -> List[Dict[str, Any]]:
    """Merge adjacent action segments with the same label."""
    if not actions:
        return []
    
    # Sort by start time, then by confidence (higher first)
    sorted_actions = sorted(actions, key=lambda x: (x["start_time"], -x["confidence"]))
    
    merged: List[Dict[str, Any]] = []
    current = sorted_actions[0].copy()
    
    for action in sorted_actions[1:]:
        if (
            action["label"] == current["label"]
            and action["start_time"] - current["end_time"] <= max_gap
        ):
            current["end_time"] = max(current["end_time"], action["end_time"])
            current["confidence"] = max(current["confidence"], action["confidence"])
        else:
            merged.append(current)
            current = action.copy()
    
    merged.append(current)
    return merged


# =============================================================================
# Main Recognition Function
# =============================================================================


def _recognize_actions(
    video_path: str,
    output_folder: str,
    settings: _ActionsSettings,
    *,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Internal implementation for action recognition."""
    # Load model and get expected frame count
    model, processor, device, model_num_frames = _ensure_action_model(
        settings.model, debug=debug
    )
    if model is None or processor is None or device is None:
        print("ERROR: Failed to load action recognition model")
        return None
    
    # Use model's expected num_frames if different from settings
    num_frames = settings.num_frames
    if num_frames != model_num_frames:
        debug_print(
            f"Adjusting num_frames from {num_frames} to {model_num_frames} (model requirement)",
            debug=debug,
        )
        num_frames = model_num_frames
    
    # Get video info
    try:
        fps, frame_count, width, height = _get_video_info_pyav(video_path)
    except Exception as exc:
        print(f"ERROR: Failed to probe video with PyAV: {exc}")
        return None
    
    duration = frame_count / fps if fps > 0 else 0
    debug_print(f"Video: {duration:.2f}s @ {fps:.2f} FPS, {frame_count} frames", debug=debug)
    
    clip_span = num_frames * settings.frame_sample_rate
    debug_print(f"Clip span: {clip_span} frames ({clip_span/fps:.2f}s)", debug=debug)
    
    # Run activity pre-scan
    debug_print("Scanning video for activity regions", debug=debug)
    try:
        segments, scan_meta = _fast_activity_scan(
            video_path,
            sample_fps=settings.scan_fps,
            motion_threshold=settings.motion_threshold,
            min_duration=settings.min_activity_duration,
            merge_gap=settings.merge_gap,
            segment_padding=settings.padding,
            debug=debug,
        )
    except Exception as exc:
        debug_print(f"WARNING: Activity scan failed, using uniform sampling: {exc}", debug=debug)
        segments = []
        scan_meta = {}
    
    # Generate clip positions with optimized sparse sampling
    clip_starts = _generate_activity_clips(
        segments=segments,
        total_frames=frame_count,
        clip_span=clip_span,
        clip_overlap=settings.clip_overlap,
    )
    
    debug_print(
        f"Generated {len(clip_starts)} clips from {len(segments)} activity segments",
        debug=debug,
    )
    
    # Process clips in batches
    all_predictions: List[Dict[str, Any]] = []
    total_batches = len(range(0, len(clip_starts), settings.batch_size))
    
    iterator = range(0, len(clip_starts), settings.batch_size)
    if debug:
        try:
            iterator = tqdm(iterator, desc="Actions", unit="batch", colour="#888888")
        except TypeError:
            iterator = tqdm(iterator, desc="Actions", unit="batch")
    else:
        debug_print(f"INFO: Processing {len(clip_starts)} video clips for action recognition", debug=debug)
    
    # OPTIMIZATION: Open PyAV container ONCE for the whole loop
    try:
        container = av.open(video_path)
        batch_count = 0
        
        for batch_idx in iterator:
            batch_start_indices = clip_starts[batch_idx:batch_idx + settings.batch_size]
            batch_clips: List[List[np.ndarray]] = []
            batch_times: List[Tuple[int, float, float]] = []  # (start_frame, start_time, end_time)
            
            for start_frame in batch_start_indices:
                indices = _sample_frame_indices(
                    num_frames=num_frames,
                    frame_sample_rate=settings.frame_sample_rate,
                    total_frames=frame_count,
                    start_idx=start_frame,
                )
                
                # Use the open container
                frames = _extract_frames_pyav_fast(container, indices)
                
                if frames and len(frames) == num_frames:
                    batch_clips.append(frames)
                    start_time = start_frame / fps
                    end_time = min((start_frame + clip_span) / fps, duration)
                    batch_times.append((start_frame, start_time, end_time))
            
            batch_count += 1
            update_sub_progress(batch_count, total_batches, "batches")
            
            if not batch_clips:
                continue
            
            batch_results = _process_batch(model, processor, device, batch_clips, debug=debug)
            
            for idx, result in enumerate(batch_results):
                probs = result.get("probabilities")
                labels = result.get("labels", {})
                
                if probs is None:
                    continue
                
                start_frame, start_time, end_time = batch_times[idx]
                
                # Get top-k predictions above threshold
                top_indices = np.argsort(probs)[::-1][:settings.top_k]
                
                for class_idx in top_indices:
                    confidence = float(probs[class_idx])
                    if confidence < settings.conf_threshold:
                        continue
                    
                    label = labels.get(class_idx, f"class_{class_idx}")
                    
                    all_predictions.append({
                        "start_frame": start_frame,
                        "end_frame": min(start_frame + clip_span, frame_count),
                        "start_time": round(start_time, 3),
                        "end_time": round(end_time, 3),
                        "label": label,
                        "confidence": round(confidence, 4),
                        "class_id": int(class_idx),
                    })
    except Exception as e:
        print(f"ERROR during batch processing: {e}")
    finally:
        # Close the container when done
        try:
            container.close()
        except Exception:
            pass
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Merge adjacent segments with same label
    merged_actions = _merge_adjacent_actions(all_predictions)
    
    # Sort by start_time (ascending)
    merged_actions.sort(key=lambda x: x["start_time"])
    
    # Assign clip IDs and export clips if enabled
    actions_dir = os.path.join(output_folder, "actions")
    clips_dir = os.path.join(actions_dir, "clips")
    os.makedirs(actions_dir, exist_ok=True)
    
    if settings.save_clips and merged_actions:
        os.makedirs(clips_dir, exist_ok=True)
        debug_print(f"INFO: Saving {len(merged_actions)} action clips", debug=debug)
        
        # Prepare clip export tasks
        clip_tasks = []
        for idx, action in enumerate(merged_actions):
            clip_id = idx + 1
            clip_filename = f"{clip_id:08d}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)
            
            # Apply padding for context before/after the action
            padded_start = max(0, action["start_time"] - settings.padding)
            padded_end = min(duration, action["end_time"] + settings.padding)
            
            clip_tasks.append({
                "clip_id": clip_id,
                "clip_path": clip_path,
                "start_time": padded_start,
                "end_time": padded_end,
                "action_idx": idx,
            })
        
        # Export clips in parallel
        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_CLIP_EXPORT_WORKERS) as executor:
            future_to_task = {
                executor.submit(
                    _save_video_clip,
                    video_path,
                    task["clip_path"],
                    task["start_time"],
                    task["end_time"],
                    duration,
                ): task
                for task in clip_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    success = future.result()
                    results_map[task["action_idx"]] = (task["clip_id"], task["clip_path"] if success else None)
                except Exception:
                    results_map[task["action_idx"]] = (task["clip_id"], None)
        
        # Update actions with clip info
        for idx, action in enumerate(merged_actions):
            clip_id, clip_path = results_map.get(idx, (idx + 1, None))
            action["clip_id"] = clip_id
            if clip_path:
                action["clip_path"] = os.path.relpath(clip_path, output_folder)
            else:
                action["clip_path"] = None
    else:
        # No clips to save, just assign IDs
        for idx, action in enumerate(merged_actions):
            action["clip_id"] = idx + 1
            action["clip_path"] = None
    
    # Clean up internal fields not needed in output
    for action in merged_actions:
        action.pop("start_frame", None)
        action.pop("end_frame", None)
    
    # Build results
    results = {
        "video_path": os.path.abspath(video_path),
        "duration": round(duration, 3),
        "fps": round(fps, 2),
        "resolution": {"width": width, "height": height},
        "model": settings.model,
        "settings": {
            "num_frames": num_frames,
            "frame_sample_rate": settings.frame_sample_rate,
            "clip_span_seconds": round(clip_span / fps, 3) if fps > 0 else 0,
            "conf_threshold": settings.conf_threshold,
            "top_k": settings.top_k,
            "clip_overlap": settings.clip_overlap,
            "padding": settings.padding,
            "clips_processed": len(clip_starts),
            "activity_segments": len(segments),
            "save_clips": settings.save_clips,
        },
        "actions": merged_actions,
        "action_count": len(merged_actions),
        "unique_labels": list(set(a["label"] for a in merged_actions)),
    }
    
    # Save results
    output_path = os.path.join(actions_dir, "actions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)
    
    debug_print(f"INFO: Detected {len(merged_actions)} action segments, {len(results['unique_labels'])} unique labels", debug=debug)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def handle(
    input_file: str,
    output_folder: str,
    config: "ActionsConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Main entry point for action recognition."""
    info_print("Recognizing actions in video")

    # Extract config values
    if config:
        settings = _ActionsSettings(
            model=config.model,
            num_frames=config.num_frames,
            frame_sample_rate=config.frame_sample_rate,
            conf_threshold=config.conf_threshold,
            top_k=config.top_k,
            batch_size=config.batch_size,
            save_clips=config.save_clips,
            padding=config.padding,
            clip_overlap=config.clip_overlap,
            scan_fps=config.scan_fps,
            motion_threshold=config.motion_threshold,
            min_activity_duration=config.min_activity_duration,
            merge_gap=config.merge_gap,
        )
    else:
        settings = _ActionsSettings()

    debug_print(f"Model: {settings.model}", debug=debug)
    debug_print(f"Frames: {settings.num_frames}, Sample rate: {settings.frame_sample_rate}", debug=debug)
    debug_print(f"Motion threshold: {settings.motion_threshold}, Save clips: {settings.save_clips}", debug=debug)
    debug_print(f"Clip overlap: {settings.clip_overlap}, Padding: {settings.padding}s", debug=debug)

    return _recognize_actions(input_file, output_folder, settings, debug=debug)