"""Adaptive keyframe extraction for long-form video processing.

This module uses PySceneDetect's adaptive content detector to find scene
boundaries with minimal hand tuning, then selects representative frames per
scene to feed downstream analysis.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from config import SegmentsConfig as SegmentsConfigType

import cv2 as _cv

try:  # pragma: no cover - dependency guard
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector
    from scenedetect.frame_timecode import FrameTimecode
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scenedetect[opencv] is required for video.modules.segment. Install it via requirements."
    ) from exc

from .utils.logging import debug_print


@dataclass(frozen=True)
class SegmentConfig:
    include_last_frame: bool = True
    debug: bool = False
    target_detection_fps: Optional[float] = 12.0


@dataclass(frozen=True)
class SceneSummary:
    scene_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    representative_frames: List[int]


@dataclass(frozen=True)
class SegmentArtifacts:
    representative_indices: List[int]
    scenes: List[SceneSummary]
    sampled_indices: List[int]
    total_frames: int
    fps: float
    duration_seconds: float
    frame_skip: int


def segment_video(video_path: str, *, config: Optional[SegmentConfig] = None) -> SegmentArtifacts:
    cfg = config or SegmentConfig()

    total_frames, fps = _probe_video_stats(video_path)
    duration = (total_frames / fps) if (fps > 0.0 and total_frames > 0) else 0.0

    detection_start = time.perf_counter()
    scene_boundaries, frame_skip = _detect_scene_boundaries(
        video_path,
        cfg.debug,
        fps,
        cfg.target_detection_fps,
    )
    detection_time = time.perf_counter() - detection_start

    if not scene_boundaries:
        base_timecode = FrameTimecode(timecode=0, fps=fps if fps > 0 else 30.0)
        end_timecode = FrameTimecode(timecode=total_frames or 1, fps=base_timecode.framerate)
        scene_boundaries = [(base_timecode, end_timecode)]

    summaries = _summarize_scenes(scene_boundaries, fps)

    keyframes = sorted({frame for scene in summaries for frame in scene.representative_frames})

    if cfg.include_last_frame and total_frames > 0 and (total_frames - 1) not in keyframes:
        keyframes.append(total_frames - 1)

    keyframes.sort()

    debug_print(
        (
            f"[segment] keyframes={len(keyframes)} scenes={len(summaries)} "
            f"total_frames={total_frames} fps={fps:.3f} frame_skip={frame_skip} "
            f"detect_time={detection_time:.2f}s"
        ),
        debug=cfg.debug,
    )

    return SegmentArtifacts(
        representative_indices=keyframes,
        scenes=summaries,
        sampled_indices=keyframes.copy(),
        total_frames=total_frames,
        fps=fps,
        duration_seconds=duration,
        frame_skip=frame_skip,
    )


def _probe_video_stats(video_path: str) -> Tuple[int, float]:
    capture = _cv.VideoCapture(video_path)
    try:
        total_frames = int(capture.get(_cv.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(_cv.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()

    if fps <= 1e-6:
        fps = 30.0
    return total_frames, fps


def _compute_frame_skip(native_fps: float, target_fps: Optional[float]) -> int:
    if not target_fps or target_fps <= 0.0 or native_fps <= 0.0:
        return 0

    if native_fps <= target_fps:
        return 0

    ratio = native_fps / target_fps
    skip = max(0, math.ceil(ratio) - 1)
    return skip


def _detect_scene_boundaries(
    video_path: str,
    debug: bool,
    fps: float,
    target_detection_fps: Optional[float],
) -> Tuple[List[Tuple[FrameTimecode, FrameTimecode]], int]:
    logger = logging.getLogger("pyscenedetect")
    previous_level = logger.level
    logger.setLevel(logging.INFO if debug else logging.WARNING)

    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())

    video = open_video(video_path)
    frame_skip = _compute_frame_skip(fps, target_detection_fps)

    try:
        scene_manager.detect_scenes(
            video=video,
            frame_skip=frame_skip,
            show_progress=debug,
        )
        scene_list = scene_manager.get_scene_list()
    finally:
        logger.setLevel(previous_level)

    debug_print(
        f"[segment] detected {len(scene_list)} scenes (frame_skip={frame_skip})",
        debug=debug,
    )

    return scene_list, frame_skip


def _summarize_scenes(
    boundaries: Sequence[Tuple[FrameTimecode, FrameTimecode]],
    fps: float,
) -> List[SceneSummary]:
    durations = [
        max(end.get_seconds() - start.get_seconds(), 1.0 / max(fps, 1.0))
        for start, end in boundaries
    ]

    base_span = _derive_base_span(durations)
    summaries: List[SceneSummary] = []

    for scene_id, (start_tc, end_tc) in enumerate(boundaries):
        start_frame = start_tc.get_frames()
        end_frame = max(start_frame, end_tc.get_frames() - 1)
        duration = max(end_tc.get_seconds() - start_tc.get_seconds(), 0.0)

        frame_count = _decide_frame_count(duration, base_span)
        frame_count = min(frame_count, max(1, end_frame - start_frame + 1))

        frames = _distribute_frames(start_frame, end_frame, frame_count)

        summaries.append(
            SceneSummary(
                scene_id=scene_id,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_tc.get_seconds(),
                end_time=end_tc.get_seconds(),
                representative_frames=frames,
            )
        )

    return summaries


def _derive_base_span(durations: Sequence[float]) -> float:
    positive = [d for d in durations if d > 0.0]
    if not positive:
        return 1.0
    if len(positive) == 1:
        return positive[0]

    median = statistics.median(positive)
    mean = statistics.fmean(positive)
    return max(median, mean)


def _decide_frame_count(duration: float, base_span: float) -> int:
    if duration <= 0.0:
        return 1

    effective_span = base_span if base_span > 1e-6 else duration
    ratio = duration / effective_span
    frame_count = max(1, int(round(ratio)))

    return frame_count


def _distribute_frames(start: int, end: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if start >= end:
        return [start]

    if count == 1:
        return [int(round((start + end) / 2))]

    frames = [start]
    segments = count - 1
    step = (end - start) / max(segments, 1)
    for i in range(segments):
        frame = int(round(start + (i + 1) * step))
        frames.append(min(max(start, frame), end))

    frames = sorted(set(frames))
    if not frames:
        frames = [start]
    return frames


def _export_frames(
    video_path: str,
    artifacts: SegmentArtifacts,
    output_dirs: List[Tuple[Path, bool]],
    *,
    image_format: str,
) -> None:
    """Export frames in a single video pass to multiple output directories.

    Args:
        video_path: Path to the video file.
        artifacts: Segment artifacts with scene info.
        output_dirs: List of (output_dir, flat) tuples. If flat=True, all
            frames go into the directory directly; otherwise they are grouped
            by scene sub-folders.
        image_format: Image file extension (e.g. 'png').
    """
    ext = image_format.lower().lstrip(".") or "png"

    # Build a mapping: frame_idx -> list of (target_dir, scene_id) for each output config
    frame_targets: dict[int, list[tuple[Path, int]]] = {}
    for scene in artifacts.scenes:
        if not scene.representative_frames:
            continue
        for out_dir, flat in output_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            target_dir = out_dir if flat else out_dir / f"scene_{scene.scene_id:03d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in scene.representative_frames:
                frame_targets.setdefault(frame_idx, []).append((target_dir, scene.scene_id))

    if not frame_targets:
        return

    video = open_video(video_path)
    fps = artifacts.fps if artifacts.fps > 0.0 else getattr(video, "frame_rate", 30.0)

    # Process all frames in sorted order (single video pass)
    for frame_idx in sorted(frame_targets.keys()):
        timecode = FrameTimecode(frame_idx, fps)
        video.seek(timecode)
        frame = video.read()
        if frame is None:
            continue
        filename = f"frame_{frame_idx:08d}.{ext}"
        for target_dir, _ in frame_targets[frame_idx]:
            path = target_dir / filename
            _cv.imwrite(str(path), frame)


def handle(
    input_file: str,
    output_folder: str,
    config: "SegmentsConfigType | None" = None,
    *,
    debug: bool = False,
    save_frames: bool = False,
) -> Tuple[str, List[dict], str]:
    """Main entry point for segment extraction.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: SegmentsConfig instance or None for defaults.
        debug: Enable verbose debug output.
        save_frames: Whether to save extracted frames to disk.

    Returns:
        Tuple of (frames_folder, frame_metadata, segments_json_path).
    """
    return _extract(
        input_file,
        output_folder,
        target_detection_fps=config.target_detection_fps if config else 12.0,
        include_last_frame=config.include_last_frame if config else True,
        save_frames=save_frames,
        debug=debug,
    )


def _extract(
    video_path: str,
    output_dir: str,
    *,
    target_detection_fps: Optional[float] = 12.0,
    include_last_frame: bool = True,
    save_frames: bool = False,
    debug: bool = False,
) -> Tuple[str, List[dict], str]:
    """Detect video segments and optionally export representative frames.

    Returns a tuple of (frames_folder, frame_metadata, segments_json_path).
    """

    print("INFO: Detecting representative video segments")

    results_path =os.path.join(output_dir,"frames", "segments")

    output_path = Path(results_path)
    scenes_dir = output_path / "scenes"
    flat_dir = output_path / "keyframes"
    for directory in (output_path, scenes_dir, flat_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = SegmentConfig(
        include_last_frame=include_last_frame,
        debug=debug,
        target_detection_fps=target_detection_fps,
    )

    artifacts = segment_video(video_path, config=config)

    fps = artifacts.fps if artifacts.fps > 0.0 else None
    frames_metadata: List[dict] = []
    for scene in artifacts.scenes:
        for frame_idx in scene.representative_frames:
            time_seconds = (frame_idx / fps) if fps else None
            frames_metadata.append(
                {
                    "index": frame_idx,
                    "scene_id": scene.scene_id,
                    "pts_time": time_seconds,
                    "scene_start_time": scene.start_time,
                    "scene_end_time": scene.end_time,
                }
            )

    existing_indices = {item["index"] for item in frames_metadata}
    for frame_idx in artifacts.representative_indices:
        if frame_idx in existing_indices:
            continue
        matching_scene = next(
            (
                scene
                for scene in artifacts.scenes
                if scene.start_frame <= frame_idx <= scene.end_frame
            ),
            None,
        )
        time_seconds = (frame_idx / fps) if fps else None
        frames_metadata.append(
            {
                "index": frame_idx,
                "scene_id": matching_scene.scene_id if matching_scene else -1,
                "pts_time": time_seconds,
                "scene_start_time": matching_scene.start_time if matching_scene else None,
                "scene_end_time": matching_scene.end_time if matching_scene else None,
            }
        )

    frames_metadata.sort(key=lambda item: item["index"])

    segments_payload = {
        "video": str(Path(video_path).resolve()),
        "fps": artifacts.fps,
        "total_frames": artifacts.total_frames,
        "duration_seconds": artifacts.duration_seconds,
        "frame_skip": artifacts.frame_skip,
        "target_detection_fps": target_detection_fps,
        "include_last_frame": config.include_last_frame,
        "keyframes": artifacts.representative_indices,
        "frames": frames_metadata,
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "start_frame": scene.start_frame,
                "end_frame": scene.end_frame,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "frames": scene.representative_frames,
            }
            for scene in artifacts.scenes
        ],
    }

    segments_path = output_path / "segments.json"
    segments_path.write_text(json.dumps(segments_payload, indent=2), encoding="utf-8")
    debug_print(f"INFO: Wrote segments metadata to {segments_path}", debug=debug)

    if save_frames:
        debug_print("INFO: Exporting keyframes to disk", debug=debug)
        _export_frames(
            video_path,
            artifacts,
            [(scenes_dir, False), (flat_dir, True)],
            image_format="png",
        )
    else:
        debug_print("INFO: Skipping frame export (set save_frames=True to write images)", debug=debug)

    return str(flat_dir), frames_metadata, str(segments_path)


__all__ = ["handle"]
