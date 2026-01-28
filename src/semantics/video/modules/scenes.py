from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Tuple, TYPE_CHECKING

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import ScenesConfig

__all__ = ["handle"]


def _close_video(handle) -> None:
    if handle is None:
        return
    try:
        release = getattr(handle, "release", None)
        if callable(release):
            release()
            return
        close = getattr(handle, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _fallback_scene_bounds(video_handle) -> Tuple[FrameTimecode, FrameTimecode]:
    frame_rate = video_handle.frame_rate if getattr(video_handle, "frame_rate", None) else 30.0
    if getattr(video_handle, "frame_number", 0) and video_handle.frame_number > 0:
        frame_count = int(video_handle.frame_number)
    else:
        elapsed_seconds = getattr(video_handle, "position_ms", 0.0) / 1000.0
        base_frame = int(getattr(video_handle.base_timecode, "frame_num", 0))
        frame_count = int(base_frame + (elapsed_seconds * frame_rate))
    end_timecode = FrameTimecode("00:00:00.000", frame_rate) + frame_count
    return video_handle.base_timecode, end_timecode


def _build_ffmpeg_override(use_codec_copy: bool) -> str:
    base_flags = "-hide_banner -loglevel error"
    if use_codec_copy:
        return f"{base_flags} -c:v copy -c:a copy"
    audio_args = "-c:a aac -b:a 192k"
    return f"{base_flags} -map 0:v:0 -map 0:1 {audio_args}"


def handle(
    input_file: str,
    output_folder: str,
    config: "ScenesConfig | None" = None,
    *,
    debug: bool = False,
):
    """Main entry point for scene splitting.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: ScenesConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Dictionary with scenes data or None on failure.
    """
    return _split_scenes(
        input_file,
        output_folder,
        detector_type=config.detector_type if config else "content",
        threshold=config.threshold if config else 27.0,
        min_scene_len=config.min_scene_len if config else 15,
        use_codec_copy=config.use_codec_copy if config else False,
        debug=debug,
    )


def _split_scenes(
    input_file: str,
    output_folder: str,
    detector_type: str = "content",
    threshold: float = 27.0,
    min_scene_len: int = 15,
    use_codec_copy: bool = False,
    *,
    debug: bool = False,
):
    if not os.path.exists(input_file):
        print(f"Error: Input video file not found at {input_file}")
        return None

    pyscene_logger = logging.getLogger("pyscenedetect")
    pyscene_logger.setLevel(logging.INFO if debug else logging.ERROR)

    scenes_root = os.path.join(output_folder, "scenes")
    os.makedirs(scenes_root, exist_ok=True)
    results_path = os.path.join(scenes_root, "scenes.json")

    print("INFO: Detecting video scenes")

    video_handle = open_video(input_file)
    scene_manager = SceneManager()

    if detector_type == "content":
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        debug_print(
            f"Using ContentDetector (threshold={threshold}, min_scene_len={min_scene_len})",
            debug=debug,
        )
    elif detector_type == "threshold":
        scene_manager.add_detector(ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len))
        debug_print(
            f"Using ThresholdDetector (threshold={threshold}, min_scene_len={min_scene_len})",
            debug=debug,
        )
    else:
        print(f"Error: Unknown detector type '{detector_type}'.")
        _close_video(video_handle)
        return None

    start_time = time.time()
    debug_print("Running scene detection", debug=debug)
    with gray_debug_output(debug):
        scene_manager.detect_scenes(video=video_handle, show_progress=debug)
    scene_list = scene_manager.get_scene_list() or []
    debug_print(f"Detected {len(scene_list)} scene boundaries", debug=debug)

    if not scene_list:
        print("INFO: No scene cuts detected. Treating the entire video as a single scene.")
        scene_list = [
            _fallback_scene_bounds(video_handle),
        ]

    detection_elapsed = time.time() - start_time
    debug_print(f"Scene detection completed in {detection_elapsed:.2f} seconds", debug=debug)

    _close_video(video_handle)

    output_template = os.path.join(scenes_root, "scene-$SCENE_NUMBER.mp4")
    ffmpeg_override = _build_ffmpeg_override(use_codec_copy)
    debug_print(
        f"Splitting video into {len(scene_list)} scene clip(s) using override '{ffmpeg_override}'",
        debug=debug,
    )

    with gray_debug_output(debug):
        split_video_ffmpeg(
            input_video_path=input_file,
            scene_list=scene_list,
            output_file_template=output_template,
            output_dir=scenes_root,
            show_progress=debug,
            show_output=False,
            arg_override=ffmpeg_override,
        )

    results: List[dict] = []
    for index, (start_tc, end_tc) in enumerate(scene_list, start=1):
        clip_name = f"scene-{index:03d}.mp4"
        clip_path = os.path.abspath(os.path.join(scenes_root, clip_name))
        if not os.path.exists(clip_path):
            print(
                f"Warning: Expected clip file not found: {clip_path}. Splitting may have failed for scene {index}."
            )
        results.append(
            {
                "scene_number": index,
                "start_time_seconds": start_tc.get_seconds(),
                "end_time_seconds": end_tc.get_seconds(),
                "start_timecode": start_tc.get_timecode(),
                "end_timecode": end_tc.get_timecode(),
                "duration_seconds": end_tc.get_seconds() - start_tc.get_seconds(),
                "filepath": clip_path,
            }
        )

    debug_print(
        f"Prepared metadata for {len(results)} scene clip(s)",
        debug=debug,
    )

    try:
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump({"scenes": results}, handle, indent=4)
        debug_print(
            f"Scene analysis results saved to: {os.path.abspath(results_path)}",
            debug=debug,
        )
    except IOError as exc:
        print(f"Error saving JSON file to {results_path}: {exc}")
        return None

    total_elapsed = time.time() - start_time
    debug_print("-" * 30, debug=debug)
    debug_print(
        f"Scene processing completed in {total_elapsed:.2f} seconds",
        debug=debug,
    )
    debug_print("INFO: Scene detection completed", debug=debug)

    return {"scenes": results}
