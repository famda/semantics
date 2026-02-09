from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING

from global_helpers import VIDEO_FILE_TYPES
from .utils.logging import gray_debug_output, info_print

if TYPE_CHECKING:
    from config import FramesConfig

__all__ = ["handle"]

env = os.environ.copy()
env["AV_LOG_FORCE_NOCOLOR"] = "1"

MIN_CHUNK_DURATION_SECONDS = 120.0


def _coerce_frame_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _resolve_frame_index(entry: dict, fallback: int) -> int:
    if not isinstance(entry, dict):
        return fallback

    candidate_keys = (
        "index",
        "frame",
        "frame_index",
        "frame_number",
        "coded_picture_number",
        "display_picture_number",
        "pkt_pts",
        "pkt_dts",
        "best_effort_timestamp",
    )

    for key in candidate_keys:
        if key in entry:
            number = _coerce_frame_number(entry.get(key))
            if number is not None:
                return number

    return fallback


def _run_command(command, *, debug: bool, stdout_path: Optional[str] = None) -> None:
    if debug:
        if stdout_path is None:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines: List[str] = []
            try:
                assert process.stdout is not None
                with gray_debug_output(True):
                    for line in process.stdout:
                        output_lines.append(line)
                        if line:
                            print(line.rstrip())
            finally:
                if process.stdout is not None:
                    process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command, output="".join(output_lines))
            return

        with open(stdout_path, "w", encoding="utf-8") as file_handle:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=file_handle,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            try:
                assert process.stderr is not None
                with gray_debug_output(True):
                    for line in process.stderr:
                        if line:
                            print(line.rstrip())
            finally:
                if process.stderr is not None:
                    process.stderr.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        return

    if stdout_path is not None:
        with open(stdout_path, "w", encoding="utf-8") as f:
            subprocess.run(
                command,
                check=True,
                env=env,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
            )
    else:
        subprocess.run(
            command,
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def _probe_video_info(path: str) -> Tuple[float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,r_frame_rate",
        "-of",
        "json",
        path,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return 0.0, 0.0

    if completed.returncode != 0 or not completed.stdout:
        return 0.0, 0.0

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return 0.0, 0.0

    streams = payload.get("streams") or []
    if not streams:
        return 0.0, 0.0

    stream = streams[0]
    duration_value = stream.get("duration")
    try:
        duration = float(duration_value) if duration_value is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    frame_rate_raw = stream.get("r_frame_rate") or "0/0"
    fps = 0.0
    if isinstance(frame_rate_raw, str) and "/" in frame_rate_raw:
        numerator, denominator = frame_rate_raw.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
            if den > 0:
                fps = num / den
        except (TypeError, ValueError):
            fps = 0.0

    return duration, fps


def _resolve_worker_count(parallel_jobs: Optional[int], debug: bool) -> int:
    if debug:
        return 1

    if parallel_jobs and parallel_jobs > 1:
        return int(parallel_jobs)

    env_value = os.environ.get("VIDEO_FRAME_EXTRACTION_JOBS")
    if env_value:
        try:
            env_jobs = int(env_value)
            if env_jobs > 0:
                return env_jobs
        except ValueError:
            pass

    cpu_total = os.cpu_count() or 1
    return max(1, min(4, max(1, cpu_total // 2)))


def _plan_chunks(duration: float, max_workers: int) -> List[Tuple[float, Optional[float]]]:
    if duration <= 0 or max_workers <= 1:
        return [(0.0, None)]

    chunk_length = max(MIN_CHUNK_DURATION_SECONDS, duration / max_workers)
    if chunk_length >= duration:
        return [(0.0, None)]

    chunks: List[Tuple[float, Optional[float]]] = []
    start = 0.0
    while start < duration and len(chunks) < max_workers:
        end = min(duration, start + chunk_length)
        length = end - start
        if end >= duration:
            chunks.append((start, None))
            break
        chunks.append((start, length))
        start = end

    if not chunks:
        chunks.append((0.0, None))

    return chunks


def _build_ffmpeg_command(
    source_path: str,
    output_pattern: str,
    *,
    start: Optional[float],
    duration: Optional[float],
    target_fps: Optional[float],
) -> List[str]:
    command: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        "16",
    ]

    if start and start > 0:
        command.extend(["-ss", f"{start:.6f}"])

    command.extend(["-i", source_path])

    if duration and duration > 0:
        command.extend(["-t", f"{duration:.6f}"])

    command.extend(["-map", "0:v:0"])

    filters: List[str] = []
    if target_fps and target_fps > 0:
        filters.append(f"fps={target_fps}")

    if filters:
        command.extend(["-vf", ",".join(filters)])

    command.extend([
        "-vsync",
        "vfr",
        "-start_number",
        "0",
        output_pattern,
    ])

    return command


def _extract_chunk(
    source_path: str,
    chunk_dir: str,
    *,
    start: Optional[float],
    duration: Optional[float],
    target_fps: Optional[float],
    debug: bool,
) -> None:
    output_pattern = os.path.join(chunk_dir, "%08d.png")
    command = _build_ffmpeg_command(
        source_path,
        output_pattern,
        start=start,
        duration=duration,
        target_fps=target_fps,
    )
    _run_command(command, debug=debug)


def _consolidate_chunk_outputs(
    chunk_records: Sequence[Tuple[float, str]],
    final_output_folder: str,
) -> None:
    next_index = 0
    for _, chunk_dir in sorted(chunk_records, key=lambda entry: entry[0]):
        image_paths = sorted(glob.glob(os.path.join(chunk_dir, "*.png")))
        for image_path in image_paths:
            destination = os.path.join(final_output_folder, f"{next_index:08d}.png")
            shutil.move(image_path, destination)
            next_index += 1
        shutil.rmtree(chunk_dir, ignore_errors=True)


def _extract_frames_from_video(
    source_path: str,
    output_folder: str,
    *,
    target_fps: Optional[float],
    parallel_jobs: Optional[int],
    debug: bool,
) -> None:
    duration, source_fps = _probe_video_info(source_path)

    effective_target = target_fps if target_fps and target_fps > 0 else None
    if effective_target and source_fps and source_fps > 0 and effective_target > source_fps:
        effective_target = source_fps

    max_workers = _resolve_worker_count(parallel_jobs, debug)
    chunk_plan = _plan_chunks(duration, max_workers)

    if len(chunk_plan) <= 1:
        command = _build_ffmpeg_command(
            source_path,
            os.path.join(output_folder, "%08d.png"),
            start=None,
            duration=None,
            target_fps=effective_target,
        )
        _run_command(command, debug=debug)
        return

    chunk_records: List[Tuple[float, str]] = []
    with ThreadPoolExecutor(max_workers=len(chunk_plan)) as executor:
        futures = []
        for index, (start, length) in enumerate(chunk_plan):
            chunk_dir = os.path.join(output_folder, f".chunk_{index:03d}")
            os.makedirs(chunk_dir, exist_ok=True)
            futures.append(
                executor.submit(
                    _extract_chunk,
                    source_path,
                    chunk_dir,
                    start=start,
                    duration=length,
                    target_fps=effective_target,
                    debug=debug,
                )
            )
            chunk_records.append((start, chunk_dir))

        for future in as_completed(futures):
            # Propagate any raised exception immediately
            future.result()

    _consolidate_chunk_outputs(chunk_records, output_folder)


def _clean_existing_png(output_folder: str) -> None:
    for image_path in glob.glob(os.path.join(output_folder, "*.png")):
        try:
            os.remove(image_path)
        except OSError:
            pass
    for chunk_dir in glob.glob(os.path.join(output_folder, ".chunk_*")):
        shutil.rmtree(chunk_dir, ignore_errors=True)


def handle(
    input_file: str,
    output_folder: str,
    config: "FramesConfig | None" = None,
    *,
    debug: bool = False,
    save_frames: bool = False,
):
    """Main entry point for frame extraction.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: FramesConfig instance or None for defaults.
        debug: Enable verbose debug output.
        save_frames: Whether to save extracted frames to disk.

    Returns:
        Tuple of (frames_folder, frame_data, frames_file).
    """
    return _extract(
        input_file,
        output_folder,
        debug=debug,
        save_frames=save_frames,
        target_fps=config.target_fps if config else None,
        parallel_jobs=config.parallel_jobs if config else None,
    )


def _extract(
    file,
    temp_folder,
    *,
    debug: bool = False,
    save_frames: bool = False,
    target_fps: Optional[float] = None,
    parallel_jobs: Optional[int] = None,
):

    file_type = file.split(".")[-1].lower()

    if file_type not in VIDEO_FILE_TYPES:
        print("Error: Frame extraction is only supported for video files.", file=sys.stderr)
        sys.exit(1)

    output_folder = os.path.join(temp_folder, "frames")
    frames_file = os.path.join(output_folder, "frames.json")
    has_output_folder = os.path.exists(output_folder)
    frames_file_exists = os.path.exists(frames_file)
    has_png_files = has_output_folder and any(
        os.path.isfile(os.path.join(output_folder, f)) and f.lower().endswith(".png")
        for f in os.listdir(output_folder)
    )

    needs_png_extraction = save_frames and (not has_output_folder or not has_png_files)
    needs_metadata_extraction = not frames_file_exists or needs_png_extraction

    if needs_png_extraction or needs_metadata_extraction:
        os.makedirs(output_folder, exist_ok=True)

    if needs_png_extraction:
        info_print("Extracting frames from the video file")
        _clean_existing_png(output_folder)
        effective_target_fps = float(target_fps) if target_fps and target_fps > 0 else None
        try:
            _extract_frames_from_video(
                file,
                output_folder,
                target_fps=effective_target_fps,
                parallel_jobs=parallel_jobs,
                debug=debug,
            )
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print(f"Output: {e.output}")
            print("Error occurred during frame extraction.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Error occurred during frame extraction.")

    if needs_metadata_extraction:
        info_print("Extracting frame PTS mapping")
        ffprobe_command = [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "info",
            "-threads",
            "16",
            "-i",
            file,
            "-show_frames",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
        ]

        try:
            _run_command(ffprobe_command, debug=debug, stdout_path=frames_file)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print(f"Output: {e.output}")
            print("Error occurred during frame PTS mapping extraction.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Error occurred during frame PTS mapping extraction.")

        # Read the json file and extract the frame pts and dts

    frames = []
    try:
        with open(frames_file, "r") as f:
            frames_data = json.load(f)

            frames = frames_data.get("frames", [])

            last_index = -1
            for position, frame in enumerate(frames):
                fallback = position if last_index < position else last_index + 1
                resolved = _resolve_frame_index(frame, fallback)
                if resolved <= last_index:
                    resolved = last_index + 1

                frame["source_index"] = int(resolved)
                frame["index"] = int(position)
                last_index = int(resolved)

            if save_frames:
                output_dir = os.path.dirname(frames_file)
                keyframes_folder = os.path.join(output_dir, "keyframes")
                os.makedirs(keyframes_folder, exist_ok=True)

                for index, frame in enumerate(frames):
                    if frame.get("key_frame") == 1:
                        frame_file = f"{index:08d}.png"
                        src_path = os.path.join(output_dir, frame_file)

                        # Check if the file exists before copying
                        if not os.path.isfile(src_path):
                            continue

                        dest_path = os.path.join(keyframes_folder, frame_file)
                        shutil.copy(src_path, dest_path)
    except FileNotFoundError:
        print(f"File {frames_file} not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return output_folder, frames, frames_file