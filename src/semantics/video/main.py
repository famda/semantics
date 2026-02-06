"""Video Processing CLI Tool.

A comprehensive video processing pipeline for frame extraction, object
detection, scene analysis, captioning, and more.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import click

# Setup paths before importing local modules
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    platform_root = os.path.dirname(script_dir)

    if platform_root not in sys.path:
        sys.path.insert(0, platform_root)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from global_helpers import VIDEO_FILE_TYPES, select_frame_indices

except ImportError as e:
    print(f"\n****** ERROR: Failed to import required modules ******", file=sys.stderr)
    print(f"Reason: {e}", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during initial setup: {e}", file=sys.stderr)
    sys.exit(1)


# Start timer for total execution time measurement
_start_time = time.perf_counter()
_modules_executed = False  # Track if any modules were executed


def _print_elapsed_time():
    """Print total execution time on exit (only if modules were executed)."""
    if not _modules_executed:
        return
    try:
        elapsed = time.perf_counter() - _start_time
        secs = int(elapsed)
        if secs < 60:
            print(f"Total execution time: {elapsed:.3f} seconds")
        elif secs < 3600:
            minutes = secs // 60
            seconds = secs % 60
            ms = int((elapsed - secs) * 1000)
            print(f"Total execution time: {minutes} minute(s) {seconds} second(s) {ms} ms")
        else:
            hours = secs // 3600
            minutes = (secs % 3600) // 60
            seconds = secs % 60
            print(f"Total execution time: {hours} hour(s) {minutes} minute(s) {seconds} second(s)")
    except Exception:
        pass


atexit.register(_print_elapsed_time)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _ensure_tf_keras_for_tf(debug: bool) -> None:
    """Install tf-keras if TensorFlow >= 2.20 is present but tf-keras isn't."""

    def get_version_dist(name: str) -> str | None:
        try:
            from importlib import metadata as importlib_metadata
        except Exception:
            try:
                import importlib_metadata
            except Exception:
                importlib_metadata = None
        if importlib_metadata:
            try:
                return importlib_metadata.version(name)
            except Exception:
                pass
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "show", name],
                check=False,
                capture_output=True,
                text=True,
            )
            if out.returncode == 0 and out.stdout:
                for line in out.stdout.splitlines():
                    if line.lower().startswith("version:"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return None

    tf_ver = get_version_dist("tensorflow")
    if not tf_ver:
        return

    try:
        major, minor = [int(x) for x in tf_ver.split(".")[:2]]
    except Exception:
        return

    if not (major > 2 or (major == 2 and minor >= 20)):
        return

    if get_version_dist("tf-keras") or get_version_dist("tf_keras"):
        return

    env = dict(os.environ)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    base = [
        sys.executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "-q",
    ]
    if debug:
        try:
            from modules.utils.logging import debug_print

            debug_print("Installing required dependencies...", debug=True)
        except Exception:
            pass
    subprocess.run(
        base + ["tf-keras"],
        check=False,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not (get_version_dist("tf-keras") or get_version_dist("tf_keras")):
        subprocess.run(
            base + ["tf_keras"],
            check=False,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _coerce_int(value) -> Optional[int]:
    """Convert value to int, handling floats and None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-i", "--input", "input_file", required=True, type=str, help="Input video file or YouTube URL")
@click.option("-o", "--output", "output_folder", required=True, type=click.Path(), help="Output folder path")
@click.option("-t", "--tiles", is_flag=True, help="Enable video tiling")
@click.option("-eo", "--extract-objects", is_flag=True, help="Extract objects from the video")
@click.option("-co", "--cluster-objects", is_flag=True, help="Cluster the extracted objects")
@click.option("-classes", "--object-classes", multiple=True, default=["person"], help="Object classes to extract")
@click.option("--save-annotations", is_flag=True, help="Persist detection crops and masks to disk")
@click.option("-c", "--captions", is_flag=True, help="Extract captions from the video")
@click.option("-s", "--scenes", is_flag=True, help="Enable scene extraction")
@click.option("-ocr", "--extract-text", is_flag=True, help="Enable text extraction (OCR)")
@click.option("-cl", "--classify", is_flag=True, help="Enable frame classification")
@click.option("-ner", "--named-entities", is_flag=True, help="Extract named entities from captions")
@click.option("-a", "--actions", is_flag=True, help="Recognize human actions in the video")
@click.option("--download-resolution", type=int, default=None, help="Max video height when downloading from URL")
@click.option("--from-frames", is_flag=True, help="Analyze from extracted video frames")
@click.option("--from-clustering", is_flag=True, help="Analyze from keyframe/clustering on frames")
@click.option("--from-segments", is_flag=True, help="Analyze from keyframes/segments")
@click.option("--save-frames", is_flag=True, help="Save extracted frames to disk")
@click.option("-fps", "--frames-per-second", type=int, default=1, help="Frames per second to analyze")
@click.option("--debug", is_flag=True, help="Enable verbose debug logging")
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to YAML config file")
def main(
    input_file: str,
    output_folder: str,
    tiles: bool,
    extract_objects: bool,
    cluster_objects: bool,
    object_classes: Tuple[str, ...],
    save_annotations: bool,
    captions: bool,
    scenes: bool,
    extract_text: bool,
    classify: bool,
    named_entities: bool,
    actions: bool,
    download_resolution: Optional[int],
    from_frames: bool,
    from_clustering: bool,
    from_segments: bool,
    save_frames: bool,
    frames_per_second: int,
    debug: bool,
    config: Optional[str],
) -> None:
    """
    \b
    Semantics CLI [video] - Unified interface for video intelligence
    -------------------------------------------
    Extract meaning, not just metadata. Composable AI operations designed for developers.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if debug else "3"

    # Check which modules require frame selection modes
    needs_frame_selection = any([
        tiles,
        extract_objects,
        captions,
        scenes,
        extract_text,
        classify,
        named_entities,
    ])

    # Validate that at least one frame selection mode is specified (if needed)
    if needs_frame_selection and not (from_frames or from_clustering or from_segments):
        click.echo(
            "Error: You must specify a frame selection mode. Use one of:\n"
            "  --from-frames      Analyze from extracted video frames\n"
            "  --from-clustering  Analyze from keyframe/clustering on frames\n"
            "  --from-segments    Analyze from keyframes/segments",
            err=True,
        )
        sys.exit(1)

    # Load configuration if provided
    video_config = None
    if config:
        try:
            from config import load_video_config

            video_config = load_video_config(config)
        except Exception as exc:
            click.echo(f"Error: Failed to load config from {config}: {exc}", err=True)
            sys.exit(1)

    # Create output directories
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, "temp"))

    # Handle YouTube/URL downloads
    file = input_file
    if file.startswith("https://www.youtube.com/watch?v=") or file.startswith("https://youtu.be/"):
        from modules import download as downloader

        download_cfg = video_config.download if video_config else None
        max_height = download_resolution
        if max_height is None and download_cfg:
            max_height = download_cfg.max_height
        downloaded = downloader.handle(input_file, output_folder, download_cfg)
        if not downloaded:
            click.echo(f"Error: Failed to download video from {input_file}", err=True)
            sys.exit(1)
        file = downloaded
    else:
        if not os.path.exists(input_file):
            click.echo(f"Error: The file {input_file} does not exist.", err=True)
            sys.exit(1)

    temp_folder = os.path.join(output_folder, "temp")
    file_type = file.split(".")[-1].lower()

    if file_type not in VIDEO_FILE_TYPES:
        click.echo(f"Error: The file {file} is not a supported video file type.", err=True)
        sys.exit(1)

    from modules.utils.logging import configure_external_logging, gray_debug_output

    configure_external_logging(debug)

    # Mark that we're executing modules (for execution time display)
    global _modules_executed
    _modules_executed = True

    # Frame selection state
    frame_indices_to_process: list[int] = []
    frames_folder = ""
    frame_data = None
    frames_file = ""
    selection_mode: str | None = None

    # Extract frames based on selected mode
    if from_frames:
        selection_mode = "frames"
        with gray_debug_output(debug):
            from modules import frames as frames_module

        frames_cfg = video_config.frames if video_config else None
        # Override target_fps in config if frames_per_second CLI flag was provided
        if frames_cfg and frames_per_second is not None:
            frames_cfg.target_fps = frames_per_second
        elif frames_cfg is None and frames_per_second is not None:
            from config import FramesConfig
            frames_cfg = FramesConfig(target_fps=frames_per_second)
        frames_folder, frame_data, frames_file = frames_module.handle(
            file,
            temp_folder,
            config=frames_cfg,
            save_frames=save_frames,
            debug=debug,
        )

    elif from_segments:
        selection_mode = "segments"
        with gray_debug_output(debug):
            from modules import segment as segment_module

        segments_cfg = video_config.segments if video_config else None
        frames_folder, frame_data, frames_file = segment_module.handle(
            file,
            temp_folder,
            config=segments_cfg,
            save_frames=save_frames,
            debug=debug,
        )

    elif from_clustering:
        selection_mode = "clustering"
        with gray_debug_output(debug):
            from modules import clustering as clustering_module

        clustering_cfg = video_config.clustering if video_config else None
        frames_folder, frame_data, frames_file = clustering_module.handle(
            file,
            temp_folder,
            config=clustering_cfg,
            save_frames=save_frames,
            debug=debug,
        )

    # Collect frame indices from extracted data
    if isinstance(frame_data, list) and selection_mode:
        collected: list[int] = []
        if selection_mode == "frames":
            positions = select_frame_indices(frame_data, frames_per_second)
            for pos in positions:
                if 0 <= pos < len(frame_data):
                    entry = frame_data[pos]
                    idx = _coerce_int(entry.get("index", pos) if isinstance(entry, dict) else pos)
                    if idx is not None:
                        collected.append(idx)
        elif selection_mode == "segments":
            for entry in frame_data:
                if isinstance(entry, dict):
                    idx = _coerce_int(entry.get("index"))
                    if idx is not None:
                        collected.append(idx)
        elif selection_mode == "clustering":
            for entry in frame_data:
                if isinstance(entry, dict):
                    if entry.get("keyframe") is False:
                        continue
                    idx = _coerce_int(entry.get("index") or entry.get("frame_index"))
                    if idx is not None:
                        collected.append(idx)
        if collected:
            frame_indices_to_process = sorted(dict.fromkeys(collected))

    # Scene extraction
    if scenes:
        from modules import scenes as scenes_module

        scenes_cfg = video_config.scenes if video_config else None
        scenes_module.handle(file, temp_folder, config=scenes_cfg, debug=debug)

    if frame_indices_to_process:
        print(f"INFO: Frames selected for processing ({len(frame_indices_to_process)})")

    # Caption extraction
    if captions:
        if not frame_indices_to_process:
            click.echo("ERROR: Captions requested but no frame indexes selected", err=True)
        else:
            with gray_debug_output(debug):
                from modules import captions as captions_module

            captions_cfg = video_config.captions if video_config else None
            captions_module.handle(
                file,
                temp_folder,
                config=captions_cfg,
                frame_indices=frame_indices_to_process,
                debug=debug,
            )

    # Tile creation
    if tiles:
        if not frame_indices_to_process:
            click.echo("ERROR: Tiles requested but no frame indexes selected", err=True)
        else:
            if debug:
                preview = frame_indices_to_process[:10]
                print(f"DEBUG: Sample frame indexes for tiles: {preview}")
            with gray_debug_output(debug):
                from modules import tiles as tiles_module

            tiles_cfg = video_config.tiles if video_config else None
            tiles_module.handle(
                file,
                temp_folder,
                config=tiles_cfg,
                frame_indices=frame_indices_to_process,
                debug=debug,
            )

    # OCR (text extraction)
    if extract_text:
        if not frame_indices_to_process:
            click.echo("ERROR: OCR requested but no frame indexes selected", err=True)
        else:
            with gray_debug_output(debug):
                from modules import ocr as ocr_module

            ocr_cfg = video_config.ocr if video_config else None
            ocr_module.handle(
                file,
                temp_folder,
                config=ocr_cfg,
                frame_indices=frame_indices_to_process,
                debug=debug,
            )

    # Object detection
    if extract_objects:
        _ensure_tf_keras_for_tf(debug)
        with gray_debug_output(debug):
            from modules import objects as objects_module

        objects_cfg = video_config.objects if video_config else None
        objects_module.handle(
            file,
            temp_folder,
            config=objects_cfg,
            object_classes=list(object_classes),
            frame_indices=frame_indices_to_process,
            perform_clustering=cluster_objects,
            save_annotations=save_annotations,
            debug=debug,
        )

    # Image classification
    if classify:
        if not frame_indices_to_process:
            click.echo("ERROR: Classification requested but no frame indexes selected", err=True)
        else:
            with gray_debug_output(debug):
                from modules import classify as classify_module

            classification_cfg = video_config.classification if video_config else None
            classify_module.handle(
                file,
                temp_folder,
                config=classification_cfg,
                frame_indices=frame_indices_to_process,
                save_annotations=save_annotations,
                debug=debug,
            )

    # Named Entity Recognition (from captions)
    if named_entities:
        captions_json = os.path.join(temp_folder, "captions", "captions.json")
        if not captions or not os.path.exists(captions_json):
            click.echo(
                "ERROR: NER requires captions. Use -c/--captions flag first.",
                err=True,
            )
        else:
            with gray_debug_output(debug):
                from modules import entities as entities_module

            ner_cfg = video_config.ner if video_config else None
            entities_module.handle(
                file,
                temp_folder,
                config=ner_cfg,
                captions_file=captions_json,
                debug=debug,
            )

    # Action recognition
    if actions:
        with gray_debug_output(debug):
            from modules import actions as actions_module

        actions_cfg = video_config.actions if video_config else None
        actions_module.handle(
            file,
            temp_folder,
            config=actions_cfg,
            debug=debug,
        )


if __name__ == "__main__":
    main()