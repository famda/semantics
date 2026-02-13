"""Video Processing CLI Tool.

A comprehensive video processing pipeline for frame extraction, object
detection, scene analysis, captioning, and more.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import rich_click as click

# Setup paths before importing local modules
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    platform_root = os.path.dirname(script_dir)

    if platform_root not in sys.path:
        sys.path.insert(0, platform_root)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from global_helpers import VIDEO_FILE_TYPES, select_frame_indices, coerce_int

except ImportError as e:
    print(f"\n****** ERROR: Failed to import required modules ******", file=sys.stderr)
    print(f"Reason: {e}", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during initial setup: {e}", file=sys.stderr)
    sys.exit(1)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


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
@click.option("--plain", is_flag=True, help="Disable rich formatting, use plain text output")
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to YAML config file")
@click.option("--slice-start", type=str, default=None, help="Start timestamp for slicing (HH:MM:SS or HH:MM:SS.mmm)")
@click.option("--slice-end", type=str, default=None, help="End timestamp for slicing (HH:MM:SS or HH:MM:SS.mmm)")
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
    plain: bool,
    config: Optional[str],
    slice_start: Optional[str],
    slice_end: Optional[str],
) -> None:
    """
    \b
    Semantics CLI [video] - Unified interface for video intelligence
    -------------------------------------------
    Extract meaning, not just metadata. Composable AI operations designed for developers.
    """
    from modules.utils.logging import (
        configure_external_logging,
        debug_print,
        gray_debug_output,
        info_print,
        print_header,
        print_summary_table,
        reset_timings,
        run_module,
        set_debug,
        set_input_subtitle,
        set_plain,
        skip_module,
        install_abort_handler,
        restore_abort_handler,
        register_planned_modules,
        start_pipeline,
        stop_pipeline,
    )

    set_plain(plain)
    set_debug(debug)
    reset_timings()
    _start_time = time.perf_counter()

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
    if video_config is None:
        from config import VideoConfig
        video_config = VideoConfig()

    # Detect URL input
    file = input_file
    is_url = file.startswith("https://www.youtube.com/watch?v=") or file.startswith("https://youtu.be/")

    if not is_url:
        if not os.path.exists(input_file):
            click.echo(f"Error: The file {input_file} does not exist.", err=True)
            sys.exit(1)

    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    if not is_url:
        file_type = file.split(".")[-1].lower()
        if file_type not in VIDEO_FILE_TYPES:
            click.echo(f"Error: The file {file} is not a supported video file type.", err=True)
            sys.exit(1)

    configure_external_logging(debug)
    print_header("video", file)

    # Build planned module list
    planned: list[str] = []
    if is_url:
        planned.append("Download")
    if slice_start or slice_end:
        planned.append("Slice")
    if from_frames:
        planned.append("Frame Extraction")
    elif from_segments:
        planned.append("Segment Extraction")
    elif from_clustering:
        planned.append("Frame Clustering")
    if scenes:
        planned.append("Scene Detection")
    if captions:
        planned.append("Captions")
    if tiles:
        planned.append("Tiles")
    if extract_text:
        planned.append("OCR")
    if extract_objects:
        planned.append("Object Detection")
    if classify:
        planned.append("Classification")
    if named_entities:
        planned.append("Named Entities")
    if actions:
        planned.append("Action Recognition")

    register_planned_modules(planned)
    start_pipeline(len(planned), "video", file)
    install_abort_handler()

    # Frame selection state
    frame_indices_to_process: list[int] = []
    frames_folder = ""
    frame_data = None
    frames_file = ""
    selection_mode: str | None = None

    try:
        # Download from URL if needed
        if is_url:
            from modules import download as downloader

            download_cfg = video_config.download if video_config else None
            # CLI --download-resolution overrides config
            if download_resolution is not None:
                from config import DownloadConfig
                base = download_cfg or DownloadConfig()
                download_cfg = base.model_copy(update={"max_height": download_resolution})
            file, _ = run_module(
                "Download", downloader.handle,
                input_file, output_folder, config=download_cfg, debug=debug,
            )
            if not file:
                return  # Download failed (already recorded in summary)
            # Unpack (path, title) tuple from download handle
            if isinstance(file, tuple):
                file, video_title = file
                if video_title:
                    set_input_subtitle(video_title)

        # Slice video to requested time range
        if slice_start or slice_end:
            from modules import slice as slicer

            slice_cfg = video_config.slice if video_config else None
            file, _ = run_module(
                "Slice", slicer.handle,
                file, temp_folder, config=slice_cfg,
                start_time=slice_start, end_time=slice_end, debug=debug,
            )

        # Extract frames based on selected mode
        if from_frames:
            selection_mode = "frames"
            with gray_debug_output(debug):
                from modules import frames as frames_module

            frames_cfg = video_config.frames if video_config else None
            if frames_cfg and frames_per_second is not None:
                frames_cfg = frames_cfg.model_copy(update={"target_fps": frames_per_second})
            elif frames_cfg is None and frames_per_second is not None:
                from config import FramesConfig
                frames_cfg = FramesConfig(target_fps=frames_per_second)
            (frames_folder, frame_data, frames_file), _ = run_module(
                "Frame Extraction", frames_module.handle,
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
            (frames_folder, frame_data, frames_file), _ = run_module(
                "Segment Extraction", segment_module.handle,
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
            (frames_folder, frame_data, frames_file), _ = run_module(
                "Frame Clustering", clustering_module.handle,
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
                        idx = coerce_int(entry.get("index", pos) if isinstance(entry, dict) else pos)
                        if idx is not None:
                            collected.append(idx)
            elif selection_mode == "segments":
                for entry in frame_data:
                    if isinstance(entry, dict):
                        idx = coerce_int(entry.get("index"))
                        if idx is not None:
                            collected.append(idx)
            elif selection_mode == "clustering":
                for entry in frame_data:
                    if isinstance(entry, dict):
                        if entry.get("keyframe") is False:
                            continue
                        idx = coerce_int(entry.get("index") or entry.get("frame_index"))
                        if idx is not None:
                            collected.append(idx)
            if collected:
                frame_indices_to_process = sorted(dict.fromkeys(collected))

        # Scene extraction
        if scenes:
            from modules import scenes as scenes_module

            scenes_cfg = video_config.scenes if video_config else None
            _, _ = run_module(
                "Scene Detection", scenes_module.handle,
                file, temp_folder, config=scenes_cfg, debug=debug,
            )

        if frame_indices_to_process:
            debug_print(f"Frames selected for processing: {len(frame_indices_to_process)}", debug=debug)

        # Caption extraction
        if captions:
            if not frame_indices_to_process:
                skip_module("Captions", "no frame indexes selected")
            else:
                with gray_debug_output(debug):
                    from modules import captions as captions_module

                captions_cfg = video_config.captions if video_config else None
                _, _ = run_module(
                    "Captions", captions_module.handle,
                    file,
                    temp_folder,
                    config=captions_cfg,
                    frame_indices=frame_indices_to_process,
                    debug=debug,
                )

        # Tile creation
        if tiles:
            if not frame_indices_to_process:
                skip_module("Tiles", "no frame indexes selected")
            else:
                if debug:
                    preview = frame_indices_to_process[:10]
                    debug_print(f"Sample frame indexes for tiles: {preview}", debug=debug)
                with gray_debug_output(debug):
                    from modules import tiles as tiles_module

                tiles_cfg = video_config.tiles if video_config else None
                _, _ = run_module(
                    "Tiles", tiles_module.handle,
                    file,
                    temp_folder,
                    config=tiles_cfg,
                    frame_indices=frame_indices_to_process,
                    debug=debug,
                )

        # OCR (text extraction)
        if extract_text:
            if not frame_indices_to_process:
                skip_module("OCR", "no frame indexes selected")
            else:
                with gray_debug_output(debug):
                    from modules import ocr as ocr_module

                ocr_cfg = video_config.ocr if video_config else None
                _, _ = run_module(
                    "OCR", ocr_module.handle,
                    file,
                    temp_folder,
                    config=ocr_cfg,
                    frame_indices=frame_indices_to_process,
                    debug=debug,
                )

        # Object detection
        if extract_objects:
            with gray_debug_output(debug):
                from modules import objects as objects_module

            objects_cfg = video_config.objects if video_config else None
            _, _ = run_module(
                "Object Detection", objects_module.handle,
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
                skip_module("Classification", "no frame indexes selected")
            else:
                with gray_debug_output(debug):
                    from modules import classify as classify_module

                classification_cfg = video_config.classification if video_config else None
                _, _ = run_module(
                    "Classification", classify_module.handle,
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
                skip_module("Named Entities", "requires captions (-c/--captions)")
            else:
                with gray_debug_output(debug):
                    from modules import entities as entities_module

                ner_cfg = video_config.ner if video_config else None
                _, _ = run_module(
                    "Named Entities", entities_module.handle,
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
            _, _ = run_module(
                "Action Recognition", actions_module.handle,
                file,
                temp_folder,
                config=actions_cfg,
                debug=debug,
            )

    except KeyboardInterrupt:
        pass  # abort — summary table will show remaining as "not run"
    finally:
        restore_abort_handler()
        stop_pipeline()
        total_elapsed = time.perf_counter() - _start_time
        print_summary_table(total_elapsed=total_elapsed)


if __name__ == "__main__":
    main()
