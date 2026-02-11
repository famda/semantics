"""Audio Processing CLI Tool.

A comprehensive audio processing pipeline using state-of-the-art AI models
for transcription, diarization, emotion recognition, and more.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Optional

import rich_click as click

# Suppress SyntaxWarning from third-party packages with unescaped regex patterns (Python 3.13+)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"lhotse\..*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pydub\..*")

# Setup paths before importing local modules
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    platform_root = os.path.dirname(script_dir)

    if platform_root not in sys.path:
        sys.path.insert(0, platform_root)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from global_helpers import AUDIO_FILE_TYPES, VIDEO_FILE_TYPES

except ImportError as e:
    print("\n****** ERROR: Failed to import required modules ******", file=sys.stderr)
    print(f"Reason: {e}", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during initial setup: {e}", file=sys.stderr)
    sys.exit(1)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i",
    "--input",
    "input_file",
    required=True,
    type=str,
    help="Input media file or YouTube URL",
)
@click.option(
    "-o",
    "--output",
    "output_folder",
    required=True,
    type=click.Path(),
    help="Output folder path",
)
@click.option("-e", "--enhance-audio", is_flag=True, help="Enhance audio quality")
@click.option("-n", "--denoise", is_flag=True, help="Denoise the audio file")
@click.option(
    "-s", "--stem", is_flag=True, help="Enable source separation (extract vocals)"
)
@click.option("-v", "--vad", is_flag=True, help="Enable Voice Activity Detection")
@click.option("-t", "--transcribe", is_flag=True, help="Transcribe the audio to text")
@click.option(
    "-te",
    "--transcribe-experimental",
    is_flag=True,
    help="Experimental: Ultra-fast transcription with CTC alignment",
)
@click.option("-d", "--diarize", is_flag=True, help="Enable speaker diarization")
@click.option("-ctc", "--ctc-align", is_flag=True, help="Enable CTC forced alignment")
@click.option("-c", "--classify", is_flag=True, help="Enable audio classification")
@click.option(
    "-ct",
    "--classify-timeline",
    is_flag=True,
    help="Enable timeline audio classification",
)
@click.option("-em", "--emotion", is_flag=True, help="Enable emotion recognition")
@click.option("-se", "--scene", is_flag=True, help="Enable scene/chapter detection")
@click.option("-ner", "--named-entities", is_flag=True, help="Extract named entities from transcript")
@click.option("--debug", is_flag=True, help="Enable verbose debug logging")
@click.option("--plain", is_flag=True, help="Disable rich formatting, use plain text output")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file",
)
@click.option("--slice-start", type=str, default=None, help="Start timestamp for slicing (HH:MM:SS or HH:MM:SS.mmm)")
@click.option("--slice-end", type=str, default=None, help="End timestamp for slicing (HH:MM:SS or HH:MM:SS.mmm)")
def main(
    input_file: str,
    output_folder: str,
    enhance_audio: bool,
    denoise: bool,
    stem: bool,
    vad: bool,
    transcribe: bool,
    transcribe_experimental: bool,
    diarize: bool,
    ctc_align: bool,
    classify: bool,
    classify_timeline: bool,
    emotion: bool,
    scene: bool,
    named_entities: bool,
    debug: bool,
    plain: bool,
    config: Optional[str],
    slice_start: Optional[str],
    slice_end: Optional[str],
) -> None:
    """
    \b
    Semantics CLI [audio] - Unified interface for audio intelligence
    -------------------------------------------
    Extract meaning, not just metadata. Composable AI operations designed for developers.
    """
    from modules.utils.logging import (
        configure_external_logging,
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

    # Load configuration if provided
    audio_config = None
    if config:
        try:
            from config import load_audio_config
            audio_config = load_audio_config(config)
        except Exception as exc:
            click.echo(f"Error: Failed to load config from {config}: {exc}", err=True)
            sys.exit(1)
    if audio_config is None:
        from config import AudioConfig
        audio_config = AudioConfig()

    # Detect URL input
    is_url = input_file.startswith("https://www.youtube.com/watch?v=") or input_file.startswith("https://youtu.be/")

    if not is_url:
        # Validate input file exists
        if not os.path.exists(input_file):
            click.echo(f"Error: The file {input_file} does not exist.", err=True)
            sys.exit(1)

    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    if not is_url:
        file_type = input_file.split(".")[-1].lower()
        SUPPORTED_FILE_TYPES = AUDIO_FILE_TYPES + VIDEO_FILE_TYPES
        if file_type not in SUPPORTED_FILE_TYPES:
            click.echo(
                f"Error: The file {input_file} is not a supported file type.", err=True
            )
            sys.exit(1)

    configure_external_logging(debug)
    print_header("audio", input_file)

    # Build planned module list
    planned = []
    if is_url:
        planned.append("Download")
    if slice_start or slice_end:
        planned.append("Slice")
    planned.append("Resample")
    if stem:
        planned.append("Source Separation")
    if enhance_audio:
        planned.append("Audio Enhancement")
    if denoise:
        planned.append("Denoise")
    if vad:
        planned.append("VAD")
    if transcribe:
        planned.append("Transcription")
    if transcribe_experimental:
        planned.append("Transcription (Exp.)")
    if diarize:
        planned.append("Diarization")
    if ctc_align:
        planned.append("CTC Alignment")
    if emotion:
        planned.append("Emotion Recognition")
    if scene:
        planned.append("Scene Detection")
    if named_entities:
        planned.append("Named Entities")
    if classify:
        planned.append("Classification")
    if classify_timeline:
        planned.append("Timeline Classification")

    register_planned_modules(planned)
    start_pipeline(len(planned), "audio", input_file)
    install_abort_handler()

    try:
        # Download from URL if needed
        if is_url:
            from modules import download as downloader

            download_cfg = audio_config.download if audio_config else None
            download_result, _ = run_module(
                "Download", downloader.handle,
                input_file, output_folder, config=download_cfg, debug=debug,
            )
            if not download_result:
                return  # Download failed (already recorded in summary)
            input_file, video_title = download_result
            if video_title:
                set_input_subtitle(video_title)

        # Slice media to requested time range (before resampling)
        if slice_start or slice_end:
            from modules import slice as slicer

            slice_cfg = audio_config.slice if audio_config else None
            input_file, _ = run_module(
                "Slice", slicer.handle,
                input_file, temp_folder, config=slice_cfg,
                start_time=slice_start, end_time=slice_end, debug=debug,
            )

        # Resample audio to standard format
        if os.path.exists(os.path.join(temp_folder, "audio.wav")):
            audio_file = os.path.join(temp_folder, "audio.wav")
        else:
            from modules import resample as resampler

            resample_cfg = audio_config.resample if audio_config else None
            audio_file, _ = run_module(
                "Resample", resampler.handle,
                input_file, temp_folder, config=resample_cfg, debug=debug,
            )

        # Source separation (extract vocals)
        if stem:
            from modules import stem as stem_module

            stem_cfg = audio_config.stem if audio_config else None
            audio_file, _ = run_module(
                "Source Separation", stem_module.handle,
                audio_file, temp_folder, config=stem_cfg, debug=debug,
            )

        # Audio enhancement
        if enhance_audio:
            from modules import resample as resampler

            enhance_cfg = audio_config.enhance if audio_config else None
            audio_file, _ = run_module(
                "Audio Enhancement", resampler.enhance,
                audio_file, temp_folder, config=enhance_cfg, debug=debug,
            )

        # Denoising
        if denoise:
            from modules import denoise as denoiser

            denoise_cfg = audio_config.denoise if audio_config else None
            audio_file, _ = run_module(
                "Denoise", denoiser.handle,
                audio_file, temp_folder, config=denoise_cfg, debug=debug,
            )

        # Voice Activity Detection
        if vad:
            from modules import vad as vad_module

            vad_cfg = audio_config.vad if audio_config else None
            _, _ = run_module(
                "VAD", vad_module.handle,
                audio_file, temp_folder, config=vad_cfg, debug=debug,
            )

        # Transcription
        if transcribe:
            from modules import transcribe as transcriber

            transcribe_cfg = audio_config.transcribe if audio_config else None
            _, _ = run_module(
                "Transcription", transcriber.handle,
                audio_file, temp_folder, config=transcribe_cfg, debug=debug,
            )

        # Experimental transcription (ultra-fast with CTC alignment)
        if transcribe_experimental:
            from modules import transcribe_experimental as transcriber_exp

            transcribe_exp_cfg = (
                audio_config.transcribe_experimental if audio_config else None
            )
            _, _ = run_module(
                "Transcription (Exp.)", transcriber_exp.handle,
                audio_file, temp_folder, config=transcribe_exp_cfg, debug=debug,
            )

        # Speaker diarization
        if diarize:
            from modules import diarize as diarizer

            diarize_cfg = audio_config.diarize if audio_config else None
            _, _ = run_module(
                "Diarization", diarizer.handle,
                audio_file, temp_folder, config=diarize_cfg, debug=debug,
            )

        # CTC forced alignment
        if ctc_align:
            transcript_json = os.path.join(
                temp_folder, "transcription", "transcription.json"
            )
            diarization_json = os.path.join(
                temp_folder, "diarization", "diarization.json"
            )

            if (
                not transcribe
                or not diarize
                or not os.path.exists(transcript_json)
                or not os.path.exists(diarization_json)
            ):
                skip_module("CTC Alignment", "requires transcription and diarization")
            else:
                from modules import ctc as ctc_module

                ctc_cfg = audio_config.ctc if audio_config else None
                _, _ = run_module(
                    "CTC Alignment", ctc_module.handle,
                    audio_file, temp_folder, config=ctc_cfg, debug=debug,
                )

        # Emotion recognition
        if emotion:
            ctc_json = os.path.join(temp_folder, "ctc", "alignment.json")
            transcript_json = os.path.join(
                temp_folder, "transcription", "transcription.json"
            )
            emotion_cfg = audio_config.emotion if audio_config else None

            from modules import emotions as emotions_module

            segments_file = None
            if ctc_align and os.path.exists(ctc_json):
                segments_file = ctc_json
            elif transcribe and os.path.exists(transcript_json):
                segments_file = transcript_json

            if segments_file:
                _, _ = run_module(
                    "Emotion Recognition", emotions_module.handle,
                    audio_file,
                    temp_folder,
                    config=emotion_cfg,
                    segments_file=segments_file,
                    debug=debug,
                )
            else:
                skip_module("Emotion Recognition", "requires transcription or CTC alignment")

        # Scene/chapter detection
        if scene:
            ctc_json = os.path.join(temp_folder, "ctc", "alignment.json")
            transcript_json = os.path.join(
                temp_folder, "transcription", "transcription.json"
            )
            scenes_cfg = audio_config.scenes if audio_config else None

            segments_file = None
            if ctc_align and os.path.exists(ctc_json):
                segments_file = ctc_json
            elif transcribe and os.path.exists(transcript_json):
                segments_file = transcript_json

            if segments_file:
                from modules import scenes as scenes_module

                _, _ = run_module(
                    "Scene Detection", scenes_module.handle,
                    segments_file, temp_folder, config=scenes_cfg, debug=debug,
                )
            else:
                skip_module("Scene Detection", "requires transcription or CTC alignment")

        # Named Entity Recognition
        if named_entities:
            ctc_json = os.path.join(temp_folder, "ctc", "alignment.json")
            transcript_json = os.path.join(
                temp_folder, "transcription", "transcription.json"
            )
            ner_cfg = audio_config.ner if audio_config else None

            segments_file = None
            if ctc_align and os.path.exists(ctc_json):
                segments_file = ctc_json
            elif transcribe and os.path.exists(transcript_json):
                segments_file = transcript_json

            if segments_file:
                from modules import entities as entities_module

                _, _ = run_module(
                    "Named Entities", entities_module.handle,
                    audio_file,
                    temp_folder,
                    config=ner_cfg,
                    segments_file=segments_file,
                    debug=debug,
                )
            else:
                skip_module("Named Entities", "requires transcription or CTC alignment")

        # Audio classification
        if classify:
            from modules import classify as classifier

            classify_cfg = audio_config.classify if audio_config else None
            _, _ = run_module(
                "Classification", classifier.handle,
                audio_file, temp_folder, config=classify_cfg, debug=debug,
            )

        # Timeline classification
        if classify_timeline:
            from modules import classify_timeline as timeline

            timeline_cfg = audio_config.classify_timeline if audio_config else None
            _, _ = run_module(
                "Timeline Classification", timeline.handle,
                audio_file, temp_folder, config=timeline_cfg, debug=debug,
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
