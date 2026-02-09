"""Audio denoising module.

Applies AI-based noise reduction to audio files using the audio-denoiser library.
Supports chunked processing for long audio files to manage memory usage.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from audio_denoiser.AudioDenoiser import AudioDenoiser

from .utils.chunks import cleanup_chunks, concatenate_audio, split_audio
from .utils.logging import debug_print, gray_debug_output, info_print, update_sub_progress

if TYPE_CHECKING:
    from config import DenoiseConfig

__all__ = ["handle"]

# Module-level state for auto-scale feature detection
_auto_scale_available: list[bool] = [True]  # use list to avoid global mutation
_FFMPEG_ENV = os.environ.copy()
_FFMPEG_ENV["AV_LOG_FORCE_NOCOLOR"] = "1"


@lru_cache(maxsize=1)
def _get_denoiser() -> tuple[AudioDenoiser, torch.device]:
    """Load and cache the denoiser model."""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return AudioDenoiser(device=device), device


def handle(
    input_file: str,
    output_folder: str,
    config: "DenoiseConfig | None" = None,
    *,
    debug: bool = False,
) -> str:
    """Denoise an audio file.

    Args:
        input_file: Path to the input audio file.
        output_folder: Path to the output folder for processed files.
        config: DenoiseConfig with chunk_length and allow_auto_scale settings.
        debug: Enable verbose logging.

    Returns:
        Path to the denoised audio file.
    """
    # Extract config values (use defaults if config is None)
    chunk_length = config.chunk_length if config else 900
    allow_auto_scale = config.allow_auto_scale if config else True

    info_print("Performing audio denoising")

    with gray_debug_output(debug):
        denoiser, device = _get_denoiser()
    debug_print(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} device: {device}", debug=debug)

    os.makedirs(output_folder, exist_ok=True)
    out_audio_file = os.path.join(output_folder, "denoised.wav")

    chunks, chunk_dir = split_audio(input_file, output_folder, "denoise", chunk_length)

    debug_print(
        "Running denoiser on entire file" if chunk_dir is None else f"Splitting audio into {len(chunks)} chunk(s)",
        debug=debug,
    )

    def run_denoiser(src: str, dst: str) -> None:
        """Run denoiser with auto-scale fallback."""
        use_auto_scale = allow_auto_scale and _auto_scale_available[0]

        debug_print(f"Invoking AudioDenoiser with auto_scale={use_auto_scale}", debug=debug)
        try:
            with gray_debug_output(debug):
                denoiser.process_audio_file(src, dst, auto_scale=use_auto_scale)
        except RuntimeError as exc:
            if use_auto_scale and "quantile" in str(exc).lower():
                print("WARN: Auto-scale failed due to large tensor in torch.quantile; retrying without auto scaling.")
                _auto_scale_available[0] = False
                debug_print("Retrying AudioDenoiser with auto_scale=False", debug=debug)
                with gray_debug_output(debug):
                    denoiser.process_audio_file(src, dst, auto_scale=False)
                return
            raise

    def normalize_audio(path: str) -> None:
        """Normalize audio to 16-bit PCM mono 16kHz if needed."""
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_fmt,sample_rate,channels",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ]
        try:
            result = subprocess.run(probe_cmd, check=False, env=_FFMPEG_ENV,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            return

        if result.returncode != 0 or not result.stdout:
            return

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        sample_fmt = lines[0] if len(lines) >= 1 else None
        try:
            sample_rate = int(lines[1]) if len(lines) >= 2 else None
        except ValueError:
            sample_rate = None
        try:
            channels = int(lines[2]) if len(lines) >= 3 else None
        except ValueError:
            channels = None

        # Check if conversion is needed
        needs_conversion = (
            sample_fmt not in {"s16", "s16p"}
            or channels not in {1, None}
            or sample_rate not in {16000, None}
        )
        if not needs_conversion:
            return

        debug_print(
            f"Normalising denoised audio to 16-bit PCM (fmt={sample_fmt}, sr={sample_rate}, ch={channels})",
            debug=debug,
        )

        source = Path(path)
        temp_path = source.with_name(f"{source.stem}_normalized{source.suffix}")

        ffmpeg_args = ["-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000"]
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "info" if debug else "error",
            "-nostdin", "-y", "-i", str(source), *ffmpeg_args, str(temp_path),
        ]

        try:
            subprocess.run(command, check=True, env=_FFMPEG_ENV,
                           stdout=subprocess.PIPE if not debug else None,
                           stderr=subprocess.PIPE if not debug else None, text=True)
            temp_path.replace(source)
        except subprocess.CalledProcessError as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            debug_print(f"Failed to normalise audio format: {exc.stderr or exc.stdout}", debug=debug)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    # Process audio (single file or chunked)
    if chunk_dir is None:
        run_denoiser(input_file, out_audio_file)
        normalize_audio(out_audio_file)
        return out_audio_file

    chunk_outputs = []
    try:
        for index, chunk_path in enumerate(chunks):
            update_sub_progress(index, len(chunks), "chunks")
            debug_print(f"Processing chunk {index + 1}/{len(chunks)}: {chunk_path}", debug=debug)
            chunk_output = str(chunk_dir / f"denoise_{index:05d}.wav")
            run_denoiser(chunk_path, chunk_output)
            chunk_outputs.append(chunk_output)
        update_sub_progress(len(chunks), len(chunks), "chunks")

        debug_print(f"Concatenating {len(chunk_outputs)} denoised chunk(s)", debug=debug)
        concatenate_audio(chunk_outputs, out_audio_file, chunk_dir)
        normalize_audio(out_audio_file)
        return out_audio_file
    finally:
        cleanup_chunks(chunk_dir)
