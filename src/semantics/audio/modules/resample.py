"""Audio resampling and format conversion module.

Converts input audio/video files to a standardized WAV format suitable for
downstream processing (mono, 16kHz by default).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

from colorama import Fore, Style, init

from .utils.logging import info_print

if TYPE_CHECKING:
    from config import ResampleConfig, EnhanceConfig

env = os.environ.copy()
env["AV_LOG_FORCE_NOCOLOR"] = "1"
GRAY = Fore.LIGHTBLACK_EX

__all__ = ["handle", "enhance"]

init()


def _run_ffmpeg(command, *, debug: bool) -> None:
    if debug:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        try:
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line)
                if line:
                    print(f"{GRAY}{line.rstrip()}{Style.RESET_ALL}")
            process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                output="".join(output_lines),
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


def handle(
    input_file: str,
    output_folder: str,
    config: "ResampleConfig | None" = None,
    *,
    debug: bool = False,
) -> str:
    """Resample and convert audio to standardized format.

    Args:
        input_file: Path to the input audio/video file.
        output_folder: Path to the output folder for processed files.
        config: ResampleConfig with sample_rate and channels settings.
        debug: Enable verbose logging.

    Returns:
        Path to the resampled audio file.
    """
    info_print("Converting and resampling the audio file")

    # Use config values or defaults
    sample_rate = config.sample_rate if config else 16000
    channels = config.channels if config else 1

    # Ensure the temp folder exists and is writable
    os.makedirs(output_folder, exist_ok=True)
    if not os.access(output_folder, os.W_OK):
        print(Style.RESET_ALL)
        print(f"Temp folder is not writable: {output_folder}")
        sys.exit(1)

    output_file = os.path.join(output_folder, "audio.wav")
    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if debug else "error",
        "-nostdin",
        "-threads", str(os.cpu_count() or 4),
        "-y",
        "-i", input_file,
        "-ac", str(channels),
        "-ar", str(sample_rate),
        output_file
    ]

    try:
        _run_ffmpeg(ffmpeg_command, debug=debug)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        stdout_text = getattr(e, "stdout", None) or getattr(e, "output", None)
        stderr_text = getattr(e, "stderr", None)
        if not debug:
            if stdout_text:
                print(f"ffmpeg stdout:\n{stdout_text}")
            if stderr_text:
                print(f"ffmpeg stderr:\n{stderr_text}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    return output_file


def enhance(
    input_file: str,
    output_folder: str,
    config: "EnhanceConfig | None" = None,
    *,
    debug: bool = False,
) -> str:
    """Enhance audio quality using ffmpeg filters.

    Args:
        input_file: Path to the input audio file.
        output_folder: Path to the output folder for processed files.
        config: EnhanceConfig with enhancement settings.
        debug: Enable verbose logging.

    Returns:
        Path to the enhanced audio file (or original on failure).
    """
    info_print("Enhancing the audio quality")

    output_file = os.path.join(output_folder, "enhanced.wav")
    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info" if debug else "error",
        "-threads", str(os.cpu_count() or 4),
        "-y",
        "-i", input_file,
        "-af", "afftdn,highpass=f=200,compand,acompressor,loudnorm,equalizer=f=300:t=q:w=2:g=3,equalizer=f=3000:t=q:w=2:g=3",
        "-ac", "1",
        "-ar", "16000",
        output_file,
    ]

    try:
        _run_ffmpeg(ffmpeg_command, debug=debug)
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        stdout_text = getattr(e, "stdout", None) or getattr(e, "output", None)
        stderr_text = getattr(e, "stderr", None)
        if not debug:
            if stdout_text:
                print(f"ffmpeg stdout:\n{stdout_text}")
            if stderr_text:
                print(f"ffmpeg stderr:\n{stderr_text}")
        print("Error occurred during audio enhancement. Using the original audio file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Error occurred during audio enhancement. Using the original audio file.")

    return input_file