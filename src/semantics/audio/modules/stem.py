"""Source separation module using Demucs for vocals extraction.

This module provides functionality to separate vocals from audio files
using Facebook's Demucs model. It supports chunked processing for long
audio files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from colorama import Fore, Style, init

from .utils.chunks import cleanup_chunks, concatenate_audio, split_audio

if TYPE_CHECKING:
    from ..config import StemConfig

_ENV = os.environ.copy()
_ENV["AV_LOG_FORCE_NOCOLOR"] = "1"
_GRAY = Fore.LIGHTBLACK_EX

init()


def handle(
    input_file: str,
    output_folder: str,
    config: "StemConfig | None" = None,
    *,
    debug: bool = False,
) -> str:
    """Perform source separation using Demucs.

    Args:
        input_file: Path to the input audio file.
        output_folder: Directory where output files will be written.
        config: StemConfig instance with separation parameters, or None for defaults.
        debug: If True, emit verbose debug output.

    Returns:
        Path to the extracted vocals audio file, or the original file on failure.
    """
    # Extract config values (use defaults if config is None)
    chunk_length = config.chunk_length if config else 900
    model = config.model if config else "htdemucs_ft"
    two_stems = config.two_stems if config else "vocals"

    print("INFO: Performing source separation")

    temp_path = Path(output_folder)
    temp_path.mkdir(parents=True, exist_ok=True)
    stem_dir = temp_path / "stem"

    chunks, chunk_dir = split_audio(input_file, output_folder, "stem", chunk_length)

    def run_command(command: List[str]) -> Optional[subprocess.CompletedProcess]:
        """Execute command with optional debug output streaming."""
        if debug:
            process = subprocess.Popen(
                command, env=_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            combined: List[str] = []
            progress_rendered = False
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    combined.append(line)
                    if line:
                        # Handle progress bar output
                        payload = line.rstrip("\r\n")
                        is_progress = "%|" in payload and payload.lstrip().startswith(tuple("0123456789"))
                        if is_progress:
                            sys.stdout.write(f"{_GRAY}{payload}{Style.RESET_ALL}\r")
                            progress_rendered = True
                        elif line.endswith("\n"):
                            sys.stdout.write(f"{_GRAY}{line[:-1]}{Style.RESET_ALL}\n")
                        else:
                            sys.stdout.write(f"{_GRAY}{line}{Style.RESET_ALL}")
                        sys.stdout.flush()
                process.wait()
                if progress_rendered:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            finally:
                if process.stdout:
                    process.stdout.close()

            if process.returncode:
                error = subprocess.CalledProcessError(process.returncode, command, output="".join(combined))
                error.stdout = "".join(combined)
                raise error
            return None

        return subprocess.run(command, check=True, env=_ENV, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)

    def run_demucs(input_path: Path, output_root: Path) -> Optional[Path]:
        """Run Demucs with GPU→CPU fallback, return output directory or None."""
        for device in ("cuda", "cpu"):
            command = [
                "audio", "-m", "demucs.separate", "-n", model,
                f"--two-stems={two_stems}", str(input_path),
                "-o", str(output_root), "--device", device,
            ]

            htdemucs_root = output_root / model
            if htdemucs_root.exists():
                shutil.rmtree(htdemucs_root, ignore_errors=True)

            try:
                run_command(command)
            except subprocess.CalledProcessError as exc:
                print(f"Command failed with return code {exc.returncode}")
                if not debug:
                    output_text = getattr(exc, "stdout", None) or getattr(exc, "output", None)
                    if output_text:
                        print(f"Output:\n{output_text}")
                if device == "cuda":
                    print("WARN: Demucs GPU execution failed; retrying on CPU.")
                    continue
                print("Error occurred during vocal separation. Using the original audio file.")
                return None
            except Exception as exc:
                print(f"An error occurred: {exc}")
                if device == "cuda":
                    print("WARN: Demucs GPU execution failed; retrying on CPU.")
                    continue
                print("Error occurred during vocal separation. Using the original audio file.")
                return None

            demucs_output = output_root / model / input_path.stem
            if not demucs_output.exists():
                print("WARN: Demucs output not found. Using the original audio file.")
                if device == "cuda":
                    print("WARN: Retrying Demucs on CPU.")
                    continue
                return None

            return demucs_output
        return None

    def ensure_mono_16k(audio_path: Path) -> str:
        """Force WAV to mono/16kHz in-place."""
        if not audio_path.exists():
            return str(audio_path)

        tmp_path = audio_path.with_suffix(".tmp.wav")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "info" if debug else "error",
            "-nostdin", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(tmp_path),
        ]

        try:
            run_command(command)
        except subprocess.CalledProcessError as exc:
            print(f"WARN: Failed to downmix stem to mono/16kHz (code {exc.returncode}); keeping original track.")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return str(audio_path)
        except FileNotFoundError:
            print("WARN: ffmpeg not available to normalize stem audio; keeping original track.")
            return str(audio_path)

        try:
            shutil.move(tmp_path, audio_path)
        except Exception as exc:
            print(f"WARN: Failed to replace stem audio with normalized track: {exc}")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return str(audio_path)

        return str(audio_path)

    def finalize_single_run(demucs_output: Path) -> str:
        """Copy single-file Demucs output to stem directory."""
        if stem_dir.exists():
            shutil.rmtree(stem_dir, ignore_errors=True)

        try:
            shutil.copytree(demucs_output, stem_dir)
        except Exception as exc:
            print(f"WARN: Failed to prepare stem directory: {exc}")
            return input_file
        finally:
            htdemucs_dir = temp_path / model
            if htdemucs_dir.exists():
                shutil.rmtree(htdemucs_dir, ignore_errors=True)

        vocals_path = stem_dir / f"{two_stems}.wav"
        if not vocals_path.exists():
            print("Vocals file not found after separation. Using the original audio file.")
            return input_file

        ensure_mono_16k(vocals_path)
        return str(vocals_path)

    # Process audio (single file or chunked)
    try:
        if chunk_dir is None or len(chunks) <= 1:
            demucs_output = run_demucs(Path(input_file), temp_path)
            if demucs_output:
                return finalize_single_run(demucs_output)
            return input_file

        print(f"INFO: Chunking stems into {len(chunks)} segments of up to {chunk_length} seconds")

        if stem_dir.exists():
            shutil.rmtree(stem_dir, ignore_errors=True)
        stem_dir.mkdir(parents=True, exist_ok=True)

        stem_chunks: Dict[str, List[str]] = {}

        for index, chunk_path_str in enumerate(chunks):
            chunk_path = Path(chunk_path_str)
            print(f"INFO: Processing stem chunk {index + 1}/{len(chunks)}: {chunk_path}")

            with tempfile.TemporaryDirectory(dir=output_folder) as demucs_temp:
                demucs_output = run_demucs(chunk_path, Path(demucs_temp))
                if not demucs_output:
                    raise RuntimeError(f"Demucs failed on chunk {chunk_path.name}")

                for stem_file in demucs_output.glob("*.wav"):
                    stem_name = stem_file.name
                    chunk_output = Path(chunk_dir) / f"{stem_name}_{index:05d}.wav"
                    shutil.copy2(stem_file, chunk_output)
                    stem_chunks.setdefault(stem_name, []).append(str(chunk_output))

        if not stem_chunks:
            raise RuntimeError("Demucs did not produce any stems to merge.")

        for stem_name, files in stem_chunks.items():
            concatenate_audio(files, str(stem_dir / stem_name), chunk_dir)

        vocals_path = stem_dir / f"{two_stems}.wav"
        if vocals_path.exists():
            ensure_mono_16k(vocals_path)
            return str(vocals_path)

        print("WARN: Vocals file not found after chunked separation. Using the original audio file.")
        return input_file

    except Exception as exc:
        print(f"WARN: Chunked separation failed: {exc}")
        if stem_dir.exists():
            shutil.rmtree(stem_dir, ignore_errors=True)
        return input_file
    finally:
        cleanup_chunks(chunk_dir)