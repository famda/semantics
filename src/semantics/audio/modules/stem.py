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
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

from colorama import Fore, Style, init

from .utils.chunks import cleanup_chunks, concatenate_audio, split_audio

if TYPE_CHECKING:
    from ..config import StemConfig

_ENV = os.environ.copy()
_ENV["AV_LOG_FORCE_NOCOLOR"] = "1"

__all__ = ["handle"]
_GRAY = Fore.LIGHTBLACK_EX

init()


def detect_audio_segments(
    audio_path: Path, 
    label: str, 
    onset_db: float = -45.0, 
    offset_db: float = -60.0, 
    min_duration: float = 0.2, 
    smooth_window: float = 0.1
) -> List[Dict]:
    """
    Detects active segments in a Demucs stem using hysteresis thresholding.
    
    Args:
        audio_path: Path to the .wav file.
        label: Label for the JSON ('music' or 'vocals').
        onset_db: volume (dB) required to START a segment.
        offset_db: volume (dB) required to STOP a segment (allows trailing fade-outs).
        min_duration: minimum seconds to count as a segment.
        smooth_window: window size in seconds to smooth out clicks/pops.
    """
    if not audio_path.exists():
        return []

    try:
        # Read WAV file
        sample_rate, data = wavfile.read(str(audio_path))
        
        # Normalize to float32 (-1.0 to 1.0)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Mix to mono for detection
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # RMS Calculation Settings
        hop_length = int(sample_rate * 0.01) # 10ms steps
        
        # Pad data
        pad_needed = hop_length - (len(data) % hop_length)
        if pad_needed < hop_length:
            data = np.pad(data, (0, pad_needed))
        
        # Reshape to frames
        frames = data.reshape(-1, hop_length)
        
        # Calculate Amplitude Envelope (Root Mean Square)
        envelope = np.sqrt(np.mean(frames**2, axis=1))
        
        # Convert to dB
        db_env = 20 * np.log10(envelope + 1e-9)

        # Smooth signal to remove rapid clicks/artifacts
        window_size = int(smooth_window * sample_rate / hop_length)
        if window_size > 1:
            db_env = uniform_filter1d(db_env, size=window_size)

        # Hysteresis Logic
        segments = []
        is_active = False
        start_frame = 0
        frame_duration = hop_length / sample_rate

        for i, db_val in enumerate(db_env):
            if not is_active:
                if db_val > onset_db:
                    is_active = True
                    start_frame = i
            else:
                if db_val < offset_db:
                    is_active = False
                    duration = (i - start_frame) * frame_duration
                    if duration >= min_duration:
                        segments.append({
                            "start": round(start_frame * frame_duration, 3),
                            "end": round(i * frame_duration, 3),
                            "label": label
                        })

        # Handle end of file
        if is_active:
            duration = (len(db_env) - start_frame) * frame_duration
            if duration >= min_duration:
                segments.append({
                    "start": round(start_frame * frame_duration, 3),
                    "end": round(len(db_env) * frame_duration, 3),
                    "label": label
                })

        return segments

    except Exception as e:
        print(f"WARN: Error analyzing {label} segments: {e}")
        return []


def handle(
    input_file: str,
    output_folder: str,
    config: "StemConfig | None" = None,
    *,
    debug: bool = False,
) -> str:
    """Perform source separation using Demucs and generate timestamp JSON."""
    
    chunk_length = config.chunk_length if config else 900
    model = config.model if config else "htdemucs_ft"
    two_stems = config.two_stems if config else "vocals"
    shifts = config.shifts if config else 0
    overlap = config.overlap if config else 0.1

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
        """Run Demucs with GPU->CPU fallback."""
        for device in ("cuda", "cpu"):
            command = [
                "audio", "-m", "demucs.separate", "-n", model,
                f"--two-stems={two_stems}", str(input_path),
                "-o", str(output_root), "--device", device,
                "--shifts", str(shifts),
                "--overlap", str(overlap),
            ]
            htdemucs_root = output_root / model
            if htdemucs_root.exists():
                shutil.rmtree(htdemucs_root, ignore_errors=True)

            try:
                run_command(command)
            except subprocess.CalledProcessError as exc:
                print(f"Command failed with return code {exc.returncode}")
                if device == "cuda":
                    print("WARN: Demucs GPU execution failed; retrying on CPU.")
                    continue
                return None
            except Exception as exc:
                print(f"An error occurred: {exc}")
                if device == "cuda":
                    print("WARN: Demucs GPU execution failed; retrying on CPU.")
                    continue
                return None

            demucs_output = output_root / model / input_path.stem
            if not demucs_output.exists():
                if device == "cuda": continue
                return None
            return demucs_output
        return None

    def ensure_mono_16k(audio_path: Path) -> str:
        """Force WAV to mono/16kHz in-place."""
        if not audio_path.exists(): return str(audio_path)
        tmp_path = audio_path.with_suffix(".tmp.wav")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "info" if debug else "error",
            "-nostdin", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(tmp_path),
        ]
        try:
            run_command(command)
            shutil.move(tmp_path, audio_path)
        except Exception:
            if tmp_path.exists(): tmp_path.unlink(missing_ok=True)
        return str(audio_path)

    def generate_json_and_finalize(final_output_path: Path) -> str:
        """Analyze stems for timestamps and return the final vocals path."""
        vocals_path = stem_dir / f"{two_stems}.wav"
        music_path = stem_dir / "no_vocals.wav" # Demucs default name for remainder
        
        # 1. Normalize Audio
        if vocals_path.exists(): ensure_mono_16k(vocals_path)
        if music_path.exists(): ensure_mono_16k(music_path)
        
        # 2. Detect Segments
        print("INFO: Analyzing stems for start/end timestamps")
        segments = []
        
        # Detect Vocals (Sensitivity: Medium)
        # Onset -45dB: Ignores breath noise. Offset -60dB: Catches whisper endings.
        segments.extend(detect_audio_segments(
            vocals_path, "vocals", onset_db=-45.0, offset_db=-60.0, min_duration=0.2
        ))

        # Detect Music (Sensitivity: Low)
        # Onset -40dB: Ignores tape hiss. Offset -50dB: Strict cutoff.
        segments.extend(detect_audio_segments(
            music_path, "music", onset_db=-40.0, offset_db=-50.0, min_duration=0.5
        ))
        
        # 3. Sort and Save JSON
        segments.sort(key=lambda x: x["start"])
        json_path = stem_dir / "stem.json"
        
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2)
        except Exception as e:
            print(f"WARN: Failed to save stem.json: {e}")

        # Return path to vocals
        if vocals_path.exists():
            return str(vocals_path)
        
        print("WARN: Vocals file missing. Returning original.")
        return input_file

    # --- Main Execution Logic ---
    try:
        # Case A: No chunking needed
        if chunk_dir is None or len(chunks) <= 1:
            demucs_output = run_demucs(Path(input_file), temp_path)
            if demucs_output:
                # Prepare stem dir
                if stem_dir.exists(): shutil.rmtree(stem_dir)
                shutil.copytree(demucs_output, stem_dir)
                # Cleanup Demucs raw folder
                shutil.rmtree(temp_path / model, ignore_errors=True)
                
                return generate_json_and_finalize(stem_dir)
            return input_file

        # Case B: Chunking needed
        print(f"INFO: Chunking stems into {len(chunks)} segments...")
        if stem_dir.exists(): shutil.rmtree(stem_dir)
        stem_dir.mkdir(parents=True, exist_ok=True)
        stem_chunks: Dict[str, List[str]] = {}

        for index, chunk_path_str in enumerate(chunks):
            chunk_path = Path(chunk_path_str)
            print(f"INFO: Processing stem chunk {index + 1}/{len(chunks)}")
            with tempfile.TemporaryDirectory(dir=output_folder) as demucs_temp:
                demucs_output = run_demucs(chunk_path, Path(demucs_temp))
                if not demucs_output: raise RuntimeError("Demucs failed on chunk")
                for stem_file in demucs_output.glob("*.wav"):
                    chunk_output = Path(chunk_dir) / f"{stem_file.name}_{index:05d}.wav"
                    shutil.copy2(stem_file, chunk_output)
                    stem_chunks.setdefault(stem_file.name, []).append(str(chunk_output))

        for stem_name, files in stem_chunks.items():
            concatenate_audio(files, str(stem_dir / stem_name), chunk_dir)

        return generate_json_and_finalize(stem_dir)

    except Exception as exc:
        print(f"WARN: Process failed: {exc}")
        return input_file
    finally:
        cleanup_chunks(chunk_dir)