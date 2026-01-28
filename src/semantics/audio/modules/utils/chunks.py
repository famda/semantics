import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

CHUNK_LENGTH_SECONDS = 900

_env = os.environ.copy()
_env["AV_LOG_FORCE_NOCOLOR"] = "1"


@dataclass
class AudioChunk:
    path: str
    start: float
    end: float
    core_start: float
    core_end: float


def _ensure_chunk_dir(base_dir: Path, name: str) -> Path:
    chunk_dir = base_dir / f"._chunks_{name}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for existing in chunk_dir.glob("*.wav"):
        try:
            existing.unlink()
        except Exception:
            pass

    list_file = chunk_dir / "concat_list.txt"
    if list_file.exists():
        try:
            list_file.unlink()
        except Exception:
            pass

    return chunk_dir


def split_audio(
    audio_file: str,
    temp_folder: str,
    chunk_name: str,
    chunk_length: int = CHUNK_LENGTH_SECONDS,
) -> Tuple[List[str], Optional[Path]]:
    """Split an audio file into fixed-length chunks using ffmpeg."""
    if chunk_length <= 0:
        return [audio_file], None

    base_dir = Path(temp_folder)
    chunk_dir = _ensure_chunk_dir(base_dir, chunk_name)
    pattern = str(chunk_dir / f"{chunk_name}_%05d.wav")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        audio_file,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_length),
        "-ar",
        "16000",
        "-ac",
        "1",
        pattern,
    ]

    try:
        subprocess.run(command, check=True, env=_env)
    except FileNotFoundError:
        cleanup_chunks(chunk_dir)
        return [audio_file], None
    except subprocess.CalledProcessError:
        cleanup_chunks(chunk_dir)
        return [audio_file], None

    chunk_files = sorted(chunk_dir.glob(f"{chunk_name}_*.wav"))
    if not chunk_files:
        cleanup_chunks(chunk_dir)
        return [audio_file], None

    return [str(path) for path in chunk_files], chunk_dir


def split_audio_with_overlap(
    audio_file: str,
    temp_folder: str,
    chunk_name: str,
    chunk_length: int = CHUNK_LENGTH_SECONDS,
    overlap_seconds: float = 5.0,
) -> Tuple[List[AudioChunk], Optional[Path]]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be greater than zero when using overlap")

    overlap = max(0.0, float(overlap_seconds))

    try:
        import torchaudio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchaudio is required for overlapped chunking"
        ) from exc

    audio_info = torchaudio.info(audio_file)
    if not audio_info.sample_rate:
        raise RuntimeError("Unable to determine sample rate for audio file")

    total_duration = audio_info.num_frames / audio_info.sample_rate
    if total_duration <= 0:
        return [
            AudioChunk(
                path=audio_file,
                start=0.0,
                end=0.0,
                core_start=0.0,
                core_end=0.0,
            )
        ], None

    base_dir = Path(temp_folder)
    chunk_dir = _ensure_chunk_dir(base_dir, chunk_name)

    chunk_infos: List[AudioChunk] = []
    core_start = 0.0
    index = 0

    while core_start < total_duration:
        core_end = min(total_duration, core_start + float(chunk_length))
        chunk_start = max(0.0, core_start - overlap if index > 0 else 0.0)
        chunk_end = min(
            total_duration,
            core_end + overlap if core_end < total_duration else total_duration,
        )

        duration = max(0.0, chunk_end - chunk_start)
        if duration <= 0:
            break

        output_path = chunk_dir / f"{chunk_name}_{index:05d}.wav"

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{chunk_start:.6f}",
            "-i",
            audio_file,
            "-t",
            f"{duration:.6f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]

        try:
            subprocess.run(command, check=True, env=_env)
        except FileNotFoundError:
            cleanup_chunks(chunk_dir)
            return [
                AudioChunk(
                    path=audio_file,
                    start=0.0,
                    end=total_duration,
                    core_start=0.0,
                    core_end=total_duration,
                )
            ], None
        except subprocess.CalledProcessError:
            cleanup_chunks(chunk_dir)
            return [
                AudioChunk(
                    path=audio_file,
                    start=0.0,
                    end=total_duration,
                    core_start=0.0,
                    core_end=total_duration,
                )
            ], None

        chunk_infos.append(
            AudioChunk(
                path=str(output_path),
                start=chunk_start,
                end=chunk_end,
                core_start=core_start,
                core_end=core_end,
            )
        )

        core_start = core_end
        index += 1

    return chunk_infos, chunk_dir


def concatenate_audio(
    chunk_files: Sequence[str],
    output_file: str,
    working_dir: Optional[Path] = None,
) -> str:
    """Concatenate chunk files into a single audio output using ffmpeg."""
    if not chunk_files:
        raise ValueError("No chunk files provided for concatenation")

    first_chunk = Path(chunk_files[0])
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if len(chunk_files) == 1:
        if first_chunk.resolve() != destination.resolve():
            shutil.copyfile(first_chunk, destination)
        return output_file

    work_dir = working_dir or destination.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    list_file = work_dir / "concat_list.txt"

    with list_file.open("w", encoding="utf-8") as f:
        for chunk in chunk_files:
            safe_path = Path(chunk).as_posix().replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        output_file,
    ]

    try:
        subprocess.run(command, check=True, env=_env)
    finally:
        try:
            list_file.unlink()
        except Exception:
            pass

    return output_file


def cleanup_chunks(chunk_dir: Optional[Path]) -> None:
    if not chunk_dir:
        return

    for chunk in chunk_dir.glob("*.wav"):
        try:
            chunk.unlink()
        except Exception:
            pass

    try:
        chunk_dir.rmdir()
    except Exception:
        pass


def compute_chunk_offsets(chunk_files: Sequence[str]) -> List[float]:
    try:
        import torchaudio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torchaudio is required to compute chunk offsets") from exc

    offsets: List[float] = []
    offset = 0.0

    for chunk in chunk_files:
        offsets.append(offset)
        info = torchaudio.info(chunk)
        if info.sample_rate:
            offset += info.num_frames / info.sample_rate

    return offsets
