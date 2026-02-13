"""Video slicing module.

Extracts a time range from an input video file using FFmpeg stream-copy
for speed. Falls back to re-encoding when stream-copy fails and
``fallback_reencode`` is enabled in the config.

The sliced file is written to the output (temp) folder so that
downstream modules automatically pick it up.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Optional

from colorama import Fore, Style

from .utils.logging import info_print

if TYPE_CHECKING:
    from config import SliceConfig

__all__ = ["handle"]

_GRAY = Fore.LIGHTBLACK_EX

# HH:MM:SS, HH:MM:SS.mmm, MM:SS, MM:SS.mmm, or raw seconds (int/float)
_TIMESTAMP_RE = re.compile(
    r"^(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)$"
)

_env = os.environ.copy()
_env["AV_LOG_FORCE_NOCOLOR"] = "1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "SliceConfig | None" = None,
    *,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    debug: bool = False,
) -> str:
    """Slice a video file to the requested time range.

    Args:
        input_file: Path to the source video file.
        output_folder: Temp directory where the sliced file is written.
        config: Optional ``SliceConfig`` with *codec* and
            *fallback_reencode* overrides.
        start_time: Start timestamp (``HH:MM:SS[.mmm]``). ``None`` keeps
            the original start.
        end_time: End timestamp (``HH:MM:SS[.mmm]``). ``None`` keeps the
            original end.
        debug: Enable verbose FFmpeg output.

    Returns:
        Path to the sliced file, or *input_file* unchanged when no
        slicing is requested.

    Raises:
        RuntimeError: On invalid timestamps or FFmpeg failure.
    """
    # No-op when neither bound is provided
    if start_time is None and end_time is None:
        return input_file

    codec = config.codec if config else "copy"
    fallback_reencode = config.fallback_reencode if config else True

    start_sec = _parse_timestamp(start_time) if start_time else None
    end_sec = _parse_timestamp(end_time) if end_time else None
    _validate_range(start_sec, end_sec)

    # Validate against input duration
    duration = _probe_duration(input_file, debug=debug)
    if duration is not None:
        _validate_bounds(start_sec, end_sec, duration)

    os.makedirs(output_folder, exist_ok=True)

    ext = os.path.splitext(input_file)[1] or ".mp4"
    output_file = os.path.join(output_folder, f"sliced{ext}")

    info_print(
        f"Slicing video"
        f" from {start_time or 'start'}"
        f" to {end_time or 'end'}"
    )

    cmd = _build_ffmpeg_cmd(
        input_file, output_file,
        start_sec=start_sec, end_sec=end_sec,
        codec=codec, debug=debug,
    )

    try:
        _run_ffmpeg(cmd, debug=debug)
    except subprocess.CalledProcessError:
        if codec == "copy" and fallback_reencode:
            info_print("Stream-copy failed, retrying with re-encode")
            cmd = _build_ffmpeg_cmd(
                input_file, output_file,
                start_sec=start_sec, end_sec=end_sec,
                codec="reencode", debug=debug,
            )
            try:
                _run_ffmpeg(cmd, debug=debug)
            except subprocess.CalledProcessError as exc:
                _fatal(f"FFmpeg re-encode also failed (exit {exc.returncode})")
        else:
            _fatal("FFmpeg slice failed")

    if not os.path.exists(output_file):
        _fatal(f"Expected output file not found: {output_file}")

    return output_file


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str) -> float:
    """Parse a timestamp string into total seconds.

    Accepted formats:
      - ``HH:MM:SS`` / ``HH:MM:SS.mmm``
      - ``MM:SS`` / ``MM:SS.mmm``
      - Raw numeric seconds (``"90"`` or ``"12.5"``)
    """
    # Try raw numeric seconds first
    try:
        secs = float(value)
        if secs < 0:
            _fatal(f"Timestamp must not be negative: {value}")
        return secs
    except ValueError:
        pass

    m = _TIMESTAMP_RE.match(value)
    if not m:
        _fatal(
            f"Invalid timestamp format '{value}'. "
            "Expected HH:MM:SS, MM:SS, or raw seconds."
        )

    hours = int(m.group("h") or 0)
    minutes = int(m.group("m"))
    seconds = float(m.group("s"))
    return hours * 3600.0 + minutes * 60.0 + seconds


def _validate_range(
    start: Optional[float], end: Optional[float]
) -> None:
    """Ensure *start* < *end* when both are given."""
    if start is not None and end is not None and start >= end:
        _fatal(
            f"--slice-start ({_fmt_ts(start)}) must be before "
            f"--slice-end ({_fmt_ts(end)})"
        )


def _validate_bounds(
    start: Optional[float],
    end: Optional[float],
    duration: float,
) -> None:
    """Warn / fail if timestamps exceed input duration."""
    if start is not None and start >= duration:
        _fatal(
            f"--slice-start ({_fmt_ts(start)}) exceeds input duration "
            f"({_fmt_ts(duration)})"
        )
    if end is not None and end > duration:
        info_print(
            f"--slice-end ({_fmt_ts(end)}) exceeds input duration "
            f"({_fmt_ts(duration)}), will slice to end of file"
        )


def _fmt_ts(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------


def _probe_duration(input_file: str, *, debug: bool) -> Optional[float]:
    """Use *ffprobe* to obtain the duration of *input_file* in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=_env, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        if debug:
            info_print("Could not probe input duration — skipping bounds check")
        return None


def _build_ffmpeg_cmd(
    input_file: str,
    output_file: str,
    *,
    start_sec: Optional[float],
    end_sec: Optional[float],
    codec: str,
    debug: bool,
) -> list[str]:
    """Build the FFmpeg command list."""
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info" if debug else "error",
        "-nostdin",
        "-y",
    ]

    if start_sec is not None:
        cmd += ["-ss", str(start_sec)]

    cmd += ["-i", input_file]

    if end_sec is not None:
        duration = end_sec - (start_sec or 0.0)
        cmd += ["-t", str(duration)]

    if codec == "copy":
        cmd += ["-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart"]
    elif codec == "reencode":
        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
        ]
    else:
        cmd += ["-c", codec]

    cmd.append(output_file)
    return cmd


def _run_ffmpeg(command: list[str], *, debug: bool) -> None:
    """Execute an FFmpeg command, streaming output in debug mode."""
    if debug:
        proc = subprocess.Popen(
            command,
            env=_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line)
                if line:
                    print(f"{_GRAY}{line.rstrip()}{Style.RESET_ALL}")
            proc.wait()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        if proc.returncode:
            raise subprocess.CalledProcessError(
                proc.returncode, command, output="".join(lines),
            )
    else:
        subprocess.run(
            command, check=True, env=_env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _fatal(msg: str) -> None:
    """Print an error and raise."""
    print(f"\n{Fore.RED}Error: {msg}{Style.RESET_ALL}", file=sys.stderr)
    raise RuntimeError(msg)
