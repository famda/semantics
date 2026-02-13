"""Video download module.

Downloads video from YouTube/web URLs using yt-dlp, respecting a
configurable maximum resolution. The downloaded file is saved to the
specified output folder and replaces the input path for all downstream
modules.

Quality labels (height -> resolution):
    Full HD (1080p) -- 1920x1080
    HD     ( 720p) -- 1280x720
    SD     ( 480p) --  854x480
    SD     ( 360p) --  640x360   <- default
    SD     ( 240p) --  426x240
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Optional, Tuple

import yt_dlp

from .utils.logging import info_print, update_sub_progress

if TYPE_CHECKING:
    from config import DownloadConfig

__all__ = ["handle"]

DEFAULT_FILENAME_TEMPLATE = "%(title)s_%(id)s.%(ext)s"
DEFAULT_MAX_HEIGHT = 480


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "DownloadConfig | None" = None,
    *,
    debug: bool = False,
) -> Tuple[str, str]:
    """Download a video from a URL.

    Args:
        input_file: URL to download (YouTube, etc.).
        output_folder: Directory where the file is saved.
        config: Optional ``DownloadConfig`` with *max_height* and
            *filename_template* overrides.
        debug: When *True*, enable verbose yt-dlp logging.

    Returns:
        Tuple of (absolute path to downloaded file, video title).

    Raises:
        RuntimeError: On download failure.
    """
    max_height = config.max_height if config else DEFAULT_MAX_HEIGHT
    filename_template = (
        config.filename_template if config else DEFAULT_FILENAME_TEMPLATE
    )

    info_print(f"Downloading video ({max_height}p max)")

    return _download(
        input_file,
        output_folder,
        filename_template=filename_template,
        max_height=max_height,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download(
    url: str,
    output_folder: str,
    *,
    filename_template: str,
    max_height: int,
    debug: bool,
) -> Tuple[str, str]:
    """Download media from *url* into *output_folder*.

    Returns:
        Tuple of (absolute path, video title).
    """
    absolute_output = os.path.abspath(output_folder)
    os.makedirs(absolute_output, exist_ok=True)

    options = _build_options(absolute_output, filename_template, max_height, debug)

    # Attempt 1 — DASH (separate video+audio, merged by ffmpeg)
    try:
        info_dict = _run_download(url, options)
    except yt_dlp.utils.DownloadError:
        # Attempt 2 — progressive stream (single file, lower quality).
        # Falls back when DASH fails (e.g. restricted videos, geo-blocks).
        info_print("Retrying with progressive format")
        options["format"] = _progressive_format(max_height)
        try:
            info_dict = _run_download(url, options)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Download failed: {exc}") from exc

    final_path = _resolve_path(info_dict)
    if not final_path or not os.path.exists(final_path):
        raise RuntimeError(
            "Download finished but the output file could not be determined"
        )

    # Convert to MP4 when the container is not already .mp4.
    # VP9/webm containers cause issues with downstream frame extraction
    # (OpenCV seek, segment export, etc.).  Similar to how the audio CLI
    # resamples after download, we re-mux/transcode here.
    final_path = _ensure_mp4(final_path, debug=debug)

    title = info_dict.get("title", "")
    return os.path.abspath(final_path), title


def _ensure_mp4(path: str, *, debug: bool) -> str:
    """Re-encode *path* into an H.264/AAC MP4 when the extension differs.

    VP9/webm containers (and VP9 stream-copied into MP4) cause broken
    frame-level seeking in OpenCV, which silently breaks every downstream
    module that uses ``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, …)``.
    We therefore **always** re-encode to H.264 + AAC to guarantee
    reliable seeking.

    The original file is kept alongside the new ``.mp4`` copy.  If the
    file is already ``.mp4`` no work is performed.

    Returns:
        Absolute path to the ``.mp4`` file.
    """
    root, ext = os.path.splitext(path)
    if ext.lower() == ".mp4":
        return path

    mp4_path = root + ".mp4"
    info_print("Re-encoding to H.264 MP4")

    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i", path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        mp4_path,
    ]
    if not debug:
        cmd.insert(2, "-loglevel")
        cmd.insert(3, "error")

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    if not os.path.exists(mp4_path):
        # Conversion failed silently — return original
        return path

    return mp4_path


def _build_options(
    output_folder: str,
    filename_template: str,
    max_height: int,
    debug: bool,
) -> dict:
    """Build yt-dlp options dict."""
    fmt = (
        f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
        if max_height
        else "bestvideo+bestaudio/best"
    )

    options: dict = {
        "format": fmt,
        "noplaylist": True,
        "outtmpl": os.path.join(output_folder, filename_template),
        "js_runtimes": {"node": {}},
        "progress_hooks": [_make_progress_hook()],
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if not debug:
        options["quiet"] = True
        options["no_warnings"] = True
        options["noprogress"] = True

    return options


def _make_progress_hook():
    """Create a yt-dlp progress hook that feeds ``update_sub_progress``."""

    def _hook(d: dict) -> None:
        if d.get("status") != "downloading":
            if d.get("status") == "finished":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                if total:
                    update_sub_progress(int(total // 1024), int(total // 1024), "KB")
            return
        downloaded = d.get("downloaded_bytes", 0) or 0
        total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        if total > 0:
            update_sub_progress(
                int(downloaded // 1024),
                int(total // 1024),
                "KB",
            )

    return _hook


def _progressive_format(max_height: int) -> str:
    """Progressive-only format selector (single-file, no merge needed)."""
    if max_height:
        return f"best[height<={max_height}]"
    return "best"


def _run_download(url: str, options: dict) -> dict:
    """Execute yt-dlp download and return the info dict."""
    with yt_dlp.YoutubeDL(options) as ydl:
        result = ydl.extract_info(url, download=True)
        if result is None:
            raise RuntimeError("yt-dlp returned no information")
        return result


def _resolve_path(info_dict: dict) -> Optional[str]:
    """Extract the final file path from a yt-dlp info dict."""
    path = info_dict.get("filepath")
    if path and os.path.exists(path):
        return path

    requested = info_dict.get("requested_downloads")
    if requested and isinstance(requested, list):
        for entry in requested:
            if not isinstance(entry, dict):
                continue
            for key in ("filepath", "filename", "_filename"):
                candidate = entry.get(key)
                if candidate and os.path.exists(candidate):
                    return candidate

    return None
