"""Audio download module.

Downloads audio from YouTube/web URLs using yt-dlp, prioritising the
best available audio stream. The downloaded file is saved to the
specified output folder and replaces the input path for all downstream
modules (slice, resample, etc.).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import yt_dlp

from .utils.logging import info_print, update_sub_progress

if TYPE_CHECKING:
    from config import DownloadConfig

__all__ = ["handle"]

DEFAULT_FILENAME_TEMPLATE = "%(title)s_%(id)s.%(ext)s"


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
    """Download audio from a URL.

    Args:
        input_file: URL to download (YouTube, etc.).
        output_folder: Directory where the file is saved.
        config: Optional ``DownloadConfig`` with *filename_template*
            override.
        debug: When *True*, enable verbose yt-dlp logging.

    Returns:
        Tuple of (absolute path to downloaded file, video title).

    Raises:
        RuntimeError: On download failure.
    """
    filename_template = (
        config.filename_template if config else DEFAULT_FILENAME_TEMPLATE
    )

    info_print("Downloading audio")

    return _download(
        input_file,
        output_folder,
        filename_template=filename_template,
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
    debug: bool,
) -> Tuple[str, str]:
    """Download media from *url* into *output_folder*.

    Returns:
        Tuple of (absolute path, video title).
    """
    absolute_output = os.path.abspath(output_folder)
    os.makedirs(absolute_output, exist_ok=True)

    options = _build_options(absolute_output, filename_template, debug)

    try:
        info_dict = _run_download(url, options)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc

    final_path = _resolve_path(info_dict)
    if not final_path or not os.path.exists(final_path):
        raise RuntimeError(
            "Download finished but the output file could not be determined"
        )

    title = info_dict.get("title", "")
    return os.path.abspath(final_path), title


def _build_options(
    output_folder: str,
    filename_template: str,
    debug: bool,
) -> dict:
    """Build yt-dlp options dict."""
    options: dict = {
        "format": "bestaudio/best",
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
