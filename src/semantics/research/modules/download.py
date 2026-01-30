from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yt_dlp

if TYPE_CHECKING:
    from ..config import DownloadConfig


DEFAULT_FILENAME_TEMPLATE = "%(title)s_%(id)s.%(ext)s"


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)
_VERBOSE = False
_RETRYABLE_ERROR_SNIPPETS = (
    "http error 403",
    "http error 429",
    "internal server error",
    "failed to resolve",
    "no address associated with hostname",
    "timed out",
    "connection reset",
)
_MAX_NETWORK_RETRIES = 3
_NETWORK_RETRY_BASE_DELAY = 2.0


def set_verbose(enabled: bool) -> None:
    """Adjust verbosity for this module and the embedded yt_dlp logger."""

    global _VERBOSE
    _VERBOSE = enabled

    level = logging.INFO if enabled else logging.WARNING
    logger.setLevel(level)

    yt_logger = logging.getLogger("yt_dlp")
    yt_logger.setLevel(logging.INFO if enabled else logging.ERROR)


def _make_progress_hook():
    """Create a progress hook that logs download progress when verbose."""

    def _hook(status: Dict[str, Any]) -> None:
        if not _VERBOSE:
            return
        if status.get("status") != "downloading":
            return

        percent = (status.get("_percent_str") or "").strip()
        speed = (status.get("_speed_str") or "?").strip()
        eta = (status.get("_eta_str") or "?").strip()

        logger.info("yt-dlp: %s | %s | eta %s", percent or "downloading", speed, eta)

    return _hook


def _make_postprocessor_hook():
    """Create a postprocessor hook to log merge status when verbose."""

    def _hook(status: Dict[str, Any]) -> None:
        if not _VERBOSE:
            return
        state = status.get("status")
        if not state:
            return

        postprocessor = status.get("postprocessor") or "unknown"
        destination = status.get("info_dict", {}).get("filepath")
        logger.info("yt-dlp: postprocessor %s (%s)", postprocessor, state)
        if destination:
            logger.info("yt-dlp: output -> %s", destination)

    return _hook


def _format_for_height(max_height: int | None) -> str:
    if max_height:
        return f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    return "bestvideo+bestaudio/best"


def _fallback_format_for_height(max_height: int | None) -> str:
    if max_height:
        return f"best[height<={max_height}]"
    return "best"


def _base_options(outtmpl: str, max_height: int | None) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "format": _format_for_height(max_height),
        "noplaylist": True,
        "outtmpl": outtmpl,
        # Enable Node.js as the JavaScript runtime for YouTube extraction
        "js_runtimes": {"node": {}},
        "progress_hooks": [_make_progress_hook()],
        "postprocessor_hooks": [_make_postprocessor_hook()],
        "retries": 5,
        "fragment_retries": 10,
        "continuedl": True,
        "skip_unavailable_fragments": False,
        "socket_timeout": 30,
        "forceipv4": True,
        "geo_bypass": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    if not _VERBOSE:
        options.update({"quiet": True, "no_warnings": True})

    return options


def _run_download(url: str, options: Dict[str, Any]) -> Dict[str, Any]:
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=True)


def _is_retryable_error(error: yt_dlp.utils.DownloadError) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in _RETRYABLE_ERROR_SNIPPETS)


def _run_download_with_retries(url: str, options: Dict[str, Any]) -> Dict[str, Any]:
    last_error: yt_dlp.utils.DownloadError | None = None

    for attempt in range(1, _MAX_NETWORK_RETRIES + 1):
        try:
            return _run_download(url, options)
        except yt_dlp.utils.DownloadError as exc:  # pragma: no cover - network variability
            last_error = exc

            if attempt >= _MAX_NETWORK_RETRIES or not _is_retryable_error(exc):
                raise

            backoff = _NETWORK_RETRY_BASE_DELAY * attempt
            logger.warning("Retryable yt-dlp error (attempt %s/%s): %s", attempt, _MAX_NETWORK_RETRIES, exc)
            logger.info("Sleeping %.1fs before retrying", backoff)
            time.sleep(backoff)

    if last_error:
        raise last_error

    raise yt_dlp.utils.DownloadError("Failed to download video; retries exhausted")


def _resolve_final_path(info_dict: Dict[str, Any], output_directory: str) -> str | None:
    candidates: list[str] = []

    for key in ("filepath", "filename", "_filename"):
        value = info_dict.get(key)
        if isinstance(value, str):
            candidates.append(value)

    requested = info_dict.get("requested_downloads")
    if isinstance(requested, list):
        for entry in requested:
            if not isinstance(entry, dict):
                continue
            for key in ("filepath", "filename", "_filename"):
                value = entry.get(key)
                if isinstance(value, str):
                    candidates.append(value)

    for path in candidates:
        absolute = os.path.abspath(path)
        if os.path.exists(absolute):
            return absolute

        joined = os.path.abspath(os.path.join(output_directory, path))
        if os.path.exists(joined):
            return joined

    if candidates:
        return os.path.abspath(os.path.join(output_directory, candidates[0]))

    return None


def download_video(
    url: str,
    output_folder: str,
    filename_template: str = DEFAULT_FILENAME_TEMPLATE,
    max_height: int | None = 720,
) -> str | None:
    """Download a video using yt_dlp and return the absolute output path."""

    if not url:
        raise ValueError("url must be provided")
    if not output_folder:
        raise ValueError("output_folder must be provided")

    absolute_output = os.path.abspath(output_folder)

    try:
        os.makedirs(absolute_output, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to create output directory '%s': %s", absolute_output, exc)
        return None

    outtmpl = os.path.join(absolute_output, filename_template)
    options = _base_options(outtmpl, max_height)

    try:
        info_dict = _run_download_with_retries(url, options)
    except yt_dlp.utils.DownloadError as exc:
        logger.warning("Primary download failed: %s", exc)
        fallback_options = options.copy()
        fallback_options["format"] = _fallback_format_for_height(max_height)
        logger.info("Retrying download with fallback format selector: %s", fallback_options["format"])

        try:
            info_dict = _run_download_with_retries(url, fallback_options)
        except yt_dlp.utils.DownloadError as fallback_exc:
            logger.warning("Fallback download failed: %s", fallback_exc)
            final_options = options.copy()
            final_options["format"] = "best"
            logger.info("Retrying download with final fallback format selector: best")

            try:
                info_dict = _run_download_with_retries(url, final_options)
            except yt_dlp.utils.DownloadError as final_exc:
                logger.error("yt-dlp download error: %s", final_exc)
                return None
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.error("Unexpected error during download: %s", exc, exc_info=_VERBOSE)
        return None

    if not info_dict:
        logger.error("yt-dlp returned no information for url: %s", url)
        return None

    final_path = _resolve_final_path(info_dict, absolute_output)
    if final_path:
        logger.info("Download complete: %s", final_path)
    else:
        logger.warning("Download finished but the output path could not be determined.")

    return final_path


def handle(
    urls: List[str],
    output_folder: str,
    config: "DownloadConfig | None" = None,
    *,
    debug: bool = False,
) -> List[str]:
    """Main entry point for video downloads.

    Args:
        urls: List of video URLs to download.
        output_folder: Directory for output files.
        config: DownloadConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        List of paths to downloaded video files.
    """
    set_verbose(debug)

    filename_template = config.filename_template if config else DEFAULT_FILENAME_TEMPLATE
    max_height = config.max_height if config else 720

    downloaded_paths: List[str] = []
    for url in urls:
        result_path = download_video(url, output_folder, filename_template=filename_template, max_height=max_height)
        if result_path:
            downloaded_paths.append(result_path)

    return downloaded_paths
