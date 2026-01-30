import yt_dlp
import sys
import os
import logging
import argparse
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import DownloadConfig

__all__ = ["handle"]

# --- Configuration ---
DEFAULT_FILENAME_TEMPLATE = '%(title)s_%(id)s.%(ext)s'
BASE_YDL_OPTIONS = {
    'format': 'bestvideo+bestaudio/best',
    'noplaylist': True,
    # Enable Node.js as the JavaScript runtime for YouTube extraction
    'js_runtimes': {'node': {}},
    'progress_hooks': [lambda d: print(f"Hook: Status={d.get('status')}, File={d.get('filename')}, {_progress_str(d)}") if d.get('status') == 'downloading' else None],
    'postprocessor_hooks': [lambda d: print(f"Postprocessor: Status={d.get('status')}, PP={d.get('postprocessor')}, Info={d.get('info_dict', {}).get('filepath') or ''}")],
    'http_headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    },
    # 'verbose': True,
    # 'quiet': False,
}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('yt_dlp').setLevel(logging.WARNING)


# Helper function for cleaner progress output
def _progress_str(d):
    return f"Progress={d.get('_percent_str', 'N/A')}, Speed={d.get('_speed_str', 'N/A')}, ETA={d.get('_eta_str', 'N/A')}"


def _format_for_height(max_height: int | None) -> str:
    """
    Build a yt-dlp format selector that respects max height.
    - If max_height is provided, prefer separate video+audio and merge; fallback to progressive.
    - If None, use yt-dlp's best default.
    """
    if max_height:
        return f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    return "bestvideo+bestaudio/best"


def _fallback_format_for_height(max_height: int | None) -> str:
    """Return a progressive-only format selector to use after restricted DASH failures."""
    if max_height:
        return f"best[height<={max_height}]"
    return "best"


def handle(
    input_file: str,
    output_folder: str,
    config: "DownloadConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[str]:
    """Main entry point for video downloading.

    Args:
        input_file: URL to download (YouTube, etc.).
        output_folder: Path to output directory.
        config: DownloadConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Path to downloaded file or None on failure.
    """
    max_height = config.max_height if config else 720
    filename_template = config.filename_template if config else DEFAULT_FILENAME_TEMPLATE
    return _download_video(
        input_file,
        output_folder,
        filename_template=filename_template,
        max_height=max_height,
    )


# --- Modified Function ---
def _download_video(url, output_folder, filename_template=DEFAULT_FILENAME_TEMPLATE, max_height: int | None = 720) -> str | None:
    """Internal: Download video from URL."""

    logger.info(f"Attempting to download video for: {url}")
    logger.info(f"Target output folder: {output_folder}")
    if max_height:
        logger.info(f"Requested max height: {max_height}p")

    # --- Dynamic Options Setup ---
    options = BASE_YDL_OPTIONS.copy()
    output_template = os.path.join(output_folder, filename_template)
    options['outtmpl'] = output_template
    # Override format based on desired height (default 360p if not provided)
    options['format'] = _format_for_height(max_height)
    # --- End Dynamic Options Setup ---

    logger.info("Using options:")
    for key, value in options.items():
        if key not in ['progress_hooks', 'postprocessor_hooks']:
             logger.info(f"  {key}: {value}")
        else:
             logger.info(f"  {key}: <defined>")
    logger.info("-" * 30)
    logger.info("IMPORTANT: Will merge best available video+audio within requested height when needed (requires ffmpeg).")
    logger.info(f"           Output file will be saved in '{output_folder}' with template '{filename_template}'.")
    logger.info("-" * 30)

    # --- Ensure output directory exists ---
    try:
        # Make sure output_folder is an absolute path for consistent return value
        absolute_output_folder = os.path.abspath(output_folder)
        if not os.path.exists(absolute_output_folder):
            logger.info(f"Creating output directory: {absolute_output_folder}")
            os.makedirs(absolute_output_folder)
            logger.info(f"Directory '{absolute_output_folder}' created successfully.")
        elif not os.path.isdir(absolute_output_folder):
            logger.error(f"Output path '{absolute_output_folder}' exists but is not a directory.")
            return None # Return None on pre-check failure
    except OSError as e:
        logger.error(f"Failed to create or access output directory '{absolute_output_folder}': {e}")
        return None # Return None on pre-check failure
    except Exception as e:
         logger.error(f"An unexpected error occurred during output directory setup: {e}")
         return None
    # Update options with the absolute path just in case relative was passed
    options['outtmpl'] = os.path.join(absolute_output_folder, filename_template)
    # --- Directory Check End ---


    final_filepath = None # Variable to store the result path

    def _run_download(current_options: dict) -> dict | None:
        with yt_dlp.YoutubeDL(current_options) as ydl:
            logger.info("Starting download and potential merge...")
            return ydl.extract_info(url, download=True)

    try:
        # Use extract_info with download=True to get the info dict containing the final path
        try:
            info_dict = _run_download(options)
        except yt_dlp.utils.DownloadError as e:
            if "HTTP Error 403" in str(e):
                logger.warning("Encountered HTTP 403 while fetching highest quality streams. Falling back to progressive format.")
                fallback_options = options.copy()
                fallback_options['format'] = _fallback_format_for_height(max_height)
                logger.info(f"Retrying download with fallback format selector: {fallback_options['format']}")
                info_dict = _run_download(fallback_options)
            else:
                raise

        # --- Retrieve File Path ---
        # The exact key might vary slightly based on yt-dlp version and download type,
        # but 'filepath' usually holds the final path after postprocessing (like merging).
        # 'requested_downloads'[0]['filepath'] might exist before merging.
        if info_dict:
             # Prioritize 'filepath' which usually exists after all processing
            final_filepath = info_dict.get('filepath')
            if not final_filepath:
                # Fallback: Check requested_downloads if 'filepath' isn't top-level
                requested_downloads = info_dict.get('requested_downloads')
                if requested_downloads and isinstance(requested_downloads, list) and len(requested_downloads) > 0:
                     final_filepath = requested_downloads[0].get('filepath')

        # --- Success Logging ---
        if final_filepath and os.path.exists(final_filepath):
             # Ensure the path is absolute before returning
             final_filepath = os.path.abspath(final_filepath)
             logger.info("\n-----------------------------")
             logger.info("yt-dlp download and process finished successfully.")
             logger.info(f"Final video file path: {final_filepath}")
             logger.info("-----------------------------")
             # Optional verification (already done above by checking existence)
        else:
             # This case might happen if download technically finished (no exception)
             # but the expected info wasn't found or file doesn't exist.
             logger.warning("\n-----------------------------")
             logger.warning("yt-dlp process finished, but could not determine the final file path from results.")
             logger.warning(f"Please check the output folder '{absolute_output_folder}' manually.")
             logger.warning("-----------------------------")
             final_filepath = None # Ensure we return None if path wasn't confirmed

    except yt_dlp.utils.DownloadError as e:
        logger.error("\n-----------------------------")
        logger.error(f"yt-dlp download/processing error: {e}")
        logger.error(f"Check URL, network, permissions in '{absolute_output_folder}', and ffmpeg installation.")
        logger.error("-----------------------------")
        final_filepath = None # Ensure return None on error

    except Exception as e:
        logger.error("\n-----------------------------")
        logger.error(f"An unexpected error occurred: {type(e).__name__} - {e}")
        logger.error("-----------------------------")
        final_filepath = None # Ensure return None on error

    return final_filepath # Return the path or None


if __name__ == "__main__":
    # --- CLI: optional --height and --out ---
    parser = argparse.ArgumentParser(description="Download a video with optional max height (default 360p).")
    parser.add_argument("url", nargs="?", default='https://www.youtube.com/watch?v=dQw4w9WgXcQ', help="Video URL")
    parser.add_argument("--out", default='./my_downloaded_videos', help="Output folder")
    parser.add_argument("--height", type=int, default=720, help="Max video height in pixels (e.g., 360, 720, 1080). Default: 360")
    args = parser.parse_args()

    VIDEO_URL = args.url
    TARGET_DOWNLOAD_FOLDER = args.out

    print(f"Running download script for URL: {VIDEO_URL}")
    print(f"Attempting to save to folder: {TARGET_DOWNLOAD_FOLDER}")
    if args.height:
        print(f"Requested max height: {args.height}p")
    print("-" * 30)

    # Call the modified function and capture the result
    downloaded_file_path = _download_video(VIDEO_URL, TARGET_DOWNLOAD_FOLDER, max_height=args.height)

    print("-" * 30)
    if downloaded_file_path:
        print(f"SUCCESS: Video downloaded successfully.")
        print(f"File saved at: {downloaded_file_path}")
        # You can now use 'downloaded_file_path' for further processing
        try:
             print(f"File size: {os.path.getsize(downloaded_file_path) / (1024*1024):.2f} MB")
        except OSError as e:
             print(f"Could not get file size: {e}")
    else:
        print("FAILURE: Video download failed. Check logs above for details.")

    print("\nScript finished execution.")