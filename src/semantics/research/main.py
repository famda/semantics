"""Research Processing CLI Tool.

This CLI provides web and video research capabilities including search,
crawling, content extraction, and candidate ranking.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import click

# Setup path for imports
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    platform_root = os.path.dirname(script_dir)

    if platform_root not in sys.path:
        sys.path.insert(0, platform_root)

except Exception as e:
    print(f"An unexpected error occurred during initial setup: {e}", file=sys.stderr)
    sys.exit(1)


DOWNLOAD_FROM_SEARCH = "__DOWNLOAD_FROM_SEARCH__"
_YOUTUBE_HOSTS = ("youtube.com", "youtu.be")


def _is_youtube_url(url: str | None) -> bool:
    if not url:
        return False
    normalized = url.lower()
    return any(host in normalized for host in _YOUTUBE_HOSTS)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i", "--input",
    "input_file",
    type=click.Path(exists=True),
    default=None,
    help="Input file for processing",
)
@click.option(
    "-o", "--output",
    "output_folder",
    required=True,
    type=click.Path(),
    help="Output folder path",
)
@click.option(
    "-s", "--search",
    "search_query",
    type=str,
    default=None,
    help="Text query to research",
)
@click.option(
    "--search-limit",
    type=int,
    default=None,
    help="Maximum number of web and video results to retrieve",
)
@click.option(
    "--download",
    "download_flag",
    is_flag=True,
    default=False,
    help="Download/crawl search results (use with --search)",
)
@click.option(
    "--download-url",
    "download_url",
    type=str,
    default=None,
    help="Specific URL to crawl (alternative to --download flag)",
)
@click.option(
    "--download-deep",
    is_flag=True,
    default=False,
    help="Enable BFS deep crawling",
)
@click.option(
    "--download-max-depth",
    type=int,
    default=None,
    help="Maximum traversal depth when deep crawling",
)
@click.option(
    "--download-max-pages",
    type=int,
    default=None,
    help="Page budget when deep crawling",
)
@click.option(
    "--download-include-external",
    is_flag=True,
    default=False,
    help="Allow deep crawl to follow external domains",
)
@click.option(
    "--download-word-threshold",
    type=int,
    default=None,
    help="Minimum word count required for a page to be materialized",
)
@click.option(
    "--structured",
    is_flag=True,
    default=False,
    help="Extract structured content from crawled pages",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable verbose debug logging",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file",
)
def main(
    input_file: Optional[str],
    output_folder: str,
    search_query: Optional[str],
    search_limit: Optional[int],
    download_flag: bool,
    download_url: Optional[str],
    download_deep: bool,
    download_max_depth: Optional[int],
    download_max_pages: Optional[int],
    download_include_external: bool,
    download_word_threshold: Optional[int],
    structured: bool,
    debug: bool,
    config: Optional[str],
) -> None:
    """
    \b
    Semantics CLI [research] - Unified interface for media intelligence
    -------------------------------------------
    Extract meaning, not just metadata. Composable AI operations designed for developers.
    \b
    Examples:
        research-cli -s "machine learning" -o ./output --download
        research-cli --download-url https://example.com -o ./output --structured
        research-cli -s "AI trends" -o ./output --config config.yml
    """
    # Validate inputs
    if not search_query and not input_file and not download_url and not download_flag:
        raise click.UsageError("Either --search, --download, --download-url, or --input must be specified.")

    if download_flag and not search_query:
        raise click.UsageError("--download flag requires --search to supply web results.")

    # Determine effective download mode
    do_download_from_search = download_flag and search_query
    do_download_url = download_url is not None

    # Load config if provided
    research_config = None
    if config:
        from config import load_research_config
        research_config = load_research_config(config)

    # Setup directories
    output_dir = os.path.abspath(output_folder)
    temp_path = os.path.join(output_dir, "temp")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)

    # Resolve config values with CLI overrides
    search_cfg = research_config.search if research_config else None
    crawl_cfg = research_config.crawl if research_config else None
    candidates_cfg = research_config.candidates if research_config else None

    # Search phase
    search_payload = None
    ranked_candidates: List[dict] = []
    if search_query:
        from research.modules import search as search_module
        from research.modules import candidates as candidates_module

        max_results = search_limit if search_limit is not None else (
            search_cfg.max_results if search_cfg else 100
        )

        search_payload = search_module.handle(
            search_query,
            output_folder=output_dir,
            config=search_cfg,
            max_results=max_results,
            debug=debug,
        )

        # Rank candidates
        ranked_candidates = candidates_module.handle(
            search_query,
            search_payload,
            output_folder=output_dir,
            config=candidates_cfg,
            final_top_k=max_results,
            debug=debug,
        )

        if not ranked_candidates:
            print("No ranked candidates generated from search results.")

    # Determine download targets
    download_web_targets: List[str] = []
    download_video_targets: List[str] = []
    if do_download_from_search:
        if not ranked_candidates:
            raise click.UsageError("No ranked candidates available to supply download URLs.")

        seen_web_urls = set()
        seen_video_urls = set()
        for candidate in ranked_candidates:
            item = candidate.get("item", {}) if isinstance(candidate, dict) else {}
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url and isinstance(item.get("raw"), dict):
                url = item["raw"].get("url")

            if not url:
                continue

            item_type = item.get("type")
            if item_type == "video":
                if not _is_youtube_url(url) or url in seen_video_urls:
                    continue
                seen_video_urls.add(url)
                download_video_targets.append(url)
                continue

            if item_type != "web":
                continue

            if url in seen_web_urls:
                continue

            seen_web_urls.add(url)
            download_web_targets.append(url)

        if not download_web_targets and not download_video_targets:
            raise click.UsageError("No crawlable or downloadable candidates were identified from ranked results.")
    elif do_download_url:
        if _is_youtube_url(download_url):
            download_video_targets = [download_url]
        else:
            download_web_targets = [download_url]

    # Crawl web targets
    downloaded_web_paths: List[Path] = []
    if download_web_targets:
        from research.modules import crawl as crawl_module

        crawl_output_dir = os.path.join(temp_path, "content")
        os.makedirs(crawl_output_dir, exist_ok=True)

        # Resolve crawl parameters with CLI overrides
        deep = download_deep or (crawl_cfg.deep_crawl if crawl_cfg else False)
        max_depth = download_max_depth if download_max_depth is not None else (
            crawl_cfg.max_depth if crawl_cfg else 1
        )
        max_pages = download_max_pages if download_max_pages is not None else (
            crawl_cfg.max_pages if crawl_cfg else 10
        )
        include_ext = download_include_external or (crawl_cfg.include_external if crawl_cfg else False)
        word_thresh = download_word_threshold if download_word_threshold is not None else (
            crawl_cfg.word_count_threshold if crawl_cfg else 50
        )

        downloaded_web_paths = crawl_module.handle(
            download_web_targets,
            output_folder=crawl_output_dir,
            config=crawl_cfg,
            deep_crawl=deep,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_ext,
            word_count_threshold=word_thresh,
            debug=debug,
        )

        if debug:
            print(f"Downloaded {len(downloaded_web_paths)} web targets.")

        if structured and downloaded_web_paths:
            from research.modules import extract as extract_module

            extract_cfg = research_config.extract if research_config else None
            extract_module.handle(
                Path(crawl_output_dir),
                output_folder=output_dir,
                config=extract_cfg,
                debug=debug,
            )
        elif structured:
            print("--structured flag skipped because no web content was saved from the crawl.")
    elif structured:
        print("--structured flag ignored because no crawl targets were processed. Use with --download.")

    # Download video targets
    if download_video_targets:
        from research.modules import download as download_module

        download_cfg = research_config.download if research_config else None
        video_output_dir = os.path.join(temp_path, "videos")
        os.makedirs(video_output_dir, exist_ok=True)

        downloaded_video_paths = download_module.handle(
            download_video_targets,
            output_folder=video_output_dir,
            config=download_cfg,
            debug=debug,
        )

        if debug:
            if downloaded_video_paths:
                print(f"Downloaded {len(downloaded_video_paths)} video targets.")
            else:
                print("No videos were downloaded successfully. Check logs for details.")


if __name__ == "__main__":
    main()