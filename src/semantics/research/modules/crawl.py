"""Utilities for crawling web pages into Markdown using Crawl4AI."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import AsyncGeneratorType
from typing import TYPE_CHECKING, Iterable, List

if TYPE_CHECKING:
    from ..config import CrawlConfig

from crawl4ai import AsyncWebCrawler, BFSDeepCrawlStrategy, BrowserConfig, CrawlerRunConfig
from crawl4ai.cache_context import CacheMode
from crawl4ai.models import CrawlResult, CrawlResultContainer
from playwright.async_api import Error as PlaywrightError


_LOGGER = logging.getLogger(__name__)
_DEFAULT_WORD_COUNT_THRESHOLD = 50
_DEFAULT_MAX_PAGES = 10
_SETUP_LOCK = threading.Lock()
_SETUP_COMPLETE = False
_RETRYABLE_PLAYWRIGHT_SNIPPETS = (
    "target page, context or browser has been closed",
    "browser has been closed",
    "browsertype.launch",
    "sigsegv",
)
_MAX_CRAWL_RETRIES = 3
_CRAWL_RETRY_BASE_DELAY_SECONDS = 2.0


def _playwright_roots() -> List[Path]:
    roots: List[Path] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_path:
        roots.append(Path(env_path))
    roots.append(Path.home() / ".cache" / "ms-playwright")
    return roots


def _has_playwright_installation() -> bool:
    for root in _playwright_roots():
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name.lower()
            if "chromium" not in name:
                continue
            marker = entry / "INSTALLATION_COMPLETE"
            if marker.exists() or any(child.is_dir() for child in entry.iterdir()):
                return True
    return False


def _ensure_crawl4ai_runtime(debug: bool) -> None:
    global _SETUP_COMPLETE
    with _SETUP_LOCK:
        if _SETUP_COMPLETE:
            return

        if _has_playwright_installation():
            _SETUP_COMPLETE = True
            return

        setup_exe = shutil.which("crawl4ai-setup")
        if not setup_exe:
            candidate = Path(sys.executable).with_name("crawl4ai-setup")
            if candidate.exists():
                setup_exe = str(candidate)
        if not setup_exe:
            raise RuntimeError("crawl4ai-setup command not found; install Crawl4AI extras first.")

        if debug and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.DEBUG)

        if debug:
            _LOGGER.debug("Running crawl4ai-setup to provision browser dependencies")

        try:
            subprocess.run(
                [setup_exe],
                check=True,
                stdout=None if debug else subprocess.PIPE,
                stderr=None if debug else subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on runtime environment
            if not debug and exc.stdout:
                _LOGGER.error("crawl4ai-setup stdout: %s", exc.stdout)
            if not debug and exc.stderr:
                _LOGGER.error("crawl4ai-setup stderr: %s", exc.stderr)
            raise RuntimeError("crawl4ai-setup failed to provision Playwright browsers.") from exc

        if not _has_playwright_installation():
            raise RuntimeError("crawl4ai-setup completed but no Playwright browsers were detected.")

        _SETUP_COMPLETE = True


def _extract_markdown(result: CrawlResult) -> str:
    if result.markdown:
        return result.markdown

    if result.extracted_content:
        return result.extracted_content

    if result.cleaned_html:
        return result.cleaned_html

    return result.html or ""


def _coerce_results(raw: object) -> Iterable[CrawlResult]:
    if raw is None:
        return []

    if isinstance(raw, CrawlResult):
        return [raw]

    if isinstance(raw, CrawlResultContainer):
        return list(raw)

    if isinstance(raw, (list, tuple)):
        results: List[CrawlResult] = []
        for item in raw:
            results.extend(list(_coerce_results(item)))
        return results

    raise TypeError(f"Unsupported crawl result type: {type(raw)!r}")


async def _collect_results(raw: object) -> List[CrawlResult]:
    if isinstance(raw, AsyncGeneratorType):
        results: List[CrawlResult] = []
        async for item in raw:  # pragma: no cover - streaming not exercised in tests
            results.extend(list(_coerce_results(item)))
        return results

    return list(_coerce_results(raw))


def _write_markdown_files(results: List[CrawlResult], output_directory: Path, debug: bool) -> List[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)

    next_index = 1
    written_paths: List[Path] = []

    for result in results:
        while (output_directory / f"{next_index}.md").exists():
            next_index += 1

        file_name = f"{next_index}.md"
        target_path = output_directory / file_name
        sequence_number = next_index
        next_index += 1

        # Provide lightweight provenance before the generated Markdown content.
        metadata_lines = [
            f"<!-- Source: {result.url} -->",
        ]
        if result.redirected_url and result.redirected_url != result.url:
            metadata_lines.append(f"<!-- Redirected URL: {result.redirected_url} -->")
        if result.status_code is not None:
            metadata_lines.append(f"<!-- Status: {result.status_code} -->")

        markdown_body = _extract_markdown(result)
        document = "\n".join(metadata_lines) + "\n\n" + markdown_body

        with target_path.open("w", encoding="utf-8") as handle:
            handle.write(document)
            if not document.endswith("\n"):
                handle.write("\n")

        written_paths.append(target_path)

        if debug:
            _LOGGER.debug(
                "Wrote markdown file",
                extra={
                    "path": str(target_path),
                    "url": result.url,
                    "sequence": sequence_number,
                },
            )

    return written_paths


async def crawl_url_async(
    url: str,
    output_directory: Path | str,
    *,
    deep_crawl: bool = False,
    max_depth: int = 1,
    max_pages: int = _DEFAULT_MAX_PAGES,
    include_external: bool = False,
    word_count_threshold: int = _DEFAULT_WORD_COUNT_THRESHOLD,
    debug: bool = False,
) -> List[Path]:
    """Asynchronously crawl *url* and write Markdown snapshots to *output_directory*."""

    if not url or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    if max_depth <= 0:
        raise ValueError("max_depth must be greater than zero.")

    if max_pages <= 0:
        raise ValueError("max_pages must be greater than zero.")

    if debug and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG)

    _ensure_crawl4ai_runtime(debug)

    browser_config = BrowserConfig(headless=True, verbose=debug)

    deep_strategy = None
    if deep_crawl:
        deep_strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            logger=_LOGGER if debug else None,
        )

    run_config = CrawlerRunConfig(
        word_count_threshold=word_count_threshold,
        cache_mode=CacheMode.BYPASS,
        deep_crawl_strategy=deep_strategy,
        verbose=debug,
    )

    last_error: Exception | None = None

    for attempt in range(1, _MAX_CRAWL_RETRIES + 1):
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                raw_result = await crawler.arun(url, config=run_config)
                results = await _collect_results(raw_result)

            if not results:
                raise RuntimeError(f"Crawl did not yield any results for URL: {url}")

            return _write_markdown_files(results, Path(output_directory), debug)

        except PlaywrightError as exc:  # pragma: no cover - network/runtime fragility
            last_error = exc
            message = str(exc).lower()
            retryable = any(fragment in message for fragment in _RETRYABLE_PLAYWRIGHT_SNIPPETS)
            if not retryable or attempt >= _MAX_CRAWL_RETRIES:
                raise
            delay = _CRAWL_RETRY_BASE_DELAY_SECONDS * attempt
            _LOGGER.warning(
                "Retrying crawl after Playwright failure (attempt %s/%s): %s",
                attempt,
                _MAX_CRAWL_RETRIES,
                exc,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            last_error = exc
            raise

    if last_error:
        raise last_error

    raise RuntimeError(f"Crawl failed for URL: {url}")


def crawl_url(
    url: str,
    output_directory: Path | str,
    **kwargs,
) -> List[Path]:
    """Synchronously crawl *url*; see :func:`crawl_url_async` for options."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(crawl_url_async(url, output_directory, **kwargs))

    if loop.is_running():  # pragma: no cover - depends on hosting app
        raise RuntimeError("crawl_url cannot be called from within a running event loop.")

    return loop.run_until_complete(crawl_url_async(url, output_directory, **kwargs))


def download_urls(
    urls: List[str],
    output_directory: Path | str,
    *,
    deep_crawl: bool = False,
    max_depth: int = 1,
    max_pages: int = _DEFAULT_MAX_PAGES,
    include_external: bool = False,
    word_count_threshold: int = _DEFAULT_WORD_COUNT_THRESHOLD,
    debug: bool = False,
) -> List[Path]:
    """Download a batch of URLs into Markdown snapshots."""

    if not urls:
        return []

    print("INFO: Downloading contents")

    base_output_directory = Path(output_directory)

    aggregated_paths: List[Path] = []
    for index, target_url in enumerate(urls, start=1):
        target_output_directory: Path
        if deep_crawl:
            target_output_directory = base_output_directory / f"tree-{index}"
        else:
            target_output_directory = base_output_directory

        aggregated_paths.extend(
            crawl_url(
                target_url,
                target_output_directory,
                deep_crawl=deep_crawl,
                max_depth=max_depth,
                max_pages=max_pages,
                include_external=include_external,
                word_count_threshold=word_count_threshold,
                debug=debug,
            )
        )

    return aggregated_paths


def handle(
    urls: List[str],
    output_folder: str,
    config: "CrawlConfig | None" = None,
    *,
    deep_crawl: bool | None = None,
    max_depth: int | None = None,
    max_pages: int | None = None,
    include_external: bool | None = None,
    word_count_threshold: int | None = None,
    debug: bool = False,
) -> List[Path]:
    """Main entry point for web page crawling.

    Args:
        urls: List of URLs to crawl.
        output_folder: Directory for output files.
        config: CrawlConfig instance or None for defaults.
        deep_crawl: Enable BFS deep crawling (overrides config).
        max_depth: Maximum crawl depth (overrides config).
        max_pages: Maximum pages per URL (overrides config).
        include_external: Allow external domains (overrides config).
        word_count_threshold: Minimum word count (overrides config).
        debug: Enable verbose debug output.

    Returns:
        List of paths to saved Markdown files.
    """
    resolved_deep = deep_crawl if deep_crawl is not None else (
        config.deep_crawl if config else False
    )
    resolved_depth = max_depth if max_depth is not None else (
        config.max_depth if config else 1
    )
    resolved_pages = max_pages if max_pages is not None else (
        config.max_pages if config else _DEFAULT_MAX_PAGES
    )
    resolved_external = include_external if include_external is not None else (
        config.include_external if config else False
    )
    resolved_threshold = word_count_threshold if word_count_threshold is not None else (
        config.word_count_threshold if config else _DEFAULT_WORD_COUNT_THRESHOLD
    )

    return download_urls(
        urls,
        output_folder,
        deep_crawl=resolved_deep,
        max_depth=resolved_depth,
        max_pages=resolved_pages,
        include_external=resolved_external,
        word_count_threshold=resolved_threshold,
        debug=debug,
    )
