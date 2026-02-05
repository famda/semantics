"""Utilities for performing lightweight web and video searches."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..config import SearchConfig

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException


_LOGGER = logging.getLogger(__name__)
_SAFESEARCH_LEVEL = "moderate"
_REGION = "us-en"
_QUERY_TOKEN_PATTERN = re.compile(r"\w+")
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0
_VIDEO_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _get_search_defaults() -> dict:
	"""Get default values from SearchConfig to avoid circular imports."""
	try:
		from config import SearchConfig
		cfg = SearchConfig()
		return {
			"max_results": cfg.max_results,
			"safesearch": cfg.safesearch,
			"region": cfg.region,
		}
	except Exception:
		# Fallback defaults if config import fails
		return {
			"max_results": 100,
			"safesearch": "moderate",
			"region": "us-en",
		}


def _with_retries(func, *, description: str, debug: bool):
	last_exc: Exception | None = None
	for attempt in range(1, _MAX_RETRIES + 1):
		try:
			return func()
		except (TimeoutException, DDGSException) as exc:
			last_exc = exc
			if attempt == _MAX_RETRIES:
				raise
			if debug:
				_LOGGER.debug(
					"Retrying %s after exception (attempt %d/%d)",
					description,
					attempt,
					_MAX_RETRIES,
					extra={"error": str(exc)},
				)
			time.sleep(_RETRY_DELAY_SECONDS * attempt)
	if last_exc is not None:
		raise last_exc
	raise RuntimeError(f"Failed to execute {description} for unknown reasons")


def _tokenize(text: str) -> List[str]:
	return _QUERY_TOKEN_PATTERN.findall(text.lower())


def _compute_match_score(query_terms: List[str], text_parts: List[str]) -> float:
	haystack = set()
	for part in text_parts:
		haystack.update(_tokenize(part))

	unique_terms = set(query_terms)

	if not unique_terms:
		return 0.0

	matches = sum(1 for term in unique_terms if term in haystack)
	return round(matches / len(unique_terms), 3)


def _classify_url(url: str) -> str:
	try:
		netloc = urlparse(url).netloc.lower()
	except ValueError:
		return "web"

	for host in _VIDEO_HOSTS:
		if netloc.endswith(host):
			return "video"
	return "web"


def search_content(
	query: str,
	debug: bool = False,
	max_results: int | None = None,
) -> Dict[str, object]:
	"""Execute web and video searches for *query* and return aggregated results."""
	defaults = _get_search_defaults()
	max_results = max_results if max_results is not None else defaults["max_results"]

	if not query or not query.strip():
		raise ValueError("Query must be a non-empty string.")

	if max_results <= 0:
		raise ValueError("max_results must be greater than zero.")

	if debug and not logging.getLogger().handlers:
		logging.basicConfig(level=logging.DEBUG)

	normalized_query = query.strip()
	print(f"INFO: Searching for {normalized_query}")
	query_terms = _tokenize(normalized_query)

	if debug:
		_LOGGER.setLevel(logging.DEBUG)
		_LOGGER.debug(
			"Issuing DuckDuckGo searches",
			extra={"query": normalized_query, "limit": max_results},
		)

	web_results: List[Dict[str, object]] = []
	video_results: List[Dict[str, object]] = []
	seen_video_urls: set[str] = set()

	try:
		with DDGS() as ddgs:  # type: ignore[attr-defined]
			text_search_results = _with_retries(
				lambda: list(
					ddgs.text(
						normalized_query,
						max_results=max_results,
						safesearch=_SAFESEARCH_LEVEL,
						region=_REGION,
					)
				),
				description="DuckDuckGo web search",
				debug=debug,
			)
			for rank, item in enumerate(text_search_results, start=1):
				title_text = item.get("title", "").strip()
				url = item.get("href", "")
				snippet_text = item.get("body", "").strip()

				if not url:
					continue

				classification = _classify_url(url)
				if classification == "video":
					if url in seen_video_urls:
						continue
					seen_video_urls.add(url)
					video_results.append(
						{
							"rank": rank,
							"title": title_text,
							"url": url,
							"description": snippet_text,
							"duration": item.get("duration"),
							"published": item.get("published"),
							"publisher": item.get("source", "").strip(),
							"match_score": _compute_match_score(
								query_terms,
								[title_text, snippet_text],
							),
						}
					)
					if debug:
						_LOGGER.debug(
							"Captured video result from web search",
							extra={"title": title_text, "url": url, "rank": rank},
						)
					continue

				web_results.append(
					{
						"rank": rank,
						"title": title_text,
						"url": url,
						"description": snippet_text,
						"match_score": _compute_match_score(query_terms, [title_text, snippet_text]),
					}
				)

				if debug:
					_LOGGER.debug(
						"Captured web result",
						extra={"title": title_text, "url": url, "rank": rank},
					)

				if len(web_results) >= max_results:
					break

			video_search_results: List[Dict[str, object]] = []
			video_attempts = (
				{"region": _REGION, "backend": "auto"},
				{"region": "us-en", "backend": "ytsearch"},
				{"region": "us-en", "backend": "bing"},
			)

			for attempt in video_attempts:
				if video_search_results:
					break

				try:
					candidate = _with_retries(
						lambda: ddgs.videos(
							normalized_query,
							max_results=max_results,
							safesearch=_SAFESEARCH_LEVEL,
							region=attempt["region"],
							backend=attempt["backend"],
						),
						description=f"DuckDuckGo video search ({attempt['backend']})",
						debug=debug,
					)
					if candidate:
						video_search_results = candidate
						if debug:
							_LOGGER.debug(
								"Video search succeeded",
								extra={
									"backend": attempt["backend"],
									"count": len(candidate),
								},
							)
				except DDGSException as exc:
					if debug:
						_LOGGER.debug(
							"Video search attempt failed",
							extra={
								"backend": attempt["backend"],
								"reason": str(exc),
							},
						)
					continue

			for rank, item in enumerate(video_search_results, start=1):
				title_text = item.get("title", "").strip()
				url = item.get("href") or item.get("content", "")
				description = item.get("description", "").strip()
				publisher = item.get("publisher", "").strip()

				if not url:
					continue
				if url in seen_video_urls:
					continue
				seen_video_urls.add(url)

				video_results.append(
					{
						"rank": rank,
						"title": title_text,
						"url": url,
						"description": description,
						"duration": item.get("duration"),
						"published": item.get("published"),
						"publisher": publisher,
						"match_score": _compute_match_score(
							query_terms,
							[title_text, description, publisher],
						),
					}
				)

				if debug:
					_LOGGER.debug(
						"Captured video result",
						extra={"title": title_text, "url": url, "rank": rank},
					)

				if len(video_results) >= max_results:
					break
	except Exception as exc:  # pragma: no cover - relies on network
		msg = f"DuckDuckGo search failed: {exc}"
		if debug:
			_LOGGER.error(msg)
		raise RuntimeError(msg) from exc

	if debug:
		_LOGGER.debug(
			"Total results gathered",
			extra={
				"web_count": len(web_results),
				"video_count": len(video_results),
			},
		)

	combined_results = sorted(
		[
			{
				**result,
				"type": "web",
			}
			for result in web_results
		]
		+ [
			{
				**result,
				"type": "video",
			}
			for result in video_results
		],
		key=lambda item: (
			-item.get("match_score", 0),
			0 if item.get("type") == "web" else 1,
			item.get("rank", 0),
		),
	)

	payload: Dict[str, object] = {
		"query": normalized_query,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"results": combined_results,
	}

	return payload


def search_web(
	query: str,
	debug: bool = False,
	max_results: int | None = None,
) -> List[Dict[str, object]]:
	"""Backward-compatible helper returning only web results."""

	content = search_content(query, debug=debug, max_results=max_results)
	return [item for item in content["results"] if item.get("type") == "web"]


def save_results_to_json(
	results: Dict[str, object],
	output_directory: Path | str,
	filename: str = "search.json",
) -> Path:
	"""Persist results to *filename* within *output_directory*."""

	target_dir = Path(output_directory)
	target_dir.mkdir(parents=True, exist_ok=True)
	output_path = target_dir / filename

	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(results, handle, indent=2, ensure_ascii=False)

	return output_path


def handle(
	query: str,
	output_folder: str,
	config: "SearchConfig | None" = None,
	*,
	max_results: Optional[int] = None,
	debug: bool = False,
) -> Dict[str, object]:
	"""Main entry point for web and video search.

	Args:
		query: Search query string.
		output_folder: Directory for output files.
		config: SearchConfig instance or None for defaults.
		max_results: Override for maximum results (takes precedence over config).
		debug: Enable verbose debug output.

	Returns:
		Dictionary containing search results.
	"""
	defaults = _get_search_defaults()
	resolved_max = max_results
	if resolved_max is None:
		resolved_max = config.max_results if config else defaults["max_results"]

	payload = search_content(query, debug=debug, max_results=resolved_max)
	save_results_to_json(payload, output_folder)
	return payload

