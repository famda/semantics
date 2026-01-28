"""Research CLI configuration classes.

This module defines Pydantic configuration models for all research processing modules.
Each config class holds defaults that were previously scattered across module-level
constants and argparse defaults. Import and use these classes to configure module
behavior via YAML files or programmatically.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Search Config
# ---------------------------------------------------------------------------


class SearchConfig(BaseModel):
    """Configuration for web and video search."""

    max_results: int = Field(
        default=100, description="Maximum number of search results to retrieve"
    )
    safesearch: str = Field(
        default="moderate", description="Safe search level: off, moderate, strict"
    )
    region: str = Field(default="us-en", description="Search region/locale")


# ---------------------------------------------------------------------------
# Crawl Config
# ---------------------------------------------------------------------------


class CrawlConfig(BaseModel):
    """Configuration for web page crawling."""

    deep_crawl: bool = Field(
        default=False, description="Enable BFS deep crawling from seed URLs"
    )
    max_depth: int = Field(
        default=1, description="Maximum traversal depth when deep crawling"
    )
    max_pages: int = Field(
        default=10, description="Maximum number of pages to crawl per seed URL"
    )
    include_external: bool = Field(
        default=False, description="Allow deep crawl to follow external domains"
    )
    word_count_threshold: int = Field(
        default=50, description="Minimum word count required to save a page"
    )


# ---------------------------------------------------------------------------
# Download Config
# ---------------------------------------------------------------------------


class DownloadConfig(BaseModel):
    """Configuration for video downloads (yt-dlp)."""

    filename_template: str = Field(
        default="%(title)s_%(id)s.%(ext)s",
        description="yt-dlp output filename template",
    )
    max_height: int = Field(
        default=720, description="Maximum video height in pixels"
    )
    prefer_free_formats: bool = Field(
        default=True, description="Prefer free codecs (VP9, Opus) when available"
    )


# ---------------------------------------------------------------------------
# Candidates Config
# ---------------------------------------------------------------------------


class CandidatesConfig(BaseModel):
    """Configuration for candidate ranking and selection."""

    retriever_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="Sentence embedding model for retrieval",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    max_chunk_tokens: int = Field(
        default=128, description="Maximum tokens per text chunk"
    )
    overlap_tokens: int = Field(
        default=32, description="Token overlap between adjacent chunks"
    )
    retrieval_k: int = Field(
        default=50, description="Number of candidates to retrieve before reranking"
    )
    rerank_batch_size: int = Field(
        default=32, description="Batch size for cross-encoder reranking"
    )
    aggregation_method: Literal["max", "mean", "top_k_mean", "weighted_mean"] = Field(
        default="max", description="Method for aggregating chunk scores"
    )
    min_score_threshold: float = Field(
        default=-5.0, description="Minimum score to include a candidate"
    )
    score_margin_threshold: Optional[float] = Field(
        default=None, description="Maximum score difference from top result"
    )


# ---------------------------------------------------------------------------
# Extract Config
# ---------------------------------------------------------------------------


class ExtractConfig(BaseModel):
    """Configuration for structured content extraction."""

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for structured extraction (via LiteLLM)",
    )
    max_tokens: int = Field(
        default=4096, description="Maximum tokens for LLM response"
    )
    temperature: float = Field(
        default=0.0, description="LLM temperature (0.0 for deterministic)"
    )


# ---------------------------------------------------------------------------
# Root Research Config
# ---------------------------------------------------------------------------


class ResearchConfig(BaseModel):
    """Root configuration for the research CLI."""

    search: SearchConfig = Field(default_factory=SearchConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    candidates: CandidatesConfig = Field(default_factory=CandidatesConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)


# ---------------------------------------------------------------------------
# Config Loading Helpers
# ---------------------------------------------------------------------------


def _normalize_research_payload(payload: dict) -> dict:
    """Normalize YAML payload to match ResearchConfig structure."""
    data = dict(payload or {})
    # Support both top-level and nested "research" key
    if "research" in data and isinstance(data["research"], dict):
        data = dict(data["research"])
    return data


def _parse_model(model_cls: type[BaseModel], data: dict) -> BaseModel:
    """Parse data into Pydantic model (v1/v2 compatible)."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def load_research_config(path: str) -> ResearchConfig:
    """Load and validate research configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    payload = _normalize_research_payload(raw)
    config = _parse_model(ResearchConfig, payload)
    return config
