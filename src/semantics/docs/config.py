"""Docs CLI configuration classes.

This module defines Pydantic configuration models for all document processing modules.
Each config class holds defaults that were previously scattered across module-level
constants. Import and use these classes to configure module behavior via YAML files
or programmatically.
"""

from __future__ import annotations

from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured Extraction Config
# ---------------------------------------------------------------------------


class StructuredConfig(BaseModel):
    """Configuration for structured document extraction."""

    include_metadata: bool = Field(
        default=True, description="Include element metadata in output"
    )


# ---------------------------------------------------------------------------
# Images Config
# ---------------------------------------------------------------------------


class ImagesConfig(BaseModel):
    """Configuration for image extraction."""

    dpi: int = Field(default=150, description="Resolution for rendered images")
    min_size: int = Field(
        default=100,
        description="Minimum width/height in pixels; smaller images are skipped",
    )
    format: str = Field(default="png", description="Output image format (png, jpg)")


# ---------------------------------------------------------------------------
# Tables Config
# ---------------------------------------------------------------------------


class TablesConfig(BaseModel):
    """Configuration for table extraction."""

    output_format: str = Field(
        default="csv", description="Table output format (csv)"
    )


# ---------------------------------------------------------------------------
# Markdown Config
# ---------------------------------------------------------------------------


class MarkdownConfig(BaseModel):
    """Configuration for markdown export."""

    include_images: bool = Field(
        default=True, description="Write images referenced in the markdown"
    )
    dpi: int = Field(default=150, description="Resolution for embedded images")


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------


class DocsConfig(BaseModel):
    """Root configuration for the docs CLI."""

    structured: StructuredConfig = Field(default_factory=StructuredConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    tables: TablesConfig = Field(default_factory=TablesConfig)
    markdown: MarkdownConfig = Field(default_factory=MarkdownConfig)


# ---------------------------------------------------------------------------
# Config Loading Helpers
# ---------------------------------------------------------------------------


def _normalize_docs_payload(payload: dict) -> dict:
    """Normalize YAML payload to match DocsConfig structure."""
    data = dict(payload or {})
    # Support both top-level and nested "docs" key
    if "docs" in data and isinstance(data["docs"], dict):
        data = dict(data["docs"])
    return data


def _parse_model(model_cls: type[BaseModel], data: dict) -> BaseModel:
    """Parse data into Pydantic model (v1/v2 compatible)."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def load_docs_config(path: str) -> DocsConfig:
    """Load and validate docs configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    payload = _normalize_docs_payload(raw)
    config = _parse_model(DocsConfig, payload)
    return config
