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
# NER Config
# ---------------------------------------------------------------------------


class NerConfig(BaseModel):
    """Configuration for named entity recognition."""

    model: str = Field(
        default="Jean-Baptiste/roberta-large-ner-english",
        description="HuggingFace NER model name",
    )
    device: Optional[str] = Field(
        default=None, description="Device to run NER on (cuda/cpu/None for auto)"
    )
    batch_size: int = Field(default=8, description="Batch size for NER inference")
    confidence_threshold: float = Field(
        default=0.6, description="Minimum confidence score for entities"
    )
    aggregation_strategy: str = Field(
        default="simple", description="Token aggregation strategy"
    )


# ---------------------------------------------------------------------------
# Classification Config
# ---------------------------------------------------------------------------


class ClassifyConfig(BaseModel):
    """Configuration for LLM-based document classification."""

    model: str = Field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        description="HuggingFace model for LLM-based classification",
    )
    max_tokens: int = Field(
        default=256, description="Max tokens for the classification response"
    )
    candidate_labels: list[str] = Field(
        default_factory=lambda: [
            "invoice",
            "contract",
            "report",
            "letter",
            "resume",
            "scientific paper",
            "manual",
            "form",
            "presentation",
            "legal document",
            "financial statement",
            "memo",
            "newsletter",
            "brochure",
            "engineering document",
            "architecture document",
            "cyber security report",
            "technical specification",
            "user guide",
            "policy document",
            "compliance document",
            "project plan",
            "meeting notes",
            "proposal",
            "whitepaper",
            "datasheet",
        ],
        description="Candidate labels to guide the LLM classification",
    )
    max_tags: int = Field(
        default=8, description="Maximum number of tags to return"
    )


# ---------------------------------------------------------------------------
# Overview Config
# ---------------------------------------------------------------------------


class OverviewConfig(BaseModel):
    """Configuration for LLM-based document overview generation."""

    model_id: str = Field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        description="HuggingFace model for overview generation",
    )
    max_tokens: int = Field(
        default=1024, description="Max tokens per overview generation call"
    )
    chunk_size: int = Field(
        default=6000, description="Max characters per text chunk for iterative summarization"
    )


# ---------------------------------------------------------------------------
# Captions Config
# ---------------------------------------------------------------------------


class CaptionsConfig(BaseModel):
    """Configuration for image captioning using BLIP and Qwen3-VL.

    The ``quality`` setting controls the captioning strategy:
      - ``speed``    – BLIP only (fast, ~0.5 s / image).
      - ``balanced`` – BLIP + Qwen3-VL with 256 max tokens (~12 s / image).
      - ``quality``  – BLIP + Qwen3-VL with 512 max tokens (~21 s / image). **default**
    """

    quality: str = Field(
        default="quality",
        description="Captioning quality preset: speed | balanced | quality",
    )
    model_id: str = Field(
        default="Salesforce/blip-image-captioning-large",
        description="HuggingFace BLIP model identifier for basic captions",
    )
    precision: str = Field(
        default="fp16", description="Model precision (fp16, bf16, fp32)"
    )
    detailed_model_id: str = Field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        description="HuggingFace VLM model for very detailed captions",
    )
    detailed_max_tokens: int = Field(
        default=0,
        description="Max tokens for detailed caption (0 = auto from quality preset)",
    )
    run_ner: bool = Field(
        default=True,
        description="Run NER on detailed captions to extract entities per image",
    )
    ner_model: str = Field(
        default="Jean-Baptiste/roberta-large-ner-english",
        description="NER model to extract entities from detailed captions",
    )
    ner_confidence: float = Field(
        default=0.6,
        description="Minimum confidence score for NER entities",
    )


# ---------------------------------------------------------------------------
# Forms Config
# ---------------------------------------------------------------------------


class FormsConfig(BaseModel):
    """Configuration for form/key-value extraction."""

    device: Optional[str] = Field(
        default=None, description="Device (cuda/cpu/None for auto)"
    )


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------


class DocsConfig(BaseModel):
    """Root configuration for the docs CLI."""

    structured: StructuredConfig = Field(default_factory=StructuredConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    tables: TablesConfig = Field(default_factory=TablesConfig)
    markdown: MarkdownConfig = Field(default_factory=MarkdownConfig)
    ner: NerConfig = Field(default_factory=NerConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    overview: OverviewConfig = Field(default_factory=OverviewConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    forms: FormsConfig = Field(default_factory=FormsConfig)


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

