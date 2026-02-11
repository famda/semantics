"""Audio CLI configuration classes.

This module defines Pydantic configuration models for all audio processing modules.
Each config class holds defaults that were previously scattered across module-level
constants. Import and use these classes to configure module behavior via YAML files
or programmatically.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field

try:
    from .constants import TRANSCRIPTION_ALLOWED_MODELS
except ImportError:
    from constants import TRANSCRIPTION_ALLOWED_MODELS


# ---------------------------------------------------------------------------
# Resample / Enhance Config
# ---------------------------------------------------------------------------


class ResampleConfig(BaseModel):
    """Configuration for audio resampling and format conversion."""

    sample_rate: int = Field(default=16000, description="Target sample rate in Hz")
    channels: int = Field(
        default=1, description="Number of output audio channels (1=mono)"
    )


class EnhanceConfig(BaseModel):
    """Configuration for audio enhancement (post-resample processing)."""

    enabled: bool = Field(default=False, description="Whether to run enhancement pass")


# ---------------------------------------------------------------------------
# Stem Separation Config
# ---------------------------------------------------------------------------


class StemConfig(BaseModel):
    """Configuration for source separation (Demucs vocals extraction)."""

    model: str = Field(default="htdemucs_ft", description="Demucs model variant to use")
    chunk_length: int = Field(
        default=900, description="Chunk duration in seconds for long audio"
    )
    two_stems: str = Field(
        default="vocals", description="Stem type to extract (vocals, drums, etc.)"
    )
    shifts: int = Field(
        default=0,
        description="Random shifts for equivariant stabilization (0=off, 1=default Demucs, higher=slower but better)",
    )
    overlap: float = Field(
        default=0.1,
        description="Overlap between processing segments (Demucs default 0.25, 0.1 is faster with minimal quality loss)",
    )


# ---------------------------------------------------------------------------
# Denoise Config
# ---------------------------------------------------------------------------


class DenoiseConfig(BaseModel):
    """Configuration for audio denoising."""

    chunk_length: int = Field(
        default=900, description="Chunk duration in seconds for processing"
    )
    allow_auto_scale: bool = Field(
        default=True, description="Allow automatic amplitude scaling"
    )


# ---------------------------------------------------------------------------
# VAD Config
# ---------------------------------------------------------------------------


class VadConfig(BaseModel):
    """Configuration for Voice Activity Detection."""

    chunk_length: int = Field(
        default=900, description="Chunk duration in seconds for processing"
    )
    threshold: float = Field(
        default=0.5, description="Speech probability threshold (0.0-1.0)"
    )
    min_speech_duration_ms: int = Field(
        default=250, description="Minimum speech segment length in ms"
    )
    min_silence_duration_ms: int = Field(
        default=100, description="Minimum silence gap to split segments in ms"
    )


# ---------------------------------------------------------------------------
# Transcribe Config
# ---------------------------------------------------------------------------


class TranscribeConfig(BaseModel):
    """Configuration for speech transcription (Whisper)."""

    model: TRANSCRIPTION_ALLOWED_MODELS = Field(
        default="distil-large-v3.5", description="Whisper model variant to use"
    )
    chunk_length: int = Field(
        default=900, description="Chunk duration in seconds for long audio"
    )
    chunk_overlap_seconds: float = Field(
        default=5.0, description="Overlap between chunks to avoid boundary artifacts"
    )
    segment_epsilon_seconds: float = Field(
        default=0.3, description="Tolerance for segment boundary merging"
    )
    model_attempts: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("cuda", "float16"),
            ("cuda", "int8_float16"),
            ("cpu", "int8"),
        ],
        description="Device/precision fallback sequence: [(device, compute_type), ...]",
    )


class TranscribeExperimentalConfig(BaseModel):
    """Configuration for experimental transcription using HuggingFace Transformers.

    This uses batched inference with Flash Attention 2 or SDPA for ultra-fast
    transcription, combined with optional CTC forced alignment for accurate
    word-level timestamps.
    """

    model: str = Field(
        default="openai/whisper-large-v3",
        description="HuggingFace model ID for Whisper (e.g., openai/whisper-large-v3, distil-whisper/large-v3)",
    )
    batch_size: int = Field(
        default=8,
        description="Number of parallel batches for inference (reduce for OOM)",
    )
    chunk_length_s: int = Field(
        default=60,
        description="Chunk length in seconds for batched processing",
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Use Flash Attention 2 if available (falls back to SDPA)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en') or None for auto-detection",
    )
    enable_ctc_refinement: bool = Field(
        default=True,
        description="Run CTC forced alignment to refine word timestamps",
    )
    ctc_model: str = Field(
        default="stt_en_quartznet15x5",
        description="NeMo CTC model for forced alignment refinement",
    )
    enable_diarization: bool = Field(
        default=True,
        description="Run speaker diarization to identify speakers",
    )


# ---------------------------------------------------------------------------
# Diarize Config
# ---------------------------------------------------------------------------


class DiarizeConfig(BaseModel):
    """Configuration for speaker diarization (NeMo MSDD)."""

    domain_type: str = Field(
        default="telephonic", description="Domain hint: telephonic or meeting"
    )
    pretrained_vad: str = Field(
        default="vad_multilingual_marblenet", description="NeMo VAD model name"
    )
    pretrained_speaker_model: str = Field(
        default="titanet_large", description="NeMo speaker embedding model name"
    )
    msdd_model: str = Field(
        default="diar_msdd_telephonic", description="NeMo MSDD overlap detection model"
    )
    onset: float = Field(default=0.8, description="VAD onset threshold")
    offset: float = Field(default=0.6, description="VAD offset threshold")
    pad_offset: float = Field(
        default=-0.05, description="Padding around speech segments in seconds"
    )

    # Long audio heuristics (audio > long_audio_threshold_secs triggers reduced clustering)
    long_audio_threshold_secs: float = Field(
        default=1800.0, description="Duration threshold to activate long-audio mode"
    )
    long_audio_window_length_in_sec: List[float] = Field(
        default_factory=lambda: [1.5, 1.0], description="Multiscale window lengths"
    )
    long_audio_shift_length_in_sec: List[float] = Field(
        default_factory=lambda: [0.75, 0.5], description="Multiscale shift lengths"
    )
    long_audio_multiscale_weights: List[float] = Field(
        default_factory=lambda: [1.0, 1.0], description="Multiscale combination weights"
    )
    long_audio_embeddings_per_chunk_max: int = Field(
        default=600, description="Max embeddings per chunk in long-audio mode"
    )
    long_audio_chunk_cluster_count_max: int = Field(
        default=24, description="Max clusters per chunk in long-audio mode"
    )
    long_audio_max_num_speakers_max: int = Field(
        default=6, description="Max speakers for MSDD in long-audio mode"
    )
    long_audio_sparse_search_volume_max: int = Field(
        default=10, description="Sparse search volume for long-audio mode"
    )
    long_audio_max_rp_threshold_max: float = Field(
        default=0.2, description="Max RP threshold for long-audio clustering"
    )


# ---------------------------------------------------------------------------
# CTC Alignment Config
# ---------------------------------------------------------------------------


class CtcConfig(BaseModel):
    """Configuration for CTC forced alignment."""

    segment_margin_seconds: float = Field(
        default=0.75, description="Margin added around aligned segments"
    )
    segment_margin_backoffs: Tuple[float, ...] = Field(
        default=(0.0, 2.0, 5.0, 10.0),
        description="Fallback margin sequence on alignment failure",
    )
    min_speaker_overlap_seconds: float = Field(
        default=0.0, description="Minimum overlap required to assign speaker to segment"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for CTC model: 'cuda', 'cpu', or None for auto-detect",
    )


# ---------------------------------------------------------------------------
# Classify Config
# ---------------------------------------------------------------------------


class ClassifyConfig(BaseModel):
    """Configuration for audio classification (AST model)."""

    model_id: str = Field(
        default="MIT/ast-finetuned-audioset-10-10-0.4593",
        description="HuggingFace model ID for audio classification",
    )
    expected_sample_rate: int = Field(
        default=16000, description="Expected input sample rate"
    )
    chunk_length: int = Field(
        default=900, description="Chunk duration in seconds for long audio"
    )
    top_n: int = Field(default=5, description="Number of top classes to return")


# ---------------------------------------------------------------------------
# Timeline Classifier Config
# ---------------------------------------------------------------------------


class TimelineConfig(BaseModel):
    """Configuration for timeline-based audio classification (sliding window)."""

    device: Optional[str] = Field(
        default=None, description="Device override (cuda/cpu/None=auto)"
    )
    batch_size: int = Field(default=32, description="Batch size for inference")
    window_size: float = Field(
        default=2.0, description="Analysis window size in seconds"
    )
    hop_size: float = Field(
        default=2.0, description="Hop between analysis windows in seconds"
    )
    target_sample_rate: int = Field(
        default=16000, description="Target sample rate for analysis"
    )
    emotion_threshold: float = Field(
        default=0.12, description="Minimum confidence for emotion labels"
    )
    audio_event_threshold: float = Field(
        default=0.35, description="Minimum confidence for audio events"
    )
    min_segment_duration: float = Field(
        default=0.5, description="Minimum segment duration in seconds"
    )
    emotion_temperature: float = Field(
        default=0.7, description="Softmax temperature for emotion model"
    )
    emotion_prob_smoothing: int = Field(
        default=5, description="Window size for probability smoothing"
    )
    emotion_confidence_gamma: float = Field(
        default=1.35, description="Gamma for confidence shaping"
    )
    min_speech_overlap_ratio: float = Field(
        default=0.15, description="Minimum VAD overlap to consider segment as speech"
    )
    vad_threshold: float = Field(default=0.5, description="VAD probability threshold")
    vad_min_speech_duration_ms: int = Field(
        default=250, description="VAD minimum speech duration"
    )
    vad_min_silence_duration_ms: int = Field(
        default=100, description="VAD minimum silence duration"
    )
    energy_window: float = Field(
        default=1.0, description="Energy analysis window in seconds"
    )


# ---------------------------------------------------------------------------
# Emotion Config
# ---------------------------------------------------------------------------


class EmotionConfig(BaseModel):
    """Configuration for speech emotion recognition."""

    model_name: str = Field(
        default="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        description="HuggingFace model ID for emotion recognition",
    )
    device: Optional[str] = Field(
        default=None, description="Device override (cuda/cpu/None=auto)"
    )
    batch_size: int = Field(default=8, description="Batch size for inference")
    temperature: float = Field(
        default=0.7, description="Softmax temperature for predictions"
    )
    prob_smoothing: int = Field(
        default=5, description="Window size for probability smoothing"
    )
    confidence_gamma: float = Field(
        default=1.35, description="Gamma for confidence shaping"
    )
    threshold: float = Field(default=0.12, description="Minimum confidence threshold")
    target_sample_rate: int = Field(
        default=16000, description="Target sample rate for analysis"
    )
    min_duration: float = Field(
        default=0.35, description="Minimum segment duration in seconds"
    )
    pad_seconds: float = Field(
        default=0.15, description="Padding around segments in seconds"
    )
    use_forced_alignment: bool = Field(
        default=True,
        description="Prefer CTC alignment segments over transcript segments",
    )


# ---------------------------------------------------------------------------
# Scenes / Chapters Config
# ---------------------------------------------------------------------------


class ScenesConfig(BaseModel):
    """Configuration for chapter/scene detection from transcripts."""

    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence embedding model for boundary detection",
    )
    llm_model: str = Field(
        default="microsoft/phi-4-mini-instruct",
        description="LLM for chapter overview/summary generation",
    )
    title_model: str = Field(
        default="microsoft/phi-4-mini-instruct",
        description="Lightweight LLM for chapter title generation",
    )
    min_chapter_length: float = Field(
        default=120.0, description="Minimum chapter length in seconds"
    )
    boundary_percentile: float = Field(
        default=75.0, description="Percentile threshold for boundary detection"
    )
    max_new_tokens: int = Field(
        default=256, description="Max new tokens for LLM generation"
    )
    max_ctx_chars: int = Field(
        default=6000, description="Max context characters for LLM prompt"
    )
    global_chunk_chars: int = Field(
        default=4000, description="Chunk size for global summary"
    )
    context_snippet_chars: int = Field(
        default=1200, description="Context snippet size for boundary prompts"
    )
    global_chunk_tokens: int = Field(
        default=160, description="Token limit for global chunk summaries"
    )
    global_summary_tokens: int = Field(
        default=220, description="Token limit for global summary"
    )
    temperature: float = Field(
        default=0.7, description="LLM temperature for generation"
    )
    top_p: float = Field(default=0.9, description="LLM top-p (nucleus sampling)")
    enable_llm: bool = Field(
        default=True, description="Enable LLM for title/summary (False=embedding only)"
    )


# ---------------------------------------------------------------------------
# Named Entity Recognition Config
# ---------------------------------------------------------------------------


class NerConfig(BaseModel):
    """Configuration for Named Entity Recognition on transcription segments."""

    model_name: str = Field(
        default="Jean-Baptiste/roberta-large-ner-english",
        description="HuggingFace model ID for NER",
    )
    device: Optional[str] = Field(
        default=None, description="Device override (cuda/cpu/None=auto)"
    )
    batch_size: int = Field(default=8, description="Batch size for inference")
    confidence_threshold: float = Field(
        default=0.92, description="Minimum confidence for entity detection"
    )
    aggregate_strategy: str = Field(
        default="simple",
        description="Token aggregation strategy: simple, first, average, max",
    )


# ---------------------------------------------------------------------------
# Slice Config
# ---------------------------------------------------------------------------


class SliceConfig(BaseModel):
    """Configuration for media slicing (time-range extraction)."""

    codec: str = Field(
        default="copy",
        description="FFmpeg codec: 'copy' for stream-copy (fastest) or a specific codec name.",
    )
    fallback_reencode: bool = Field(
        default=True,
        description="Re-encode automatically when stream-copy fails.",
    )


# ---------------------------------------------------------------------------
# Download Config
# ---------------------------------------------------------------------------


class DownloadConfig(BaseModel):
    """Configuration for audio downloading from URLs."""

    filename_template: str = Field(
        default="%(title)s_%(id)s.%(ext)s",
        description="yt-dlp filename template for downloaded audio.",
    )


# ---------------------------------------------------------------------------
# Root Audio Config
# ---------------------------------------------------------------------------


class AudioConfig(BaseModel):
    """Root configuration for the audio CLI."""

    slice: SliceConfig = Field(default_factory=SliceConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    resample: ResampleConfig = Field(default_factory=ResampleConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)
    stem: StemConfig = Field(default_factory=StemConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    transcribe_experimental: TranscribeExperimentalConfig = Field(
        default_factory=TranscribeExperimentalConfig
    )
    diarize: DiarizeConfig = Field(default_factory=DiarizeConfig)
    ctc: CtcConfig = Field(default_factory=CtcConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    classify_timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    scenes: ScenesConfig = Field(default_factory=ScenesConfig)
    ner: NerConfig = Field(default_factory=NerConfig)


# ---------------------------------------------------------------------------
# Config Loading Helpers
# ---------------------------------------------------------------------------


def _coerce_attempts(attempts: Sequence[Sequence[Any]]) -> List[Tuple[str, str]]:
    """Convert raw YAML list of lists into typed tuples for model_attempts."""
    converted: List[Tuple[str, str]] = []
    for item in attempts:
        if not item or len(item) < 2:
            continue
        converted.append((str(item[0]), str(item[1])))
    return converted


def _normalize_audio_payload(payload: dict) -> dict:
    """Normalize YAML payload to match AudioConfig structure."""
    data = dict(payload or {})
    # Support both top-level and nested "audio" key
    if "audio" in data and isinstance(data["audio"], dict):
        data = dict(data["audio"])

    # Alias support for common variations
    if "transcription" in data and "transcribe" not in data:
        data["transcribe"] = data.pop("transcription")

    return data


def _parse_model(model_cls: type[BaseModel], data: dict) -> BaseModel:
    """Parse data into Pydantic model (v1/v2 compatible)."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def load_audio_config(path: str) -> AudioConfig:
    """Load and validate audio configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    payload = _normalize_audio_payload(raw)
    config: AudioConfig = _parse_model(AudioConfig, payload)  # type: ignore[assignment]

    # Ensure model_attempts is properly typed
    if config.transcribe.model_attempts:
        config.transcribe.model_attempts = _coerce_attempts(
            config.transcribe.model_attempts
        )

    return config
