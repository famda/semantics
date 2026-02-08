"""Reusable emotion analysis utilities for audio timelines.

This module provides speech emotion recognition using Wav2Vec2 models.
It analyzes audio segments from transcription or forced alignment to
predict emotions with confidence scores.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import librosa
import numpy as np
import torch

__all__ = ["handle"]
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Module-level model cache: model_name -> (model, processor, labels)
_EMOTION_MODEL_CACHE: dict[str, tuple] = {}

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from ..config import EmotionConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class _EmotionPrediction:
    """Structured output for a single audio window."""

    label: str
    confidence: float
    scores: dict[str, float]
    window_index: Optional[int] = None


@dataclass
class _SegmentWindow:
    """Represents a segment to analyze."""

    index: int
    start: float
    end: float
    text: str
    source_id: Optional[Union[int, str]]
    payload: Dict[str, Any]


class _EmotionAnalyzer:
    """Lightweight wrapper around the Wav2Vec2-based SER model.

    Parameters mirror the defaults used in the audio timeline classifier so behaviour is
    consistent regardless of the caller.
    """

    def __init__(
        self,
        *,
        model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device: Optional[str] = None,
        batch_size: int = 8,
        temperature: float = 0.7,
        prob_smoothing: int = 5,
        confidence_gamma: float = 1.35,
        threshold: float = 0.12,
        debug: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.temperature = temperature
        self.prob_smoothing = prob_smoothing
        self.confidence_gamma = confidence_gamma
        self.threshold = threshold

        debug_print(
            f"Loading emotion recognition model '{model_name}' on {self.device}",
            debug=debug,
        )

        # Check module-level cache first
        if model_name in _EMOTION_MODEL_CACHE:
            cached_model, cached_processor, cached_labels = _EMOTION_MODEL_CACHE[model_name]
            self.model = cached_model
            self.processor = cached_processor
            self.model.to(torch.device(self.device))  # type: ignore[arg-type]
            self.model.eval()
            self.labels = cached_labels
            debug_print(f"Emotion model '{model_name}' loaded from cache", debug=debug)
            return

        with gray_debug_output(debug):
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model.to(torch.device(self.device))  # type: ignore[arg-type]
        self.model.eval()

        config_labels = getattr(self.model.config, "id2label", None)
        if isinstance(config_labels, dict) and config_labels:
            try:
                ordered = [
                    config_labels[str(i)]
                    if str(i) in config_labels
                    else config_labels[i]
                    for i in range(len(config_labels))
                ]
                self.labels = [str(label).lower() for label in ordered]
            except Exception:
                self.labels = [str(label).lower() for label in config_labels.values()]
        else:
            self.labels = [
                "angry",
                "calm",
                "disgust",
                "fearful",
                "happy",
                "neutral",
                "sad",
                "surprised",
            ]

        # Store in cache for subsequent calls
        _EMOTION_MODEL_CACHE[model_name] = (self.model, self.processor, self.labels)

    def _post_process(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Apply smoothing and confidence shaping to raw probabilities."""

        if self.prob_smoothing and self.prob_smoothing > 1 and len(prob_matrix) > 1:
            window = max(1, int(self.prob_smoothing))
            if window % 2 == 0:
                window += 1
            pad = window // 2
            padded = np.pad(prob_matrix, ((pad, pad), (0, 0)), mode="edge")
            smoothed = []
            for idx in range(len(prob_matrix)):
                smoothed.append(padded[idx : idx + window].mean(axis=0))
            prob_matrix = np.array(smoothed)

        if self.confidence_gamma and self.confidence_gamma != 1.0:
            prob_matrix = np.power(
                np.clip(prob_matrix, 1e-8, 1.0), self.confidence_gamma
            )
            prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

        return prob_matrix

    def predict(
        self,
        windows: Sequence[np.ndarray],
        *,
        sampling_rate: int = 16000,
        start_index: int = 0,
    ) -> List[_EmotionPrediction]:
        """Return emotion predictions for pre-extracted audio windows."""

        if not windows:
            return []

        probs_history: List[np.ndarray] = []
        indices: List[int] = []

        for offset in range(0, len(windows), self.batch_size):
            batch = list(windows[offset : offset + self.batch_size])
            inputs = self.processor(
                batch,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                if self.temperature and self.temperature != 1.0:
                    logits = logits / self.temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)

            probs_history.extend(probs.cpu().numpy())
            indices.extend(
                range(start_index + offset, start_index + offset + len(batch))
            )

        matrix = np.array(probs_history)
        matrix = self._post_process(matrix)

        predictions: List[_EmotionPrediction] = []
        for idx, probs in zip(indices, matrix):
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            predictions.append(
                _EmotionPrediction(
                    label=self.labels[top_idx],
                    confidence=confidence,
                    scores={self.labels[i]: float(probs[i]) for i in range(len(probs))},
                    window_index=idx,
                )
            )

        return predictions

    def confident_predictions(
        self,
        windows: Sequence[np.ndarray],
        *,
        sampling_rate: int = 16000,
        start_index: int = 0,
    ) -> Iterable[_EmotionPrediction]:
        """Yield predictions that exceed the configured confidence threshold."""

        for pred in self.predict(
            windows, sampling_rate=sampling_rate, start_index=start_index
        ):
            if pred.confidence >= self.threshold:
                yield pred


def _analyze(
    audio_path: Union[str, Path],
    working_dir: Union[str, Path],
    *,
    analyzer: Optional[_EmotionAnalyzer] = None,
    use_forced_alignment: bool = True,
    target_sample_rate: int = 16000,
    min_duration: float = 0.35,
    pad_seconds: float = 0.15,
    forced_alignment_path: Optional[Union[str, Path]] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """High-level helper that runs emotion analysis for the workspace layout."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    working_dir = Path(working_dir)
    output_dir = working_dir / "emotions"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = analyzer or _EmotionAnalyzer(debug=debug)

    # Forced alignment provides better timestamps when available.
    if use_forced_alignment:
        forced_path = (
            Path(forced_alignment_path)
            if forced_alignment_path is not None
            else working_dir / "ctc" / "alignment.json"
        )
        if forced_path.is_file():
            LOGGER.info(
                "Running emotion analysis using forced alignment from %s", forced_path
            )
            debug_print(
                f"Using forced alignment segments from {forced_path}",
                debug=debug,
            )
            try:
                result = _analyze_segments_from_file(
                    audio_path=audio_path,
                    segments_path=forced_path,
                    analyzer=analyzer,
                    output_dir=output_dir,
                    output_name="emotions.json",
                    target_sample_rate=target_sample_rate,
                    min_duration=min_duration,
                    pad_seconds=pad_seconds,
                    debug=debug,
                )
                if result:
                    return result
            except Exception as exc:  # pragma: no cover - surfaced to CLI
                LOGGER.warning(
                    "Emotion analysis using forced alignment failed: %s", exc
                )

    transcript_path = working_dir / "transcription" / "transcription.json"
    if transcript_path.is_file():
        LOGGER.info(
            "Running emotion analysis using transcription from %s", transcript_path
        )
        debug_print(
            f"Falling back to transcription segments from {transcript_path}",
            debug=debug,
        )
        try:
            return _analyze_segments_from_file(
                audio_path=audio_path,
                segments_path=transcript_path,
                analyzer=analyzer,
                output_dir=output_dir,
                output_name="emotions.json",
                target_sample_rate=target_sample_rate,
                min_duration=min_duration,
                pad_seconds=pad_seconds,
                debug=debug,
            )
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            LOGGER.warning("Emotion analysis using transcription failed: %s", exc)
            raise

    LOGGER.warning(
        "No transcript context available for emotion analysis in %s", working_dir
    )
    return None


def handle(
    input_file: str,
    output_folder: str,
    config: "EmotionConfig | None" = None,
    *,
    segments_file: Optional[str] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Perform emotion recognition on audio segments.

    Args:
        input_file: Path to input audio file.
        output_folder: Path to output directory.
        config: EmotionConfig instance or None for defaults.
        segments_file: Optional path to segments JSON (transcript or forced alignment).
        debug: Enable verbose debug output.

    Returns:
        Emotion analysis results or None if no segments available.
    """
    print("INFO: Performing emotion analysis")

    # Extract config values with defaults
    model_name = (
        config.model_name
        if config
        else "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    device = config.device if config else None
    batch_size = config.batch_size if config else 8
    temperature = config.temperature if config else 0.7
    prob_smoothing = config.prob_smoothing if config else 5
    confidence_gamma = config.confidence_gamma if config else 1.35
    threshold = config.threshold if config else 0.12
    target_sample_rate = config.target_sample_rate if config else 16000
    min_duration = config.min_duration if config else 0.35
    pad_seconds = config.pad_seconds if config else 0.15
    use_forced_alignment = config.use_forced_alignment if config else True

    # Create analyzer with config values
    analyzer = _EmotionAnalyzer(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        temperature=temperature,
        prob_smoothing=prob_smoothing,
        confidence_gamma=confidence_gamma,
        threshold=threshold,
        debug=debug,
    )

    return _analyze(
        audio_path=input_file,
        working_dir=output_folder,
        analyzer=analyzer,
        use_forced_alignment=use_forced_alignment,
        target_sample_rate=target_sample_rate,
        min_duration=min_duration,
        pad_seconds=pad_seconds,
        forced_alignment_path=segments_file,
        debug=debug,
    )


def _coerce_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _prepare_segment_windows(segments: Sequence[dict]) -> List[_SegmentWindow]:
    windows: List[_SegmentWindow] = []
    for index, segment in enumerate(segments):
        start = _coerce_float(segment.get("start"))
        end = _coerce_float(segment.get("end"))
        if start is None or end is None or end <= start:
            continue

        windows.append(
            _SegmentWindow(
                index=index,
                start=float(start),
                end=float(end),
                text=_normalize_text(segment.get("text")),
                source_id=segment.get("id"),
                payload=dict(segment),
            )
        )

    return windows


def _load_audio_waveform(
    audio_path: Union[str, Path],
    target_sr: int,
    *,
    debug: bool = False,
) -> Tuple[np.ndarray, int]:
    with gray_debug_output(debug):
        waveform, sample_rate = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return waveform, int(sample_rate)


def _summarize_predictions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "total_segments": 0,
            "segments_analyzed": 0,
            "emotion_counts": {},
        }

    counts = Counter(result["emotion"] for result in results)
    confidences = [result["confidence"] for result in results]

    return {
        "total_segments": len(results),
        "segments_analyzed": len(results),
        "emotion_counts": dict(counts),
        "average_confidence": mean(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
    }


def _analyze_segments(
    audio_path: Union[str, Path],
    segments: Sequence[dict],
    *,
    analyzer: Optional[_EmotionAnalyzer] = None,
    target_sample_rate: int = 16000,
    min_duration: float = 0.35,
    pad_seconds: float = 0.15,
    output_path: Optional[Union[str, Path]] = None,
    source: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run emotion analysis on timestamped segments.

    Parameters
    ----------
    audio_path:
        Path to the audio waveform used for inference.
    segments:
        Iterable of dictionaries containing at least ``start`` and ``end`` keys.
    analyzer:
    Optional pre-initialised emotion analyzer instance.
    target_sample_rate:
        Target sampling rate for loading the waveform (default ``16000``).
    min_duration:
        Minimum clip duration in seconds. Shorter segments are padded.
    pad_seconds:
        Context padding (lead/trail) to add around each segment before slicing.
    output_path:
        Optional path to persist the resulting JSON payload.
    source:
        Metadata about the origin of the segments (e.g. ``transcription``).
    """

    segment_windows = _prepare_segment_windows(segments)
    if not segment_windows:
        return None

    audio_path = Path(audio_path)
    waveform, sample_rate = _load_audio_waveform(
        audio_path,
        target_sample_rate,
        debug=debug,
    )
    duration = len(waveform) / float(sample_rate) if sample_rate else 0.0

    analyzer = analyzer or _EmotionAnalyzer(debug=debug)

    min_samples = int(max(min_duration, 0.0) * sample_rate) if min_duration > 0 else 0

    audio_windows: List[np.ndarray] = []
    metadata: List[_SegmentWindow] = []
    skipped = 0

    for window in segment_windows:
        start = max(0.0, window.start - pad_seconds)
        end = min(duration, window.end + pad_seconds)

        if end <= start:
            skipped += 1
            continue

        start_idx = max(0, int(round(start * sample_rate)))
        end_idx = min(len(waveform), int(round(end * sample_rate)))
        clip = waveform[start_idx:end_idx]

        if clip.size == 0:
            skipped += 1
            continue

        if min_samples and clip.size < min_samples:
            pad_amount = min_samples - clip.size
            clip = np.pad(clip, (0, pad_amount), mode="edge")

        audio_windows.append(clip)
        metadata.append(window)

    if not audio_windows:
        return None

    predictions = analyzer.predict(audio_windows, sampling_rate=sample_rate)
    results: List[Dict[str, Any]] = []

    for window, prediction in zip(metadata, predictions):
        record: Dict[str, Any] = {
            "segment_index": window.index,
            "segment_id": window.source_id
            if window.source_id is not None
            else window.index,
            "start": window.start,
            "end": window.end,
            "duration": window.end - window.start,
            "text": window.text,
            "emotion": prediction.label,
            "confidence": prediction.confidence,
            "scores": prediction.scores,
        }

        if "speaker" in window.payload:
            record["speaker"] = window.payload["speaker"]
        if "speakers" in window.payload and isinstance(
            window.payload["speakers"], list
        ):
            record["speakers"] = window.payload["speakers"]
        if "words" in window.payload and isinstance(window.payload["words"], list):
            record["word_count"] = len(window.payload["words"])

        results.append(record)

    payload = {
        "audio": str(audio_path),
        "sample_rate": sample_rate,
        "source": source or {},
        "settings": {
            "min_duration": min_duration,
            "pad_seconds": pad_seconds,
            "threshold": analyzer.threshold,
            "temperature": analyzer.temperature,
            "prob_smoothing": analyzer.prob_smoothing,
            "confidence_gamma": analyzer.confidence_gamma,
        },
        "summary": {
            **_summarize_predictions(results),
            "segments_provided": len(segment_windows),
            "segments_skipped": skipped + (len(segment_windows) - len(metadata)),
        },
        "segments": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        payload["output_path"] = str(output_path)

    return payload


def _infer_source_type(payload: Dict[str, Any]) -> str:
    if "transcript" in payload and "audio" in payload:
        return "forced_alignment"
    if "transcription" in payload:
        return "transcription"
    return "segments"


def _analyze_segments_from_file(
    audio_path: Union[str, Path],
    segments_path: Union[str, Path],
    *,
    analyzer: Optional[_EmotionAnalyzer] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_name: Optional[str] = None,
    target_sample_rate: int = 16000,
    min_duration: float = 0.35,
    pad_seconds: float = 0.15,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load transcript-style JSON segments and compute emotion predictions."""

    segments_path = Path(segments_path)
    if not segments_path.is_file():
        return None

    with segments_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return None

    source_type = _infer_source_type(payload)
    source_info = {
        "type": source_type,
        "path": str(segments_path),
        "segment_count": len(raw_segments),
    }

    output_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_name or "emotions.json"
        output_path = output_dir / filename

    return _analyze_segments(
        audio_path=audio_path,
        segments=raw_segments,
        analyzer=analyzer,
        target_sample_rate=target_sample_rate,
        min_duration=min_duration,
        pad_seconds=pad_seconds,
        output_path=output_path,
        source=source_info,
        debug=debug,
    )
