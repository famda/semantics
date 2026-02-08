"""Forced alignment utilities using NeMo CTC models.

This module provides a helper to align an existing transcript to an audio file
using character-based CTC acoustic models from NVIDIA NeMo together with the
`ctc-segmentation` library. The resulting word-level timestamps can then serve
as an authoritative timeline that downstream tooling.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

from nemo.collections.asr.models import EncDecCTCModel
from .utils.logging import debug_print, gray_debug_output

__all__ = ["handle"]

if TYPE_CHECKING:
    from ..config import CtcConfig

try:
    from ctc_segmentation import (
        CtcSegmentationParameters,
        ctc_segmentation,
        determine_utterance_segments,
        prepare_text,
    )
except ImportError as exc:
    raise RuntimeError(
        "The 'ctc-segmentation' package is required for forced alignment."
    ) from exc

_SANITIZE_PATTERN = re.compile(r"[^a-z' ]+")
_APOSTROPHE_VARIANTS = {"\u2018": "'", "\u2019": "'", "\u02bc": "'"}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordEntry:
    index: int
    text: str
    sanitized: Optional[str]
    segment_index: int
    segment_start: float
    segment_end: float
    original_start: Optional[float]
    original_end: Optional[float]


def _coerce_float(value: object) -> Optional[float]:
    """Safely coerce a value to float."""
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_apostrophes(text: str) -> str:
    """Normalize various apostrophe characters to standard ASCII."""
    for original, replacement in _APOSTROPHE_VARIANTS.items():
        text = text.replace(original, replacement)
    return text


def _sanitize_word(text: str, allowed_characters: Sequence[str]) -> Optional[str]:
    """Map a transcript token to the CTC model vocabulary.

    Returns None when the token does not contain any characters supported by the
    model (for example, pure punctuation)."""
    text = _normalize_apostrophes(text.lower())
    text = text.replace("-", " ")
    text = _SANITIZE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    filtered = "".join(ch for ch in text if ch in allowed_characters)
    return filtered or None


def _fallback_timing(
    entry: WordEntry, audio_duration: float
) -> Tuple[Optional[float], Optional[float]]:
    """Get fallback timing for a word entry."""
    start = (
        entry.original_start
        if entry.original_start is not None
        else entry.segment_start
    )
    end = entry.original_end if entry.original_end is not None else entry.segment_end
    if start is not None:
        start = max(0.0, min(float(start), audio_duration))
    if end is not None:
        end = max(start if start is not None else 0.0, min(float(end), audio_duration))
    return start, end


def _load_diarization_segments(diarization_path: Path) -> List[Dict[str, object]]:
    if not diarization_path.is_file():
        return []

    try:
        with diarization_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        LOGGER.warning(
            "Failed to read diarization data from %s: %s", diarization_path, exc
        )
        return []

    segments: List[Dict[str, object]] = []
    if not isinstance(payload, list):
        LOGGER.warning(
            "Unexpected diarization payload structure in %s", diarization_path
        )
        return segments

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        start = _coerce_float(entry.get("start"))
        end = _coerce_float(entry.get("end"))
        speaker = entry.get("speaker")
        if start is None or end is None or not isinstance(speaker, str):
            continue
        if end <= start:
            continue

        sanitized_entry = dict(entry)
        sanitized_entry["start"] = float(start)
        sanitized_entry["end"] = float(end)
        segments.append(sanitized_entry)

    # start is guaranteed to be float after sanitization above
    segments.sort(key=lambda item: float(item.get("start") or 0.0))  # type: ignore[arg-type]
    return segments


def _collect_transcript_words(
    transcript_path: Path, allowed: Sequence[str]
) -> Tuple[List[WordEntry], List[dict]]:
    with transcript_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Transcription JSON must contain a 'segments' list")

    words: List[WordEntry] = []
    counter = 0
    for segment_index, segment in enumerate(segments):
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        try:
            seg_start_f = float(seg_start) if seg_start is not None else 0.0
        except (TypeError, ValueError):
            seg_start_f = 0.0
        try:
            seg_end_f = float(seg_end) if seg_end is not None else seg_start_f
        except (TypeError, ValueError):
            seg_end_f = seg_start_f

        for maybe_word in segment.get("words") or []:
            raw_text = str(maybe_word.get("word", "")).strip()
            if not raw_text:
                counter += 1
                continue

            sanitized = _sanitize_word(raw_text, allowed)
            start_val = maybe_word.get("start")
            end_val = maybe_word.get("end")
            try:
                word_start = float(start_val) if start_val is not None else None
            except (TypeError, ValueError):
                word_start = None
            try:
                word_end = float(end_val) if end_val is not None else None
            except (TypeError, ValueError):
                word_end = None

            words.append(
                WordEntry(
                    index=counter,
                    text=raw_text,
                    sanitized=sanitized,
                    segment_index=segment_index,
                    segment_start=seg_start_f,
                    segment_end=seg_end_f,
                    original_start=word_start,
                    original_end=word_end,
                )
            )
            counter += 1

    if not words:
        raise ValueError("No per-word entries found in transcription JSON")

    return words, segments


def _load_audio_waveform(
    audio_path: Path, target_sample_rate: int
) -> Tuple[np.ndarray, int]:
    signal, sample_rate = sf.read(str(audio_path), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if sample_rate != target_sample_rate:
        import librosa

        signal = librosa.resample(
            signal, orig_sr=sample_rate, target_sr=target_sample_rate
        )
        sample_rate = target_sample_rate

    return np.asarray(signal, dtype=np.float32), sample_rate


def _slice_audio_chunk(
    waveform: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    start_sample = max(0, int(round(start_sec * sample_rate)))
    end_sample = min(len(waveform), int(round(end_sec * sample_rate)))
    if end_sample <= start_sample:
        return None

    chunk = waveform[start_sample:end_sample]
    tensor = torch.from_numpy(chunk.copy()).unsqueeze(0)
    lengths = torch.tensor([tensor.shape[1]], dtype=torch.int64)
    return tensor, lengths


@lru_cache(maxsize=4)
def _load_model(model_key: str, device: str, *, debug: bool = False) -> EncDecCTCModel:
    """Load and cache the requested NeMo CTC model."""

    debug_print(f"Loading NeMo CTC model '{model_key}' on {device}", debug=debug)

    torch_device = torch.device(device)
    with gray_debug_output(debug):
        if model_key.endswith(".nemo"):
            model = EncDecCTCModel.restore_from(model_key, map_location=torch_device)
        else:
            model = EncDecCTCModel.from_pretrained(
                model_name=model_key, map_location=torch_device
            )

        model = model.to(torch_device)  # type: ignore[union-attr]
        model.preprocessor.to(torch_device)  # type: ignore[union-attr]

    model.eval()  # type: ignore[union-attr]
    return model  # type: ignore[return-value]


def _prepare_ctc_parameters(
    model: EncDecCTCModel, frame_duration: float
) -> CtcSegmentationParameters:
    vocabulary: List[str] = list(model.decoder.vocabulary)  # type: ignore[arg-type]
    blank_symbol = "<blank>"

    if blank_symbol in vocabulary:
        raise ValueError(
            "Unexpected vocabulary containing <blank>; please inspect model configuration"
        )

    char_list = vocabulary + [blank_symbol]

    params = CtcSegmentationParameters(char_list=char_list)
    params.blank = len(char_list) - 1
    params.space = " " if " " in vocabulary else ""  # type: ignore[assignment]
    params.index_duration = frame_duration
    params.frame_duration_ms = frame_duration * 1000.0  # type: ignore[assignment]
    params.subsampling_factor = 1  # type: ignore[assignment]
    params.max_window_size = max(params.max_window_size, 6000)
    params.update_excluded_characters()
    return params


def _align_transcript(
    audio_path: Path,
    transcript_path: Path,
    output_path: Path,
    *,
    segment_margin: float = 0.75,
    segment_backoffs: Tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
    min_overlap: float = 0.0,
    model: str = "stt_en_quartznet15x5",
    device: str = "cpu",
    debug: bool = False,
) -> Path:
    """Align transcript to audio and write alignment JSON."""
    audio_path = audio_path.expanduser().resolve()
    transcript_path = transcript_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    debug_print(f"Running forced alignment with model {model} on {device}", debug=debug)

    nemo_model = _load_model(model_key=model, device=device, debug=debug)

    sample_rate = int(getattr(nemo_model.cfg.preprocessor, "sample_rate", 16000))
    with gray_debug_output(debug):
        waveform, sample_rate = _load_audio_waveform(
            audio_path, target_sample_rate=sample_rate
        )
    audio_duration = len(waveform) / float(sample_rate)

    allowed_characters: Sequence[str] = list(nemo_model.decoder.vocabulary)  # type: ignore[arg-type]
    word_entries, transcript_segments = _collect_transcript_words(
        transcript_path, allowed=allowed_characters
    )

    diarization_path = (
        transcript_path.parent.parent / "diarization" / "diarization.json"
    )
    diarization_segments: List[Dict[str, object]] = _load_diarization_segments(
        diarization_path
    )
    if diarization_segments:
        LOGGER.info(
            "Loaded %s diarization segments from %s",
            len(diarization_segments),
            diarization_path,
        )
        debug_print(
            f"Loaded {len(diarization_segments)} diarization segments from {diarization_path}",
            debug=debug,
        )

    sanitized_count = sum(1 for entry in word_entries if entry.sanitized)
    if sanitized_count == 0:
        raise RuntimeError(
            "None of the transcript tokens could be mapped to the CTC vocabulary"
        )

    index_to_timing: Dict[int, Tuple[float, float, Optional[float]]] = {}
    aligned_segments = 0
    frame_duration_sample: Optional[float] = None

    for segment_index in range(len(transcript_segments)):
        segment_words = [
            entry for entry in word_entries if entry.segment_index == segment_index
        ]
        sanitized_pairs = [
            (entry.index, entry.sanitized) for entry in segment_words if entry.sanitized
        ]
        if not sanitized_pairs:
            continue

        first_word = segment_words[0]
        last_word = segment_words[-1]

        alignment_success = False
        frame_duration = None
        last_exception: Optional[Exception] = None

        for extra_margin in segment_backoffs:
            margin = segment_margin + extra_margin
            chunk_start = max(0.0, first_word.segment_start - margin)
            chunk_end = min(audio_duration, last_word.segment_end + margin)
            if chunk_end <= chunk_start:
                chunk_end = min(audio_duration, chunk_start + 2.0)
                if chunk_end <= chunk_start:
                    continue

            chunk = _slice_audio_chunk(waveform, sample_rate, chunk_start, chunk_end)
            if chunk is None:
                continue

            chunk_audio, chunk_lengths = chunk
            chunk_duration = chunk_audio.shape[1] / float(sample_rate)

            chunk_audio = chunk_audio.to(device)
            chunk_lengths = chunk_lengths.to(device)

            with torch.inference_mode():
                processed_signal, processed_length = nemo_model.preprocessor(
                    input_signal=chunk_audio, length=chunk_lengths
                )
                processed_signal = processed_signal.to(device)
                processed_length = processed_length.to(device)

                log_probs, encoded_length, _ = nemo_model(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_length,
                )

            valid_frames = int(encoded_length[0].item())
            if valid_frames == 0:
                continue

            log_probs_np = log_probs[0, :valid_frames].cpu().numpy()
            frame_duration = chunk_duration / max(valid_frames, 1)
            params = _prepare_ctc_parameters(nemo_model, frame_duration=frame_duration)

            sanitized_indices, sanitized_tokens = zip(*sanitized_pairs)
            ground_truth, utt_begin_indices = prepare_text(
                params, list(sanitized_tokens)
            )
            try:
                timings, char_probs, _ = ctc_segmentation(
                    params, log_probs_np, ground_truth
                )
                local_segments = determine_utterance_segments(
                    params,
                    utt_begin_indices,
                    char_probs,
                    timings,
                    list(sanitized_tokens),
                )
            except Exception as exc:
                last_exception = exc
                if (
                    "Audio is shorter than text" in str(exc)
                    and extra_margin < segment_backoffs[-1]
                ):
                    continue

                LOGGER.warning(
                    "CTC segmentation failed for segment %s: %s", segment_index, exc
                )
                break

            if len(local_segments) != len(sanitized_tokens):
                mismatch_message = (
                    "Segment %s alignment mismatch (expected %s steps, got %s)"
                    % (
                        segment_index,
                        len(sanitized_tokens),
                        len(local_segments),
                    )
                )
                last_exception = RuntimeError(mismatch_message)
                if extra_margin < segment_backoffs[-1]:
                    continue
                LOGGER.warning(mismatch_message)
                break

            for (local_start, local_end, conf), idx in zip(
                local_segments, sanitized_indices
            ):
                start_abs = max(0.0, float(chunk_start + local_start))
                end_abs = max(start_abs, float(chunk_start + local_end))
                index_to_timing[idx] = (start_abs, end_abs, float(conf))

            alignment_success = True
            break

        if not alignment_success:
            if last_exception and "Audio is shorter than text" not in str(
                last_exception
            ):
                # Already logged inside loop for other errors
                pass
            else:
                LOGGER.warning(
                    "CTC segmentation failed for segment %s after margin backoffs: %s",
                    segment_index,
                    last_exception or "no audio window produced",
                )

            for entry in segment_words:
                fallback_start, fallback_end = _fallback_timing(entry, audio_duration)
                if fallback_start is None and fallback_end is None:
                    continue
                safe_start = (
                    float(fallback_start) if fallback_start is not None else 0.0
                )
                safe_end = (
                    float(fallback_end) if fallback_end is not None else safe_start
                )
                index_to_timing[entry.index] = (
                    safe_start,
                    max(safe_start, safe_end),
                    None,
                )
            continue

        aligned_segments += 1
        if frame_duration_sample is None and frame_duration is not None:
            frame_duration_sample = frame_duration

    words_out: List[Dict[str, object]] = []
    for entry in word_entries:
        maybe = index_to_timing.get(entry.index)
        if maybe:
            start, end, conf = maybe
        else:
            start = end = conf = None
        words_out.append(
            {
                "index": entry.index,
                "word": entry.text,
                "sanitized": entry.sanitized,
                "original_start": entry.original_start,
                "original_end": entry.original_end,
                "segment_index": entry.segment_index,
                "start": start,
                "end": end,
                "confidence": conf,
            }
        )

    segments_out: List[Dict[str, object]] = []
    segment_audio_slices: List[Tuple[str, float, float]] = []
    for segment_index, segment_payload in enumerate(transcript_segments):
        original_words = list(segment_payload.get("words") or [])
        forced_words = [
            word for word in words_out if word.get("segment_index") == segment_index
        ]

        original_segment_start = _coerce_float(segment_payload.get("start"))
        original_segment_end = _coerce_float(segment_payload.get("end"))

        if original_words and len(original_words) != len(forced_words):
            LOGGER.warning(
                "Segment %s word-count mismatch (transcript=%s, forced=%s)",
                segment_index,
                len(original_words),
                len(forced_words),
            )

        new_segment = copy.deepcopy(segment_payload)
        new_words: List[Dict[str, object]] = []

        for word_idx, orig_word in enumerate(original_words):
            forced_ref = (
                forced_words[word_idx] if word_idx < len(forced_words) else None
            )
            merged = dict(orig_word)
            orig_word_start = _coerce_float(orig_word.get("start"))
            orig_word_end = _coerce_float(orig_word.get("end"))
            if forced_ref:
                merged["start"] = forced_ref.get("start")
                merged["end"] = forced_ref.get("end")
                merged["sanitized"] = forced_ref.get("sanitized")
                merged["forced_index"] = forced_ref.get("index")
                merged["confidence_ctc"] = forced_ref.get("confidence")
                merged["original_start"] = forced_ref.get("original_start")
                merged["original_end"] = forced_ref.get("original_end")
            elif "start" not in merged:
                merged["start"] = None
                merged.setdefault("end", None)
                merged.setdefault("original_start", orig_word_start)
                merged.setdefault("original_end", orig_word_end)
            else:
                merged.setdefault("end", None)
                merged.setdefault("original_start", orig_word_start)
                merged.setdefault("original_end", orig_word_end)
            new_words.append(merged)

        new_segment["words"] = new_words

        aligned_starts: List[float] = [
            float(w.get("start"))  # type: ignore[arg-type]
            for w in new_words
            if isinstance(w.get("start"), (int, float))
        ]
        aligned_ends: List[float] = [
            float(w.get("end"))  # type: ignore[arg-type]
            for w in new_words
            if isinstance(w.get("end"), (int, float))
        ]

        if aligned_starts:
            new_segment["start"] = min(aligned_starts)
        elif original_segment_start is not None:
            new_segment["start"] = original_segment_start
        else:
            new_segment.setdefault("start", None)
        if aligned_ends:
            new_segment["end"] = max(aligned_ends)
        elif original_segment_end is not None:
            new_segment["end"] = original_segment_end
        else:
            new_segment.setdefault("end", None)

        if original_segment_start is not None:
            new_segment["original_start"] = original_segment_start
        else:
            new_segment.setdefault("original_start", None)
        if original_segment_end is not None:
            new_segment["original_end"] = original_segment_end
        else:
            new_segment.setdefault("original_end", None)

        segment_start_for_speakers = _coerce_float(new_segment.get("start"))
        segment_end_for_speakers = _coerce_float(new_segment.get("end"))
        if segment_start_for_speakers is None:
            segment_start_for_speakers = original_segment_start
        if segment_end_for_speakers is None:
            segment_end_for_speakers = original_segment_end

        speakers: List[Dict[str, object]] = []
        if (
            diarization_segments
            and segment_start_for_speakers is not None
            and segment_end_for_speakers is not None
            and segment_end_for_speakers > segment_start_for_speakers
        ):
            for diar_entry in diarization_segments:
                entry_start = _coerce_float(diar_entry.get("start"))
                entry_end = _coerce_float(diar_entry.get("end"))
                if entry_start is None or entry_end is None:
                    continue
                if entry_end <= segment_start_for_speakers:
                    continue
                if entry_start >= segment_end_for_speakers:
                    break

                overlap_start = max(segment_start_for_speakers, entry_start)
                overlap_end = min(segment_end_for_speakers, entry_end)
                if overlap_end - overlap_start >= min_overlap:
                    speaker_entry = dict(diar_entry)
                    speaker_entry["start"] = float(overlap_start)
                    speaker_entry["end"] = float(overlap_end)
                    speakers.append(speaker_entry)

        new_segment["speakers"] = speakers

        new_segment["alignment"] = {
            "source": "ctc",
            "aligned_words": len([w for w in new_words if w.get("start") is not None]),
            "total_words": len(new_words),
        }

        segment_identifier = new_segment.get("id", segment_index)
        if isinstance(segment_identifier, bool):
            segment_identifier = int(segment_identifier)
        if isinstance(segment_identifier, (int, float)) and not isinstance(
            segment_identifier, bool
        ):
            segment_label = str(int(segment_identifier))
        else:
            segment_label = str(segment_identifier)

        segment_start_audio = _coerce_float(new_segment.get("start"))
        segment_end_audio = _coerce_float(new_segment.get("end"))
        if segment_start_audio is None:
            segment_start_audio = original_segment_start
        if segment_end_audio is None:
            segment_end_audio = original_segment_end

        if (
            segment_label
            and segment_start_audio is not None
            and segment_end_audio is not None
            and segment_end_audio - segment_start_audio >= 1e-3
        ):
            segment_audio_slices.append(
                (segment_label, float(segment_start_audio), float(segment_end_audio))
            )

        segments_out.append(new_segment)

    if segment_audio_slices:
        segments_dir = output_path.parent / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        for existing in segments_dir.glob("*.wav"):
            try:
                existing.unlink()
            except Exception as exc:
                LOGGER.warning(
                    "Failed to remove old segment file %s: %s", existing, exc
                )

        for segment_label, start_sec, end_sec in segment_audio_slices:
            start_sample = max(0, int(round(start_sec * sample_rate)))
            end_sample = min(len(waveform), int(round(end_sec * sample_rate)))
            if end_sample <= start_sample:
                continue

            segment_waveform = waveform[start_sample:end_sample]
            target_path = segments_dir / f"{segment_label}.wav"
            try:
                sf.write(str(target_path), segment_waveform, sample_rate)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to write segment %s to %s: %s",
                    segment_label,
                    target_path,
                    exc,
                )

    coverage = sum(
        1 for _, (start, end, _) in index_to_timing.items() if end is not None
    )
    payload = {
        "audio": str(audio_path),
        "transcript": str(transcript_path),
        "model": {
            "name": model,
            "sample_rate": sample_rate,
            "device": device,
            "frame_duration": frame_duration_sample,
        },
        "segments": segments_out,
        "metrics": {
            "total_words": len(word_entries),
            "aligned_words": coverage,
            "unaligned_words": len(word_entries) - coverage,
            "aligned_segments": aligned_segments,
            "segment_count": len(transcript_segments),
            "segment_margin_seconds": segment_margin,
            "audio_duration_seconds": audio_duration,
            "frame_duration_seconds": frame_duration_sample,
        },
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    return output_path


def handle(
    input_file: str,
    output_folder: str,
    config: "CtcConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[Path]:
    """Perform CTC forced alignment on transcription.

    This is the standardized entry point for the CTC alignment module.

    Args:
        input_file: Path to the input audio file.
        output_folder: Directory where output files will be written.
        config: CtcConfig instance with alignment parameters, or None for defaults.
        debug: If True, emit verbose debug output.

    Returns:
        Path to the alignment output JSON, or None if transcription is missing.
    """
    # Extract config values with defaults
    segment_margin = config.segment_margin_seconds if config else 0.75
    segment_backoffs = (
        config.segment_margin_backoffs if config else (0.0, 2.0, 5.0, 10.0)
    )
    min_overlap = config.min_speaker_overlap_seconds if config else 0.0

    audio_path = Path(input_file)
    working_dir = Path(output_folder)
    transcript_path = working_dir / "transcription" / "transcription.json"

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not transcript_path.is_file():
        LOGGER.warning(
            "Transcript JSON not found at %s; skipping forced alignment",
            transcript_path,
        )
        return None

    ctc_dir = working_dir / "ctc"
    ctc_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctc_dir / "alignment.json"

    # Auto-detect device: use CUDA when available unless overridden
    import torch as _torch
    device = config.device if config and config.device else None
    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    try:
        _align_transcript(
            audio_path=audio_path,
            transcript_path=transcript_path,
            output_path=output_path,
            segment_margin=segment_margin,
            segment_backoffs=segment_backoffs,
            min_overlap=min_overlap,
            device=device,
            debug=debug,
        )
        LOGGER.info("Alignment written to %s", output_path)
        return output_path
    except Exception as exc:
        LOGGER.error("Alignment failed: %s", exc)
        raise
