"""Speech transcription module.

Transcribes audio to text using Whisper models via faster-whisper.
Supports chunked processing for long audio files with overlap to avoid
boundary artifacts.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import torch
from faster_whisper import WhisperModel

from .utils.chunks import AudioChunk, cleanup_chunks, split_audio, split_audio_with_overlap
from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import TranscribeConfig

# Module-level model cache for reuse across multiple calls
_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


def handle(
    input_file: str,
    output_folder: str,
    config: "TranscribeConfig | None" = None,
    *,
    debug: bool = False,
) -> tuple[dict, str]:
    """Transcribe an audio file to text.

    Args:
        input_file: Path to the input audio file.
        output_folder: Path to the output folder for results.
        config: TranscribeConfig with model and chunking settings.
        debug: Enable verbose logging.

    Returns:
        Tuple of (transcription_data dict, full_transcription string).
    """
    # Extract config values (use defaults if config is None)
    model_name = config.model if config else "distil-large-v3.5"
    chunk_length = config.chunk_length if config else 900
    chunk_overlap = config.chunk_overlap_seconds if config else 5.0
    epsilon = config.segment_epsilon_seconds if config else 0.3
    model_attempts = list(config.model_attempts) if config and config.model_attempts else [
        ("cuda", "float16"),
        ("cuda", "int8_float16"),
        ("cpu", "int8"),
    ]

    print("INFO: Transcribing the audio file")

    transcription_folder = os.path.join(output_folder, "transcription")
    os.makedirs(transcription_folder, exist_ok=True)

    # Split audio into chunks with overlap
    chunk_infos, chunk_dir, overlap_used = _split_audio_chunks(
        input_file, output_folder, chunk_length, chunk_overlap, debug
    )

    # Load model with device fallback
    attempt_idx = 0
    model, attempt_idx = _load_model(model_name, model_attempts, attempt_idx, debug)

    # Transcribe all chunks
    segments: List[dict] = []
    languages: set[str] = set()
    segment_id = 0

    try:
        for idx, chunk in enumerate(chunk_infos, start=1):
            print(f"INFO: Processing chunk {idx}/{len(chunk_infos)}: {chunk.path}")

            while True:
                try:
                    with gray_debug_output(debug):
                        seg_iter, info = model.transcribe(
                            chunk.path,
                            beam_size=5,
                            word_timestamps=True,
                            vad_filter=True,
                        )

                    # Process segments from this chunk
                    for seg in seg_iter:
                        seg_dict = asdict(seg)
                        seg_dict.pop("tokens", None)
                        seg_dict["text"] = (seg_dict.get("text") or "").strip()

                        # Apply time offset from chunk start
                        seg_dict["start"] = (seg_dict.get("start") or 0.0) + chunk.start
                        seg_dict["end"] = (seg_dict.get("end") or 0.0) + chunk.start

                        # Clean and offset words
                        words = seg_dict.get("words") or []
                        cleaned_words = []
                        for w in words:
                            if not w:
                                continue
                            cleaned_words.append({
                                "word": (w.get("word") or "").strip(),
                                "start": (w.get("start") or 0.0) + chunk.start,
                                "end": (w.get("end") or 0.0) + chunk.start,
                            })
                        seg_dict["words"] = cleaned_words

                        # Trim segment to core region (skip overlap portions)
                        if chunk.core_start > 0:
                            if seg_dict["end"] <= chunk.core_start + epsilon:
                                continue  # Segment ends before core region
                            if seg_dict["start"] < chunk.core_start - epsilon:
                                # Trim words and recalculate boundaries
                                trimmed = [w for w in cleaned_words if w["end"] > chunk.core_start]
                                if trimmed:
                                    for w in trimmed:
                                        w["start"] = max(w["start"], chunk.core_start)
                                    seg_dict["words"] = trimmed
                                    seg_dict["start"] = trimmed[0]["start"]
                                    seg_dict["end"] = max(w["end"] for w in trimmed)
                                    seg_dict["text"] = " ".join(w["word"] for w in trimmed if w["word"]).strip()
                                else:
                                    seg_dict["start"] = max(seg_dict["start"], chunk.core_start)
                                    if seg_dict["end"] - seg_dict["start"] <= epsilon:
                                        continue

                        # Skip empty segments
                        if not seg_dict.get("text"):
                            continue

                        # Skip duplicates at chunk boundaries
                        if segments:
                            prev = segments[-1]
                            if (
                                prev.get("text") == seg_dict["text"]
                                and (
                                    seg_dict["start"] <= prev["end"] + epsilon
                                    or abs(prev["start"] - seg_dict["start"]) <= epsilon
                                )
                            ):
                                continue

                        seg_dict["id"] = segment_id
                        segment_id += 1
                        segments.append(seg_dict)

                    if info.language:
                        languages.add(info.language)
                        debug_print(
                            f"Detected language '{info.language}' with probability {info.language_probability}",
                            debug=debug,
                        )
                    break

                except KeyboardInterrupt:
                    raise
                except BaseException as exc:
                    err_msg = str(exc).lower()
                    is_gpu_error = any(t in err_msg for t in ("cuda", "cudnn", "cublas", "gpu", "out of memory"))
                    can_retry = attempt_idx + 1 < len(model_attempts)

                    if is_gpu_error and can_retry:
                        print(
                            f"WARN: GPU transcription failed ({exc}). "
                            f"Falling back to {model_attempts[attempt_idx + 1][0]}/{model_attempts[attempt_idx + 1][1]}."
                        )
                        # Clear cache entry and free GPU memory
                        cache_key = (model_name, model_attempts[attempt_idx][0], model_attempts[attempt_idx][1])
                        _MODEL_CACHE.pop(cache_key, None)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        attempt_idx += 1
                        model, attempt_idx = _load_model(model_name, model_attempts, attempt_idx, debug)
                        continue
                    raise
    finally:
        cleanup_chunks(chunk_dir)

    # Build output data
    full_text = " ".join(s["text"] for s in segments if s.get("text"))

    result = {
        "transcription": full_text,
        "segments": segments,
        "languages": sorted(languages),
        "model": {
            "name": model_name,
            "device": model_attempts[attempt_idx][0],
            "compute_type": model_attempts[attempt_idx][1],
        },
        "chunking": {
            "length_seconds": chunk_length,
            "overlap_seconds": overlap_used,
            "chunk_count": len(chunk_infos),
        },
    }

    # Write output files
    with open(os.path.join(transcription_folder, "transcription.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(os.path.join(transcription_folder, "transcription.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    with open(os.path.join(transcription_folder, "transcription.srt"), "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start_ts = _format_srt_time(seg["start"])
            end_ts = _format_srt_time(seg["end"])
            f.write(f"{i + 1}\n{start_ts} --> {end_ts}\n{seg['text']}\n\n")

    return result, full_text


def _split_audio_chunks(
    input_file: str,
    output_folder: str,
    chunk_length: int,
    chunk_overlap: float,
    debug: bool,
) -> tuple[List[AudioChunk], Path | None, float]:
    """Split audio into chunks with overlap, falling back to simple split if needed."""
    overlap_used = 0.0
    try:
        with gray_debug_output(debug):
            chunk_infos, chunk_dir = split_audio_with_overlap(
                input_file, output_folder, "transcribe", chunk_length, chunk_overlap
            )
        if len(chunk_infos) > 1 and any(c.start < c.core_start for c in chunk_infos[1:]):
            overlap_used = chunk_overlap
        return chunk_infos, chunk_dir, overlap_used
    except RuntimeError:
        with gray_debug_output(debug):
            fallback_paths, chunk_dir = split_audio(input_file, output_folder, "transcribe", chunk_length)

        # Compute offsets for fallback chunks
        offsets = [0.0] * len(fallback_paths)
        if fallback_paths:
            try:
                from .utils.chunks import compute_chunk_offsets
                with gray_debug_output(debug):
                    offsets = compute_chunk_offsets(fallback_paths)
            except Exception:
                pass

        chunk_infos = [
            AudioChunk(
                path=p,
                start=off,
                end=off + chunk_length,
                core_start=off,
                core_end=off + chunk_length,
            )
            for p, off in zip(fallback_paths, offsets)
        ]
        return chunk_infos, chunk_dir, 0.0


def _load_model(
    model_name: str,
    model_attempts: List[Tuple[str, str]],
    attempt_idx: int,
    debug: bool,
) -> tuple[WhisperModel, int]:
    """Load Whisper model with device/precision fallback."""
    while attempt_idx < len(model_attempts):
        device, compute_type = model_attempts[attempt_idx]
        cache_key = (model_name, device, compute_type)

        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key], attempt_idx

        debug_print(f"INFO: Loading Whisper model '{model_name}' on {device} ({compute_type})", debug=debug)

        try:
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            with gray_debug_output(debug):
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _MODEL_CACHE[cache_key] = model
            return model, attempt_idx

        except BaseException as exc:
            err_msg = str(exc).lower()
            print(f"WARN: Whisper model initialization error: {err_msg or exc}", flush=True)

            can_retry = attempt_idx + 1 < len(model_attempts)
            should_retry = can_retry and (device == "cuda" or "cuda" in err_msg or "out of memory" in err_msg)

            if should_retry:
                next_device, next_compute = model_attempts[attempt_idx + 1]
                print(f"WARN: Falling back to {next_device} ({next_compute}) configuration.", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                attempt_idx += 1
                continue
            raise

    raise RuntimeError("Unable to load Whisper model with available configurations")


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    ms = int((seconds % 1) * 1000)
    total_secs = int(seconds)
    hrs, remainder = divmod(total_secs, 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"