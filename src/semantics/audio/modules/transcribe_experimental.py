"""Experimental transcription module using HuggingFace Transformers.

This module provides fast transcription using Whisper's native long-form
transcription mechanism (sequential chunking with condition-on-previous-tokens)
combined with Flash Attention 2 (or SDPA fallback) for optimized attention.

Key features:
- Native Whisper long-form transcription (model.generate() with return_timestamps)
- Sequential chunking with condition-on-previous-tokens for accuracy
- Flash Attention 2 / SDPA for optimized attention computation
- Word-level timestamps from Whisper
- Optional CTC post-processing to refine word boundaries
- Speaker diarization with NeMo MSDD
- Compatible output format with existing transcribe.py for A/B comparison

Note: Uses model.generate() directly instead of pipeline() for long-form audio,
as recommended by the Whisper paper Section 3.8 and HuggingFace documentation.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import re
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import TranscribeExperimentalConfig

__all__ = ["handle"]


# =============================================================================
# Default Constants
# =============================================================================

# Use distil-whisper for faster inference while maintaining good accuracy.
# This matches the same model used in standard transcription (distil-large-v3.5)
# via faster-whisper for compatibility in A/B comparisons.
DEFAULT_MODEL = "distil-whisper/distil-large-v3.5"
# Chunk length for Whisper's native long-form transcription (30s is optimal)
DEFAULT_CHUNK_LENGTH_S = 30
DEFAULT_USE_FLASH_ATTENTION = True
DEFAULT_ENABLE_CTC_REFINEMENT = True
DEFAULT_CTC_MODEL = "stt_en_quartznet15x5"

# Regex for sanitizing text for CTC alignment
_SANITIZE_PATTERN = re.compile(r"[^a-z' ]+")
_APOSTROPHE_VARIANTS = {"\u2018": "'", "\u2019": "'", "\u02bc": "'"}


# =============================================================================
# Settings Dataclass
# =============================================================================


@dataclass
class _TranscribeSettings:
    """Internal settings for transcription."""

    model: str = DEFAULT_MODEL
    chunk_length_s: int = DEFAULT_CHUNK_LENGTH_S
    use_flash_attention: bool = DEFAULT_USE_FLASH_ATTENTION
    language: Optional[str] = None
    enable_ctc_refinement: bool = DEFAULT_ENABLE_CTC_REFINEMENT
    ctc_model: str = DEFAULT_CTC_MODEL
    enable_diarization: bool = True


# =============================================================================
# Model Cache (Singleton Pattern using dataclass)
# =============================================================================


@dataclass
class _WhisperCache:
    """Cache for Whisper model and processor to avoid reloading."""

    model: Optional[Any] = None
    processor: Optional[Any] = None
    model_name: Optional[str] = None
    device: Optional[torch.device] = None

    def clear(self) -> None:
        """Clear the cache and free memory."""
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        self.model_name = None
        self.device = None


# Single instance of the cache (immutable reference, mutable contents)
_whisper_cache = _WhisperCache()


# =============================================================================
# GPU Memory Management
# =============================================================================


def _clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# =============================================================================
# Whisper Model Management
# =============================================================================


def _get_device() -> torch.device:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _check_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    try:
        # Use dynamic import so static analyzers don't error when the optional
        # package is not installed.
        if importlib.util.find_spec("flash_attn") is None:
            return False
        mod = importlib.import_module("flash_attn")
        return hasattr(mod, "flash_attn_func")
    except Exception:
        return False


def _get_whisper_model(
    model_name: str,
    use_flash_attention: bool,
    debug: bool,
) -> Tuple[Any, Any, torch.device]:
    """Get or create the cached Whisper model and processor.

    Uses the module-level cache to avoid reloading on subsequent calls.

    Returns:
        Tuple of (model, processor, device)
    """
    # Return cached model if name matches
    if _whisper_cache.model is not None and _whisper_cache.model_name == model_name:
        assert _whisper_cache.device is not None
        return _whisper_cache.model, _whisper_cache.processor, _whisper_cache.device

    device = _get_device()

    debug_print(f"Loading Whisper model: {model_name}", debug=debug)
    debug_print(f"Device: {device}", debug=debug)

    # Determine attention implementation
    attn_implementation = (
        "sdpa"  # Default to Scaled Dot-Product Attention (PyTorch 2.0+)
    )
    if use_flash_attention and _check_flash_attention_available():
        attn_implementation = "flash_attention_2"
        debug_print("Using Flash Attention 2", debug=debug)
    else:
        debug_print("Using SDPA (Scaled Dot-Product Attention)", debug=debug)

    # Suppress warnings during model loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gray_debug_output(debug):
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attn_implementation,
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_name)

    # Update cache
    _whisper_cache.model = model
    _whisper_cache.processor = processor
    _whisper_cache.model_name = model_name
    _whisper_cache.device = device

    return model, processor, device


# =============================================================================
# CTC Alignment (Linux only)
# =============================================================================


def _is_ctc_available() -> bool:
    """Check if CTC alignment dependencies are available (Linux only)."""
    if platform.system() == "Windows":
        return False
    try:
        from nemo.collections.asr.models import EncDecCTCModel  # noqa: F401
        from ctc_segmentation import ctc_segmentation  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _get_ctc_model(model_name: str, debug: bool) -> Tuple[Any, List[str], int]:
    """Load and cache the CTC model for forced alignment.

    Returns:
        Tuple of (model, allowed_chars, sample_rate)
    """
    from nemo.collections.asr.models import EncDecCTCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with gray_debug_output(debug):
        ctc_model: Any = EncDecCTCModel.from_pretrained(
            model_name=model_name, map_location=device
        )
        ctc_model = ctc_model.to(device)
        ctc_model.preprocessor.to(device)
        ctc_model.eval()

    # Extract vocabulary from model
    vocab = list(ctc_model.decoder.vocabulary)
    allowed_chars = [ch for ch in vocab if ch not in (" ", "<blank>")]

    # Get sample rate from model config
    sample_rate = int(getattr(ctc_model.cfg.preprocessor, "sample_rate", 16000))

    debug_print(
        f"CTC model loaded: {model_name} (vocab size: {len(vocab)}, sample_rate: {sample_rate})",
        debug=debug,
    )

    return ctc_model, allowed_chars, sample_rate


def _normalize_apostrophes(text: str) -> str:
    """Normalize various apostrophe characters to ASCII."""
    for orig, repl in _APOSTROPHE_VARIANTS.items():
        text = text.replace(orig, repl)
    return text


def _sanitize_word(text: str, allowed_chars: List[str]) -> Optional[str]:
    """Sanitize a word for CTC alignment vocabulary."""
    text = _normalize_apostrophes(text.lower())
    text = text.replace("-", " ")
    text = _SANITIZE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    filtered = "".join(ch for ch in text if ch in allowed_chars)
    return filtered or None


def _prepare_ctc_parameters(model: Any, frame_duration: float) -> Any:
    """Prepare CTC segmentation parameters matching the working ctc.py approach."""
    from ctc_segmentation import CtcSegmentationParameters

    vocabulary: List[str] = list(model.decoder.vocabulary)
    blank_symbol = "<blank>"

    # NeMo vocabulary doesn't contain <blank>, we add it
    char_list = vocabulary + [blank_symbol]

    params = CtcSegmentationParameters(char_list=char_list)
    params.blank = len(char_list) - 1  # Blank is the last character
    params.space = " " if " " in vocabulary else ""  # type: ignore[assignment]
    params.index_duration = frame_duration
    params.frame_duration_ms = frame_duration * 1000.0  # type: ignore[assignment]
    params.subsampling_factor = 1  # type: ignore[assignment]
    params.max_window_size = max(params.max_window_size, 6000)
    params.update_excluded_characters()
    return params


def _run_ctc_alignment(
    audio_path: str,
    segments: List[Dict[str, Any]],
    ctc_model_name: str,
    debug: bool,
) -> List[Dict[str, Any]]:
    """Run CTC forced alignment to refine word timestamps.

    This uses NeMo CTC models with ctc-segmentation library to get
    more accurate word boundaries than Whisper's native timestamps.

    Matches the approach from the working ctc.py module.
    """
    if not _is_ctc_available():
        debug_print("CTC alignment not available (Linux only with NeMo)", debug=debug)
        return segments

    from ctc_segmentation import (
        ctc_segmentation,
        determine_utterance_segments,
        prepare_text,
    )

    try:
        model, allowed_chars, ctc_sample_rate = _get_ctc_model(ctc_model_name, debug)
    except Exception as exc:
        debug_print(f"Failed to load CTC model: {exc}", debug=debug)
        return segments

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load audio at the CTC model's expected sample rate
    try:
        audio_data, file_sample_rate = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample if needed
        if file_sample_rate != ctc_sample_rate:
            import librosa

            audio_data = librosa.resample(
                audio_data, orig_sr=file_sample_rate, target_sr=ctc_sample_rate
            )
        sample_rate = ctc_sample_rate
    except Exception as exc:
        debug_print(f"Failed to load audio for CTC: {exc}", debug=debug)
        return segments

    audio_duration = len(audio_data) / sample_rate

    # Segment margin and backoff strategy (matching ctc.py)
    segment_margin = 0.75
    segment_backoffs = (0.0, 2.0, 5.0, 10.0)

    # Process each segment
    refined_segments = []
    for segment in segments:
        words = segment.get("words", [])
        if not words:
            refined_segments.append(segment)
            continue

        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", audio_duration)

        # Prepare text for alignment
        word_texts = []
        word_indices = []
        for idx, word in enumerate(words):
            text = word.get("word", "").strip()
            sanitized = _sanitize_word(text, allowed_chars)
            if sanitized:
                word_texts.append(sanitized)
                word_indices.append(idx)

        if not word_texts:
            refined_segments.append(segment)
            continue

        # Try alignment with increasing margins (backoff strategy)
        alignment_success = False
        frame_duration = None

        for extra_margin in segment_backoffs:
            margin = segment_margin + extra_margin
            chunk_start = max(0.0, seg_start - margin)
            chunk_end = min(audio_duration, seg_end + margin)

            if chunk_end <= chunk_start:
                chunk_end = min(audio_duration, chunk_start + 2.0)
                if chunk_end <= chunk_start:
                    continue

            start_sample = int(chunk_start * sample_rate)
            end_sample = int(chunk_end * sample_rate)
            chunk_audio = audio_data[start_sample:end_sample]

            if len(chunk_audio) < sample_rate * 0.1:  # Less than 100ms
                continue

            try:
                # Run CTC forward pass
                with torch.inference_mode():
                    audio_tensor = torch.tensor(chunk_audio).unsqueeze(0).to(device)
                    audio_len = torch.tensor([len(chunk_audio)]).to(device)

                    processed_signal, processed_length = model.preprocessor(
                        input_signal=audio_tensor, length=audio_len
                    )
                    processed_signal = processed_signal.to(device)
                    processed_length = processed_length.to(device)

                    log_probs, encoded_length, _ = model(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_length,
                    )
                    log_probs = log_probs[0].cpu().numpy()

                if log_probs is None or len(log_probs) == 0:
                    continue

                # Calculate frame duration
                num_frames = log_probs.shape[0]
                chunk_duration = len(chunk_audio) / sample_rate
                frame_duration = chunk_duration / num_frames if num_frames > 0 else 0.02

                # Prepare CTC parameters (matching ctc.py)
                params = _prepare_ctc_parameters(model, frame_duration)

                # Run CTC segmentation
                ground_truth_mat, utt_begin_indices = prepare_text(params, word_texts)
                timings, char_probs, state_list = ctc_segmentation(
                    params, log_probs, ground_truth_mat
                )
                utt_segments = determine_utterance_segments(
                    params, utt_begin_indices, char_probs, timings, word_texts
                )

                # Update word timestamps
                for i, (start_time, end_time, score) in enumerate(utt_segments):
                    if i >= len(word_indices):
                        break
                    word_idx = word_indices[i]

                    # Convert times relative to chunk start
                    abs_start = chunk_start + start_time
                    abs_end = chunk_start + end_time

                    # Store original timestamps for reference
                    words[word_idx]["original_start"] = words[word_idx].get("start")
                    words[word_idx]["original_end"] = words[word_idx].get("end")

                    # Update with aligned timestamps
                    words[word_idx]["start"] = round(abs_start, 3)
                    words[word_idx]["end"] = round(abs_end, 3)
                    words[word_idx]["confidence_ctc"] = round(float(score), 3)

                alignment_success = True
                break  # Success, no need to try larger margins

            except Exception as exc:
                debug_print(
                    f"CTC alignment attempt failed (margin={margin}): {exc}",
                    debug=debug,
                )
                continue

        if not alignment_success:
            debug_print(
                f"CTC alignment failed for segment {segment.get('id', '?')} after all backoffs",
                debug=debug,
            )

        segment["words"] = words
        if alignment_success:
            segment["alignment"] = {
                "source": "ctc",
                "aligned_words": len(word_texts),
                "total_words": len(words),
            }
        refined_segments.append(segment)

    return refined_segments


# =============================================================================
# Diarization
# =============================================================================


def _run_diarization(
    audio_path: str,
    output_folder: str,
    debug: bool,
) -> List[Dict[str, Any]]:
    """Run speaker diarization using NeMo MSDD.

    Returns list of speaker segments with start, end, and speaker labels.
    """
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    from omegaconf import DictConfig, OmegaConf
    import wget

    print("INFO: Running speaker diarization")

    diarization_dir = os.path.join(output_folder, "diarization_experimental")
    os.makedirs(diarization_dir, exist_ok=True)

    # Create NeMo MSDD configuration
    domain_type = "telephonic"
    config_dir = os.path.join(diarization_dir, "nemo_msdd_configs")
    config_file = f"diar_infer_{domain_type}.yaml"
    config_path = os.path.join(config_dir, config_file)

    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{config_file}"
        debug_print(f"Downloading diarization config from {config_url}", debug=debug)
        with gray_debug_output(debug):
            config_path = wget.download(config_url, config_path)

    with gray_debug_output(debug):
        nemo_config = OmegaConf.load(config_path)

    # Write input manifest
    manifest = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    manifest_path = os.path.join(diarization_dir, "input_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
        f.write("\n")

    # Configure NeMo
    nemo_config.num_workers = 0
    nemo_config.diarizer.manifest_filepath = manifest_path
    nemo_config.diarizer.out_dir = diarization_dir
    nemo_config.diarizer.speaker_embeddings.model_path = "titanet_large"
    nemo_config.diarizer.oracle_vad = False
    nemo_config.diarizer.clustering.parameters.oracle_num_speakers = False
    nemo_config.diarizer.vad.model_path = "vad_multilingual_marblenet"
    nemo_config.diarizer.vad.parameters.onset = 0.8
    nemo_config.diarizer.vad.parameters.offset = 0.6
    nemo_config.diarizer.vad.parameters.pad_offset = -0.05
    nemo_config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"

    with gray_debug_output(debug):
        msdd_model_instance = NeuralDiarizer(cfg=cast(DictConfig, nemo_config))

    audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_file = os.path.join(diarization_dir, "pred_rttms", f"{audio_file_name}.rttm")

    try:
        with gray_debug_output(debug):
            msdd_model_instance.diarize()
    except ValueError as exc:
        if "silence" in str(exc).lower():
            debug_print(
                "Diarization detected only silence; returning empty result.",
                debug=debug,
            )
            return []
        raise
    except RuntimeError as exc:
        if "kernel size can't be greater than actual input size" in str(exc).lower():
            debug_print(
                "MSDD refinement skipped (insufficient context); using clustering output only.",
                debug=debug,
            )
        else:
            raise

    if not os.path.exists(rttm_file):
        debug_print(
            "Diarization output RTTM not found; returning empty result.", debug=debug
        )
        return []

    # Parse RTTM output
    segments = []
    with open(rttm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start_time = float(parts[3])
                duration = float(parts[4])
                segments.append(
                    {
                        "start": start_time,
                        "end": start_time + duration,
                        "speaker": parts[7],
                    }
                )

    # Save diarization JSON
    output_json_path = os.path.join(diarization_dir, "diarization.json")
    with open(output_json_path, "w") as f:
        json.dump(segments, f, indent=4)

    debug_print(f"Diarization found {len(segments)} speaker segments", debug=debug)
    return segments


def _assign_speakers_to_segments(
    segments: List[Dict[str, Any]],
    speaker_segments: List[Dict[str, Any]],
    debug: bool,
) -> List[Dict[str, Any]]:
    """Assign speaker labels to transcription segments based on overlap.

    Uses a simple overlap-based approach to match speakers to segments.
    """
    if not speaker_segments:
        debug_print("No speaker segments available for assignment", debug=debug)
        return segments

    for segment in segments:
        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", 0.0)
        seg_mid = (seg_start + seg_end) / 2

        # Find best matching speaker segment (by overlap or midpoint)
        best_speaker = None
        best_overlap = 0.0

        for spk_seg in speaker_segments:
            spk_start = spk_seg.get("start", 0.0)
            spk_end = spk_seg.get("end", 0.0)

            # Calculate overlap
            overlap_start = max(seg_start, spk_start)
            overlap_end = min(seg_end, spk_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk_seg.get("speaker")

            # Also check if segment midpoint falls within speaker segment
            if spk_start <= seg_mid <= spk_end and best_speaker is None:
                best_speaker = spk_seg.get("speaker")

        if best_speaker:
            segment["speaker"] = best_speaker

        # Also assign speakers to words if we have word-level data
        for word in segment.get("words", []):
            word_start = word.get("start", 0.0)
            word_end = word.get("end", 0.0)
            word_mid = (word_start + word_end) / 2

            for spk_seg in speaker_segments:
                spk_start = spk_seg.get("start", 0.0)
                spk_end = spk_seg.get("end", 0.0)

                if spk_start <= word_mid <= spk_end:
                    word["speaker"] = spk_seg.get("speaker")
                    break

    return segments


# =============================================================================
# Transcription with Native Long-Form Generation
# =============================================================================


def _load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """Load audio file and return waveform at target sample rate.

    Returns:
        Tuple of (audio_array, duration_seconds)
    """
    import librosa  # Lazy import to avoid loading at module level

    try:
        audio_data, file_sr = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono

        # Resample if needed
        if file_sr != target_sr:
            audio_data = librosa.resample(
                audio_data, orig_sr=file_sr, target_sr=target_sr
            )

        duration = len(audio_data) / target_sr
        return audio_data, duration
    except Exception:
        # Fallback using librosa
        audio_data, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        duration = len(audio_data) / target_sr
        return audio_data, duration


def _split_segment_to_words(
    text: str,
    start: float,
    end: float,
) -> List[Dict[str, Any]]:
    """Split a segment's text into individual words with interpolated timestamps.

    Uses proportional distribution based on word character length to estimate
    per-word timestamps within the segment bounds.

    Args:
        text: The segment text to split
        start: Segment start time in seconds
        end: Segment end time in seconds

    Returns:
        List of word dicts with 'word', 'start', 'end' keys
    """
    # Split text into words, preserving punctuation attached to words
    raw_words = text.split()
    if not raw_words:
        return []

    words = []
    duration = end - start

    # Calculate total character length for proportional distribution
    total_chars = sum(len(w) for w in raw_words)
    if total_chars == 0:
        total_chars = len(raw_words)  # Fallback if somehow all empty

    current_time = start
    for word_text in raw_words:
        # Proportional duration based on character count
        word_duration = (
            (len(word_text) / total_chars) * duration
            if total_chars > 0
            else duration / len(raw_words)
        )
        word_end = min(current_time + word_duration, end)

        words.append(
            {
                "word": word_text,
                "start": round(current_time, 3),
                "end": round(word_end, 3),
            }
        )
        current_time = word_end

    return words


def _transcribe_with_native_chunking(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    settings: _TranscribeSettings,
    debug: bool,
) -> List[Dict[str, Any]]:
    """Transcribe audio using Whisper's native long-form transcription.

    Uses model.generate() with return_timestamps=True for sequential
    chunking with condition-on-previous-tokens (Whisper paper Section 3.8).

    CRITICAL for long-form (>30s):
    - truncation=False: Don't truncate to 30s
    - padding="longest": Properly pad the features
    - return_timestamps=True: Triggers sequential long-form algorithm

    This is significantly faster than pipeline() for long audio because:
    1. No manual file chunking overhead
    2. Native sequential processing with token conditioning
    3. More efficient memory usage

    Returns:
        List of word dicts with 'word', 'start', 'end' keys
    """
    device = _whisper_cache.device or _get_device()
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    audio_duration = len(audio_array) / 16000
    debug_print(
        f"Processing {audio_duration:.1f}s audio with native long-form transcription",
        debug=debug,
    )

    # CRITICAL: For long-form transcription (>30s), must use:
    # - truncation=False: prevents cutting to 30 seconds
    # - padding="longest": properly pads the input
    # - return_attention_mask=True: needed for proper batched generation
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        truncation=False,  # CRITICAL: Don't truncate long audio
        padding="longest",  # CRITICAL: Proper padding for long audio
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device, dtype=dtype)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Prepare generation kwargs for long-form transcription
    # return_timestamps=True triggers the sequential algorithm
    generate_kwargs = {
        "task": "transcribe",
        "return_timestamps": True,  # CRITICAL: Triggers long-form sequential algorithm
        "return_segments": True,  # CRITICAL: Return segment dict with full timestamps
        "condition_on_prev_tokens": True,  # Improves accuracy across chunks
        "compression_ratio_threshold": 2.4,  # Filter hallucinated text
        "logprob_threshold": -1.0,  # Filter low-confidence predictions
        "no_speech_threshold": 0.6,  # Filter no-speech segments
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Temperature fallback
    }
    if settings.language:
        generate_kwargs["language"] = settings.language

    # Check if model is English-only
    if settings.model.split(".")[-1] == "en" or "english" in settings.model.lower():
        generate_kwargs.pop("task", None)

    debug_print(
        "Running native long-form transcription with generate()...", debug=debug
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.inference_mode():
            if attention_mask is not None:
                result = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )
            else:
                result = model.generate(
                    input_features,
                    **generate_kwargs,
                )

    # Extract words from the segments dict returned by return_segments=True
    # Result structure: {"sequences": tensor, "segments": [[{start, end, tokens}, ...]]}
    words = []

    if isinstance(result, dict) and "segments" in result:
        # return_segments=True returns a dict with segment info
        segments_list = result.get("segments", [[]])
        # segments_list is a list of lists (one per batch item)
        for batch_segments in segments_list:
            for seg in batch_segments:
                start = seg.get("start", 0.0)
                end = seg.get("end", start + 0.1)
                # Convert start/end from tensors to floats if needed
                if hasattr(start, "item"):
                    start = start.item()
                if hasattr(end, "item"):
                    end = end.item()
                # Decode segment tokens to text
                # tokens can be a tensor or list
                tokens = seg.get("tokens")
                if tokens is None:
                    continue
                # Handle both tensor and list types
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                if len(tokens) == 0:
                    continue
                segment_text = processor.decode(
                    tokens, skip_special_tokens=True
                ).strip()

                if not segment_text:
                    continue

                # Split segment into individual words with interpolated timestamps
                segment_words = _split_segment_to_words(
                    segment_text, float(start), float(end)
                )
                words.extend(segment_words)
    else:
        # Fallback: if not dict, try batch_decode (shouldn't happen with return_segments=True)
        debug_print(
            "Using fallback: batch_decode (return_segments may have failed)",
            debug=debug,
        )
        outputs = (
            result if not isinstance(result, dict) else result.get("sequences", result)
        )
        decoded = processor.batch_decode(
            outputs, skip_special_tokens=True, output_offsets=True
        )

        if decoded and isinstance(decoded[0], dict):
            for item in decoded:
                if "offsets" in item:
                    for offset in item["offsets"]:
                        segment_text = offset.get("text", "").strip()
                        if not segment_text:
                            continue
                        timestamp = offset.get("timestamp", (0.0, 0.0))
                        start = timestamp[0] if timestamp[0] is not None else 0.0
                        end = timestamp[1] if timestamp[1] is not None else start + 0.1
                        segment_words = _split_segment_to_words(
                            segment_text, start, end
                        )
                        words.extend(segment_words)
        elif decoded:
            # No timestamps available - fallback to simple word split
            full_text = decoded[0] if isinstance(decoded[0], str) else str(decoded[0])
            segment_words = _split_segment_to_words(
                full_text.strip(), 0.0, audio_duration
            )
            words.extend(segment_words)

    debug_print(f"Extracted {len(words)} words from transcription", debug=debug)
    return words


def _words_to_segments(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert word list to segment format for output compatibility."""
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current_segment: Dict[str, Any] = {
        "id": 0,
        "start": 0.0,
        "end": 0.0,
        "text": "",
        "words": [],
    }

    max_segment_words = 15
    sentence_endings = ".!?"

    for word in words:
        word_start = word.get("start", 0.0)
        word_end = word.get("end", 0.0)
        word_text = word.get("word", "")

        if not word_text:
            continue

        should_break = False
        if current_segment["words"]:
            last_word = current_segment["words"][-1]
            gap = word_start - last_word["end"]

            if gap > 0.5:
                should_break = True
            elif gap > 0.2 and last_word["word"].rstrip()[-1:] in sentence_endings:
                should_break = True
            elif (
                len(current_segment["words"]) >= max_segment_words
                and last_word["word"].rstrip()[-1:] in sentence_endings
            ):
                should_break = True

        if should_break:
            current_segment["end"] = current_segment["words"][-1]["end"]
            current_segment["text"] = " ".join(
                w["word"] for w in current_segment["words"]
            )
            segments.append(current_segment)

            current_segment = {
                "id": len(segments),
                "start": word_start,
                "end": word_end,
                "text": "",
                "words": [],
            }
        elif not current_segment["words"]:
            current_segment["start"] = word_start

        current_segment["words"].append(word.copy())
        current_segment["end"] = word_end

    if current_segment["words"]:
        current_segment["text"] = " ".join(w["word"] for w in current_segment["words"])
        segments.append(current_segment)

    return segments


def _transcribe_audio(
    audio_path: str,
    output_folder: str,
    settings: _TranscribeSettings,
    debug: bool,
) -> Tuple[Dict[str, Any], str]:
    """Run transcription using Whisper's native long-form generation.

    Uses model.generate() directly instead of pipeline() for efficient
    sequential transcription with condition-on-previous-tokens.
    """

    # Load audio
    debug_print(f"Loading audio from {audio_path}", debug=debug)
    audio_array, audio_duration = _load_audio(audio_path, target_sr=16000)
    debug_print(f"Audio duration: {audio_duration:.1f}s", debug=debug)

    # Load model and processor
    model, processor, device = _get_whisper_model(
        settings.model,
        settings.use_flash_attention,
        debug,
    )

    start_time = time.perf_counter()

    # Run transcription with native chunking
    try:
        words = _transcribe_with_native_chunking(
            model,
            processor,
            audio_array,
            settings,
            debug,
        )
    except torch.cuda.OutOfMemoryError:
        debug_print(
            "OOM during transcription, clearing cache and retrying...", debug=debug
        )

        # Clear cache and GPU memory
        del model
        del processor
        _whisper_cache.clear()
        _clear_gpu_cache()
        time.sleep(1)

        # Reload and retry
        model, processor, device = _get_whisper_model(
            settings.model,
            settings.use_flash_attention,
            debug,
        )

        words = _transcribe_with_native_chunking(
            model,
            processor,
            audio_array,
            settings,
            debug,
        )

    elapsed = time.perf_counter() - start_time
    debug_print(f"Transcription completed in {elapsed:.2f} seconds", debug=debug)

    # Convert words to segments
    segments = _words_to_segments(words)

    # Build full text
    full_text = " ".join(w["word"] for w in words)

    result = {
        "transcription": full_text,
        "segments": segments,
        "model": {
            "name": settings.model,
            "device": str(device),
            "chunk_length_s": settings.chunk_length_s,
            "use_flash_attention": settings.use_flash_attention,
        },
        "performance": {
            "transcription_seconds": round(elapsed, 3),
            "audio_duration_seconds": round(audio_duration, 3),
            "realtime_factor": round(audio_duration / elapsed, 2) if elapsed > 0 else 0,
        },
    }

    return result, full_text


# =============================================================================
# Output Generation
# =============================================================================


def _generate_srt(segments: List[Dict[str, Any]]) -> str:
    """Generate SRT subtitle content from segments."""
    lines = []
    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        text = segment.get("text", "")

        # Format timestamps as HH:MM:SS,mmm
        def fmt_time(t: float) -> str:
            hours = int(t // 3600)
            minutes = int((t % 3600) // 60)
            seconds = int(t % 60)
            millis = int((t - int(t)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        lines.append(str(i))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def _save_outputs(
    output_folder: str,
    result: Dict[str, Any],
    full_text: str,
    debug: bool,
) -> None:
    """Save transcription outputs to files."""
    transcription_folder = os.path.join(output_folder, "transcription_experimental")
    os.makedirs(transcription_folder, exist_ok=True)

    # Save JSON
    json_path = os.path.join(transcription_folder, "transcription.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    debug_print(f"Saved transcription JSON to {json_path}", debug=debug)

    # Save plain text
    txt_path = os.path.join(transcription_folder, "transcription.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    debug_print(f"Saved transcription text to {txt_path}", debug=debug)

    # Save SRT
    srt_content = _generate_srt(result.get("segments", []))
    srt_path = os.path.join(transcription_folder, "transcription.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    debug_print(f"Saved SRT subtitles to {srt_path}", debug=debug)


# =============================================================================
# Main Entry Point
# =============================================================================


def handle(
    input_file: str,
    output_folder: str,
    config: Optional["TranscribeExperimentalConfig"] = None,
    *,
    debug: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """Main entry point for experimental transcription.

    Uses HuggingFace Transformers with model.generate() for native long-form
    transcription with Flash Attention 2 / SDPA optimization, plus optional
    CTC alignment for accurate word-level timestamps.

    Args:
        input_file: Path to the audio file.
        output_folder: Path to output directory.
        config: TranscribeExperimentalConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Tuple of (transcription_data dict, full_transcription string).
    """
    print(
        "INFO: Starting experimental transcription (native long-form mode with diarization)"
    )

    # Extract config values with inline defaults (per REFACTORING_PRINCIPLES.md)
    model = config.model if config and config.model else DEFAULT_MODEL
    chunk_length_s = (
        config.chunk_length_s
        if config and config.chunk_length_s
        else DEFAULT_CHUNK_LENGTH_S
    )
    use_flash_attention = (
        config.use_flash_attention
        if config and config.use_flash_attention is not None
        else DEFAULT_USE_FLASH_ATTENTION
    )
    language = config.language if config else None
    enable_ctc_refinement = (
        config.enable_ctc_refinement
        if config and config.enable_ctc_refinement is not None
        else DEFAULT_ENABLE_CTC_REFINEMENT
    )
    ctc_model = config.ctc_model if config and config.ctc_model else DEFAULT_CTC_MODEL
    enable_diarization = (
        config.enable_diarization
        if config and config.enable_diarization is not None
        else True
    )

    # Bundle into settings for internal functions
    settings = _TranscribeSettings(
        model=model,
        chunk_length_s=chunk_length_s,
        use_flash_attention=use_flash_attention,
        language=language,
        enable_ctc_refinement=enable_ctc_refinement,
        ctc_model=ctc_model,
        enable_diarization=enable_diarization,
    )

    debug_print(f"Model: {settings.model}", debug=debug)
    debug_print(f"Flash Attention: {settings.use_flash_attention}", debug=debug)
    debug_print(f"CTC refinement: {settings.enable_ctc_refinement}", debug=debug)
    debug_print(f"Diarization: {settings.enable_diarization}", debug=debug)

    # Run transcription with native long-form generation
    result, full_text = _transcribe_audio(input_file, output_folder, settings, debug)

    # Clear GPU cache before CTC/diarization
    _clear_gpu_cache()

    # Optional CTC refinement
    if settings.enable_ctc_refinement and _is_ctc_available():
        print("INFO: Running CTC forced alignment for word timestamp refinement")
        _clear_gpu_cache()  # Clear before CTC
        start_time = time.perf_counter()

        result["segments"] = _run_ctc_alignment(
            input_file,
            result["segments"],
            settings.ctc_model,
            debug,
        )

        ctc_elapsed = time.perf_counter() - start_time
        result["performance"]["ctc_alignment_seconds"] = round(ctc_elapsed, 3)
        debug_print(
            f"CTC alignment completed in {ctc_elapsed:.2f} seconds", debug=debug
        )
    elif settings.enable_ctc_refinement:
        debug_print(
            "CTC refinement requested but not available (Linux only)", debug=debug
        )

    # Optional diarization
    if settings.enable_diarization:
        _clear_gpu_cache()  # Clear before diarization
        start_time = time.perf_counter()

        speaker_segments = _run_diarization(input_file, output_folder, debug)

        diar_elapsed = time.perf_counter() - start_time
        result["performance"]["diarization_seconds"] = round(diar_elapsed, 3)
        debug_print(f"Diarization completed in {diar_elapsed:.2f} seconds", debug=debug)

        # Assign speakers to segments and words
        if speaker_segments:
            result["segments"] = _assign_speakers_to_segments(
                result["segments"],
                speaker_segments,
                debug,
            )
            result["diarization"] = {
                "speaker_count": len(set(s.get("speaker") for s in speaker_segments)),
                "speaker_segments": len(speaker_segments),
            }

    # Save outputs
    _save_outputs(output_folder, result, full_text, debug)

    total_time = result["performance"].get("transcription_seconds", 0.0)
    total_time += result["performance"].get("ctc_alignment_seconds", 0.0)
    total_time += result["performance"].get("diarization_seconds", 0.0)
    print(f"INFO: Experimental transcription completed in {total_time:.2f} seconds")

    return result, full_text
