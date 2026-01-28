"""Speaker diarization module using NeMo MSDD.

This module provides functionality for speaker diarization using NVIDIA NeMo's
multi-scale diarization decoder (MSDD). It identifies and segments speakers
in audio files.
"""

from __future__ import annotations

import json
import os
import wave
from typing import TYPE_CHECKING, List

import wget
from fastdtw import fastdtw  # type: ignore[import-untyped]
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf

from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from ..config import DiarizeConfig


def handle(
    input_file: str,
    output_folder: str,
    config: "DiarizeConfig | None" = None,
    *,
    debug: bool = False,
) -> list:
    """Perform speaker diarization using NeMo MSDD.

    Args:
        input_file: Path to the input audio file.
        output_folder: Directory where output files will be written.
        config: DiarizeConfig instance with diarization parameters, or None for defaults.
        debug: If True, emit verbose debug output.

    Returns:
        List of speaker segments with start, end, and speaker labels.
    """
    # Extract config values (use defaults if config is None)
    domain_type = config.domain_type if config else "telephonic"
    pretrained_vad = config.pretrained_vad if config else "vad_multilingual_marblenet"
    pretrained_speaker_model = (
        config.pretrained_speaker_model if config else "titanet_large"
    )
    msdd_model = config.msdd_model if config else "diar_msdd_telephonic"
    onset = config.onset if config else 0.8
    offset = config.offset if config else 0.6
    pad_offset = config.pad_offset if config else -0.05
    long_audio_threshold_secs = config.long_audio_threshold_secs if config else 1800.0
    long_audio_window_length_in_sec = (
        list(config.long_audio_window_length_in_sec)
        if config and config.long_audio_window_length_in_sec
        else [1.5, 1.0]
    )
    long_audio_shift_length_in_sec = (
        list(config.long_audio_shift_length_in_sec)
        if config and config.long_audio_shift_length_in_sec
        else [0.75, 0.5]
    )
    long_audio_multiscale_weights = (
        list(config.long_audio_multiscale_weights)
        if config and config.long_audio_multiscale_weights
        else [1.0, 1.0]
    )
    long_audio_embeddings_per_chunk_max = (
        config.long_audio_embeddings_per_chunk_max if config else 600
    )
    long_audio_chunk_cluster_count_max = (
        config.long_audio_chunk_cluster_count_max if config else 24
    )
    long_audio_max_num_speakers_max = (
        config.long_audio_max_num_speakers_max if config else 6
    )
    long_audio_sparse_search_volume_max = (
        config.long_audio_sparse_search_volume_max if config else 10
    )
    long_audio_max_rp_threshold_max = (
        config.long_audio_max_rp_threshold_max if config else 0.2
    )

    print("INFO: Performing speaker diarization")

    diarization_dir = os.path.join(output_folder, "diarization")
    os.makedirs(diarization_dir, exist_ok=True)
    output_json_path = os.path.join(diarization_dir, "diarization.json")

    # Create NeMo MSDD configuration
    nemo_config = _create_msdd_config(
        audio_file=input_file,
        output_dir=diarization_dir,
        domain_type=domain_type,
        pretrained_vad=pretrained_vad,
        pretrained_speaker_model=pretrained_speaker_model,
        msdd_model=msdd_model,
        onset=onset,
        offset=offset,
        pad_offset=pad_offset,
        long_audio_threshold_secs=long_audio_threshold_secs,
        long_audio_window_length_in_sec=long_audio_window_length_in_sec,
        long_audio_shift_length_in_sec=long_audio_shift_length_in_sec,
        long_audio_multiscale_weights=long_audio_multiscale_weights,
        long_audio_embeddings_per_chunk_max=long_audio_embeddings_per_chunk_max,
        long_audio_chunk_cluster_count_max=long_audio_chunk_cluster_count_max,
        long_audio_max_num_speakers_max=long_audio_max_num_speakers_max,
        long_audio_sparse_search_volume_max=long_audio_sparse_search_volume_max,
        long_audio_max_rp_threshold_max=long_audio_max_rp_threshold_max,
        debug=debug,
    )

    with gray_debug_output(debug):
        # OmegaConf.create returns DictConfig when given a dict
        msdd_model_instance = NeuralDiarizer(cfg=nemo_config)  # type: ignore[arg-type]

    audio_file_name = os.path.splitext(os.path.basename(input_file))[0]
    rttm_file = os.path.join(diarization_dir, "pred_rttms", f"{audio_file_name}.rttm")

    try:
        with gray_debug_output(debug):
            msdd_model_instance.diarize()
    except ValueError as exc:
        if "silence" in str(exc).lower():
            print("WARN: Diarization detected only silence; returning empty result.")
            with open(output_json_path, "w") as f:
                json.dump([], f, indent=4)
            return []
        raise
    except RuntimeError as exc:
        if "kernel size can't be greater than actual input size" in str(exc).lower():
            if debug:
                debug_print(
                    "WARN: MSDD refinement skipped (insufficient context); using clustering output only.",
                    debug=True,
                )
            else:
                print(
                    "WARN: MSDD refinement skipped (insufficient context); using clustering output only."
                )
        else:
            raise

    if not os.path.exists(rttm_file):
        print("WARN: Diarization output RTTM not found; returning empty result.")
        with open(output_json_path, "w") as f:
            json.dump([], f, indent=4)
        return []

    # Parse RTTM output
    segments = _parse_rttm(rttm_file)

    with open(output_json_path, "w") as f:
        json.dump(segments, f, indent=4)

    return segments


def _parse_rttm(rttm_file: str) -> List[dict]:
    """Parse RTTM file into speaker segments."""
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
    return segments


def match_speakers(transcript_segments: list, speaker_segments: list) -> list:
    """Match speaker information with transcript segments using DTW."""
    transcript_times = [seg["start"] for seg in transcript_segments]
    speaker_times = [seg["start"] for seg in speaker_segments]

    _, path = fastdtw(transcript_times, speaker_times, dist=lambda u, v: abs(u - v))

    for idx_transcript, idx_speaker in path:
        if idx_transcript < len(transcript_segments) and idx_speaker < len(
            speaker_segments
        ):
            transcript_segments[idx_transcript]["speaker"] = speaker_segments[
                idx_speaker
            ]["speaker"]

    return transcript_segments


def _create_msdd_config(
    audio_file: str,
    output_dir: str,
    *,
    domain_type: str,
    pretrained_vad: str,
    pretrained_speaker_model: str,
    msdd_model: str,
    onset: float,
    offset: float,
    pad_offset: float,
    long_audio_threshold_secs: float,
    long_audio_window_length_in_sec: list,
    long_audio_shift_length_in_sec: list,
    long_audio_multiscale_weights: list,
    long_audio_embeddings_per_chunk_max: int,
    long_audio_chunk_cluster_count_max: int,
    long_audio_max_num_speakers_max: int,
    long_audio_sparse_search_volume_max: int,
    long_audio_max_rp_threshold_max: float,
    debug: bool,
):
    """Create NeMo MSDD configuration."""
    config_dir = "nemo_msdd_configs"
    config_file = f"diar_infer_{domain_type}.yaml"
    config_path = os.path.join(config_dir, config_file)

    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{config_file}"
        if debug:
            debug_print(
                f"INFO: Downloading diarization config from {config_url}", debug=True
            )
        else:
            print(f"INFO: Downloading diarization config from {config_url}")
        with gray_debug_output(debug):
            config_path = wget.download(config_url, config_path)

    with gray_debug_output(debug):
        nemo_config = OmegaConf.load(config_path)

    # Write input manifest
    manifest = {
        "audio_filepath": audio_file,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    manifest_path = os.path.join(output_dir, "input_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
        f.write("\n")

    # Configure NeMo
    nemo_config.num_workers = 0
    nemo_config.diarizer.manifest_filepath = manifest_path
    nemo_config.diarizer.out_dir = output_dir
    nemo_config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    nemo_config.diarizer.oracle_vad = False
    nemo_config.diarizer.clustering.parameters.oracle_num_speakers = False
    nemo_config.diarizer.vad.model_path = pretrained_vad
    nemo_config.diarizer.vad.parameters.onset = onset
    nemo_config.diarizer.vad.parameters.offset = offset
    nemo_config.diarizer.vad.parameters.pad_offset = pad_offset
    nemo_config.diarizer.msdd_model.model_path = msdd_model

    # Apply long-audio optimizations if needed
    long_audio_threshold = float(
        os.getenv("PLATFORM_DIAR_LONG_AUDIO_SECS", str(long_audio_threshold_secs))
    )
    duration_seconds = _get_audio_duration(audio_file)

    if duration_seconds and duration_seconds >= long_audio_threshold:
        hours = duration_seconds / 3600.0
        if debug:
            debug_print(
                f"INFO: Applying long-form diarization overrides (~{hours:.2f}h audio)",
                debug=True,
            )
        else:
            print(
                f"INFO: Applying long-form diarization overrides (~{hours:.2f}h audio)"
            )

        speaker_params = nemo_config.diarizer.speaker_embeddings.parameters
        speaker_params.window_length_in_sec = long_audio_window_length_in_sec
        speaker_params.shift_length_in_sec = long_audio_shift_length_in_sec
        speaker_params.multiscale_weights = long_audio_multiscale_weights

        clustering_params = nemo_config.diarizer.clustering.parameters
        clustering_params.embeddings_per_chunk = min(
            getattr(clustering_params, "embeddings_per_chunk", 10000),
            long_audio_embeddings_per_chunk_max,
        )
        clustering_params.chunk_cluster_count = min(
            getattr(clustering_params, "chunk_cluster_count", 50),
            long_audio_chunk_cluster_count_max,
        )
        clustering_params.max_num_speakers = min(
            getattr(clustering_params, "max_num_speakers", 8),
            long_audio_max_num_speakers_max,
        )
        clustering_params.sparse_search_volume = min(
            getattr(clustering_params, "sparse_search_volume", 30),
            long_audio_sparse_search_volume_max,
        )
        clustering_params.max_rp_threshold = min(
            getattr(clustering_params, "max_rp_threshold", 0.25),
            long_audio_max_rp_threshold_max,
        )

    return nemo_config


def _get_audio_duration(path: str) -> float | None:
    """Get audio duration in seconds."""
    try:
        with wave.open(path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate == 0:
                return None
            return wav_file.getnframes() / float(frame_rate)
    except (OSError, wave.Error):
        return None
