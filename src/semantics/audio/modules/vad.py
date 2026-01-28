"""Voice Activity Detection module.

Detects speech segments in audio using the Silero VAD model.
Outputs timestamped speech segments for downstream processing.
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torchaudio

from .utils.chunks import cleanup_chunks, compute_chunk_offsets, split_audio
from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import VadConfig


@lru_cache(maxsize=1)
def _load_vad_model():
    """Load and cache the Silero VAD model."""
    result = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    # torch.hub.load returns (model, utils) tuple for silero-vad
    vad_model = result[0]  # type: ignore[index]
    utils = result[1]  # type: ignore[index]
    return vad_model, utils


def handle(
    input_file: str,
    output_folder: str,
    config: "VadConfig | None" = None,
    *,
    debug: bool = False,
) -> dict:
    """Perform Voice Activity Detection on an audio file.

    Args:
        input_file: Path to the input audio file.
        output_folder: Path to the output folder for results.
        config: VadConfig with threshold and duration settings.
        debug: Enable verbose logging.

    Returns:
        Dictionary containing detected speech segments with timestamps.
    """
    # Extract config values with defaults
    chunk_length = config.chunk_length if config else 900
    threshold = config.threshold if config else 0.5
    min_speech_duration_ms = config.min_speech_duration_ms if config else 250
    min_silence_duration_ms = config.min_silence_duration_ms if config else 100

    print("INFO: Performing Voice Activity Detection (VAD)")

    debug_print("Loading Silero VAD model", debug=debug)
    with gray_debug_output(debug):
        vad_model, utils = _load_vad_model()

    get_speech_timestamps = utils[0]

    with gray_debug_output(debug):
        chunks, chunk_dir = split_audio(input_file, output_folder, "vad", chunk_length)
    debug_print(f"Split audio into {len(chunks)} chunk(s) for VAD", debug=debug)
    with gray_debug_output(debug):
        offsets = compute_chunk_offsets(chunks)

    segments_dir = os.path.join(output_folder, "vad", "segments")
    if os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)
    os.makedirs(segments_dir, exist_ok=True)

    json_data = {"segments": []}
    segment_index = 0

    try:
        for index, (chunk_path, offset_seconds) in enumerate(
            zip(chunks, offsets), start=1
        ):
            debug_print(
                f"Running VAD on chunk {index}/{len(chunks)}: {chunk_path}", debug=debug
            )
            with gray_debug_output(debug):
                wav, sr = torchaudio.load(chunk_path)
                with torch.inference_mode():
                    speech_timestamps = get_speech_timestamps(
                        wav,
                        vad_model,
                        sampling_rate=sr,
                        threshold=threshold,
                        min_speech_duration_ms=min_speech_duration_ms,
                        min_silence_duration_ms=min_silence_duration_ms,
                    )

            for timestamp in speech_timestamps:
                start = timestamp["start"]
                end = timestamp["end"]
                if end <= start:
                    continue

                segment_wav = wav[:, start:end]
                segment_index += 1
                segment_file = os.path.join(segments_dir, f"{segment_index}.wav")
                with gray_debug_output(debug):
                    torchaudio.save(segment_file, segment_wav, sr)

                json_data["segments"].append(
                    {
                        "id": segment_index,
                        "start": start / sr + offset_seconds,
                        "end": end / sr + offset_seconds,
                    }
                )

            del wav
    finally:
        cleanup_chunks(chunk_dir)

    json_file = os.path.join(output_folder, "vad", "timestamps.json")
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    return json_data
