import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Union
from collections import defaultdict
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
    ASTFeatureExtractor,
)
from transformers.utils import logging as transformers_logging
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from .utils.logging import debug_print, gray_debug_output, info_print
from .vad import _load_vad_model as _shared_load_vad_model

if TYPE_CHECKING:
    from ..config import TimelineConfig

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

__all__ = ["handle"]

# Module-level model caches to avoid reloading across calls
_EMOTION_MODEL_CACHE: Optional[Tuple] = None  # (model, processor, labels, device)
_AST_MODEL_CACHE: Optional[Tuple] = None  # (model, processor, device)


class _AudioClassifier:
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 8,
        *,
        window_size: float = 2.0,
        hop_size: float = 1.0,
        target_sample_rate: int = 16000,
        emotion_threshold: float = 0.12,
        audio_event_threshold: float = 0.35,
        min_segment_duration: float = 0.5,
        emotion_temperature: float = 0.7,
        emotion_prob_smoothing: int = 5,
        emotion_confidence_gamma: float = 1.35,
        min_speech_overlap_ratio: float = 0.15,
        vad_threshold: float = 0.5,
        vad_min_speech_duration_ms: int = 250,
        vad_min_silence_duration_ms: int = 100,
        debug: bool = False,
    ):
        """Initialize the audio classifier with multiple models.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Number of windows to process at once (adjust based on GPU memory)
        """
        self.debug = debug
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._debug(f"DEBUG: Using device {self.device}")
        self._debug(f"DEBUG: Batch size {batch_size}")

        # Load models
        self._debug("DEBUG: Loading classification models")
        self._load_emotion_model()
        self._load_audio_tagging_model()
        self._load_vad_model()

        # Analysis parameters
        self.window_size = float(window_size)
        self.hop_size = float(hop_size)
        self.target_sr = int(target_sample_rate)

        # Confidence thresholds (tuned for better accuracy)
        self.emotion_threshold = float(emotion_threshold)
        self.audio_event_threshold = float(audio_event_threshold)
        self.min_segment_duration = float(min_segment_duration)

        # Emotion calibration controls
        self.emotion_temperature = float(emotion_temperature)
        self.emotion_prob_smoothing = int(emotion_prob_smoothing)
        self.emotion_confidence_gamma = float(emotion_confidence_gamma)
        self.min_speech_overlap_ratio = float(min_speech_overlap_ratio)

        # VAD controls
        self.vad_threshold = float(vad_threshold)
        self.vad_min_speech_duration_ms = int(vad_min_speech_duration_ms)
        self.vad_min_silence_duration_ms = int(vad_min_silence_duration_ms)

        # Audio tagging categories and heuristics
        self.audio_category_keywords = {
            "speech": [
                "Speech",
                "Male speech",
                "Female speech",
                "Child speech",
                "Conversation",
                "Narration",
                "Monologue",
            ],
            "music": [
                "Music",
                "Musical instrument",
                "Piano",
                "Guitar",
                "Drum",
                "Singing",
                "Electronic music",
                "Rock music",
                "Pop music",
                "Classical music",
            ],
            "laughter": ["Laughter", "Giggle", "Chuckle", "Belly laugh"],
            "applause": ["Applause", "Clapping"],
            "crying": ["Crying", "Sobbing", "Baby crying", "Whimper"],
            "coughing": ["Cough", "Sneeze", "Throat clearing"],
            "breathing": ["Breathing", "Pant", "Gasp"],
            "silence": ["Silence", "Quiet"],
            "animal": ["Dog", "Cat", "Bird", "Animal"],
            "vehicle": ["Car", "Traffic", "Engine", "Vehicle"],
            "nature": ["Wind", "Rain", "Water", "Thunder", "Ocean"],
            "crowd": ["Crowd", "Cheering", "Chatter", "Hubbub"],
        }
        self.speech_categories = {"speech"}

        # Pre-compute AST label → category mapping for fast event detection
        self._precompute_category_indices()

    def _precompute_category_indices(self) -> None:
        """Build a lookup from each audio category to its matching AST label indices.

        This is done once at init and eliminates millions of repeated string
        comparisons during ``_detect_audio_events``.  For each category we
        store a tuple ``(np_indices, label_names)`` where ``np_indices`` is a
        ``numpy`` int array suitable for fancy-indexing into the probability
        vector.
        """
        self._category_np_indices: Dict[str, Tuple[np.ndarray, List[str]]] = {}

        if getattr(self, "audio_model", None) is None:
            return

        id2label = self.audio_model.config.id2label  # {int: str, ...}

        for category, keywords in self.audio_category_keywords.items():
            keywords_lower = [kw.lower() for kw in keywords]
            matching_indices: List[int] = []
            matching_labels: List[str] = []

            for idx, label in id2label.items():
                label_lower = label.lower()
                if any(kw in label_lower for kw in keywords_lower):
                    matching_indices.append(int(idx))
                    matching_labels.append(label)

            if matching_indices:
                self._category_np_indices[category] = (
                    np.array(matching_indices, dtype=np.intp),
                    matching_labels,
                )

        self._debug(
            f"DEBUG: Pre-computed category indices for {len(self._category_np_indices)} categories"
        )

    def _debug(self, message: str) -> None:
        debug_print(message, debug=self.debug)

    def _gray_context(self):
        return gray_debug_output(self.debug)

    def _load_emotion_model(self):
        """Load emotion recognition model (cached across instances)."""
        global _EMOTION_MODEL_CACHE
        if _EMOTION_MODEL_CACHE is not None:
            self.emotion_model, self.emotion_processor, self.emotion_labels, _ = _EMOTION_MODEL_CACHE
            self.emotion_model.to(torch.device(self.device))
            self._debug("DEBUG: Emotion model loaded (from cache)")
            return

        try:
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            with self._gray_context():
                self.emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name
                )
                self.emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    model_name
                )
                self.emotion_model.to(torch.device(self.device))  # type: ignore[arg-type]
            self.emotion_model.eval()
            if self.device == "cuda":
                self.emotion_model.half()  # float16 for faster Tensor Core inference

            config_labels = getattr(self.emotion_model.config, "id2label", None)
            if isinstance(config_labels, dict) and config_labels:
                try:
                    sorted_labels = [
                        config_labels[str(i)]
                        if str(i) in config_labels
                        else config_labels[i]
                        for i in range(len(config_labels))
                    ]
                    self.emotion_labels = [
                        str(label).lower() for label in sorted_labels
                    ]
                except Exception:
                    self.emotion_labels = [
                        str(label).lower() for label in config_labels.values()
                    ]
            else:
                self.emotion_labels = [
                    "angry",
                    "calm",
                    "disgust",
                    "fearful",
                    "happy",
                    "neutral",
                    "sad",
                    "surprised",
                ]
            _EMOTION_MODEL_CACHE = (self.emotion_model, self.emotion_processor, self.emotion_labels, self.device)
            self._debug("DEBUG: Emotion model loaded")
        except Exception as e:
            print(f"WARN: Could not load emotion model: {e}")
            self.emotion_model = None

    def _load_audio_tagging_model(self):
        """Load general audio classification model (cached across instances)."""
        global _AST_MODEL_CACHE
        if _AST_MODEL_CACHE is not None:
            self.audio_model, self.audio_processor, _ = _AST_MODEL_CACHE
            self.audio_model.to(self.device)
            self._debug("DEBUG: Audio tagging model loaded (from cache)")
            return

        try:
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            with self._gray_context():
                self.audio_model = AutoModelForAudioClassification.from_pretrained(
                    model_name
                )
                self.audio_processor = ASTFeatureExtractor.from_pretrained(model_name)
                self.audio_model.to(self.device)
            self.audio_model.eval()
            if self.device == "cuda":
                self.audio_model.half()  # float16 for faster Tensor Core inference
            _AST_MODEL_CACHE = (self.audio_model, self.audio_processor, self.device)
            self._debug("DEBUG: Audio tagging model loaded")
        except Exception as e:
            print(f"WARN: Could not load audio tagging model: {e}")
            self.audio_model = None

    def _load_vad_model(self):
        """Load Voice Activity Detection model (reuses cached model from vad module)."""
        try:
            with self._gray_context():
                vad_model, vad_utils = _shared_load_vad_model()
                self.vad_model = vad_model
                self.vad_model.to(self.device)
            self.get_speech_timestamps = vad_utils[0]
            self._debug("DEBUG: VAD model loaded (shared cache)")
        except Exception as e:
            print(f"WARN: Could not load VAD model: {e}")
            self.vad_model = None

    def load_audio(
        self, audio_path: str, chunk_duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file
            chunk_duration: If provided, load audio in chunks (for very long files)
        """
        self._debug(f"DEBUG: Loading audio file: {audio_path}")

        # Get audio info first
        with self._gray_context():
            info = torchaudio.info(audio_path)
        total_duration = info.num_frames / info.sample_rate
        self._debug(
            f"DEBUG: Audio duration {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)"
        )

        # For very long files, process in chunks to save memory
        if chunk_duration and total_duration > chunk_duration:
            self._debug(
                f"DEBUG: Audio will be processed in {chunk_duration} second chunk(s)"
            )
            return np.array([]), int(info.sample_rate)  # Signal chunked processing

        # Load complete audio
        with self._gray_context():
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        self._debug(f"DEBUG: Resampled audio at {sr} Hz")

        return audio, int(sr)

    def _detect_voice_activity(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect speech segments using VAD."""
        if self.vad_model is None:
            return []

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)

            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=sr,
                threshold=self.vad_threshold,
                min_speech_duration_ms=self.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.vad_min_silence_duration_ms,
            )

            # Convert to seconds
            vad_segments = []
            for ts in speech_timestamps:
                vad_segments.append(
                    {
                        "start": float(ts["start"] / sr),
                        "end": float(ts["end"] / sr),
                        "type": "speech",
                    }
                )

            return vad_segments

        except Exception as e:
            print(f"WARN: VAD failed: {e}")
            return []

    def _window_has_speech(
        self,
        start: float,
        end: float,
        speech_segments: List[Dict],
        min_overlap_ratio: Optional[float] = None,
    ) -> bool:
        """Check if the window overlaps speech segments by a minimum ratio."""
        if not speech_segments:
            return False

        window_duration = max(1e-6, end - start)
        overlap_ratio = (
            self.min_speech_overlap_ratio
            if min_overlap_ratio is None
            else min_overlap_ratio
        )

        for seg in speech_segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            overlap_start = max(start, seg_start)
            overlap_end = min(end, seg_end)
            overlap = overlap_end - overlap_start
            if overlap <= 0:
                continue

            seg_duration = max(1e-6, seg_end - seg_start)
            required_overlap = min(window_duration, seg_duration) * overlap_ratio
            if overlap >= required_overlap:
                return True

        return False

    def _should_analyze_emotion(
        self, start: float, end: float, events: List[Dict], speech_segments: List[Dict]
    ) -> bool:
        """Determine if emotion analysis should run for the window."""
        has_speech_event = any(
            evt.get("category") in self.speech_categories for evt in events
        )
        has_vad_overlap = self._window_has_speech(start, end, speech_segments)

        if has_vad_overlap or has_speech_event:
            return True

        return False

    def _extract_windows(
        self, audio: np.ndarray, sr: int
    ) -> List[Tuple[float, float, np.ndarray]]:
        """Extract overlapping windows from audio."""
        window_samples = int(self.window_size * sr)
        hop_samples = int(self.hop_size * sr)

        windows = []
        for start_sample in range(0, len(audio), hop_samples):
            end_sample = min(start_sample + window_samples, len(audio))

            if (
                end_sample - start_sample < window_samples * 0.5
            ):  # Skip very short segments
                break

            window = audio[start_sample:end_sample]

            # Pad if necessary
            if len(window) < window_samples:
                window = np.pad(
                    window, (0, window_samples - len(window)), mode="constant"
                )

            start_time = start_sample / sr
            end_time = end_sample / sr

            windows.append((start_time, end_time, window))

        return windows

    def _detect_emotion(self, audio_windows: List[np.ndarray]) -> List[Optional[Dict]]:
        """Detect emotion in audio windows (batch processing)."""
        if self.emotion_model is None:
            return [None] * len(audio_windows)

        try:
            # Process in batches
            probs_history = []

            for i in range(0, len(audio_windows), self.batch_size):
                batch = audio_windows[i : i + self.batch_size]

                # Process batch
                inputs = self.emotion_processor(
                    list(batch),
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                    padding=True,
                )
                _dtype = next(self.emotion_model.parameters()).dtype
                inputs = {k: v.to(device=self.device, dtype=_dtype) for k, v in inputs.items()}

                # Get predictions
                with torch.no_grad():
                    logits = self.emotion_model(**inputs).logits.float()  # type: ignore[union-attr]
                    if self.emotion_temperature and self.emotion_temperature != 1.0:
                        logits = logits / self.emotion_temperature
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                probs = probs.cpu().numpy()
                probs_history.extend(probs)

            if not probs_history:
                return [None] * len(audio_windows)

            prob_matrix = np.array(probs_history)

            if (
                self.emotion_prob_smoothing
                and self.emotion_prob_smoothing > 1
                and len(prob_matrix) > 1
            ):
                window = max(1, int(self.emotion_prob_smoothing))
                if window % 2 == 0:
                    window += 1  # ensure odd window for symmetric padding
                pad = window // 2
                padded = np.pad(prob_matrix, ((pad, pad), (0, 0)), mode="edge")
                smoothed_matrix = []
                for idx in range(len(prob_matrix)):
                    window_slice = padded[idx : idx + window]
                    smoothed_matrix.append(window_slice.mean(axis=0))
                prob_matrix = np.array(smoothed_matrix)

            if self.emotion_confidence_gamma and self.emotion_confidence_gamma != 1.0:
                prob_matrix = np.power(
                    np.clip(prob_matrix, 1e-8, 1.0), self.emotion_confidence_gamma
                )
                prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

            results = []
            for prob in prob_matrix:
                top_idx = int(np.argmax(prob))
                confidence = float(prob[top_idx])
                results.append(
                    {
                        "emotion": self.emotion_labels[top_idx],
                        "confidence": confidence,
                        "all_scores": {
                            self.emotion_labels[k]: float(prob[k])
                            for k in range(len(prob))
                        },
                        "is_confident": confidence >= self.emotion_threshold,
                    }
                )

            return results

        except Exception as exc:
            print(f"WARN: Emotion detection failed: {exc}")
            return [None] * len(audio_windows)

    def _detect_audio_events(self, audio_windows: List[np.ndarray]) -> List[List[Dict]]:
        """Detect various audio events (batch processing).

        Uses pre-computed ``_category_np_indices`` to avoid per-window string
        matching.  Probability extraction is done via numpy fancy-indexing
        across the entire batch for each category.
        """
        if self.audio_model is None or not self._category_np_indices:
            return [[] for _ in audio_windows]

        try:
            all_results: List[List[Dict]] = []

            for i in range(0, len(audio_windows), self.batch_size):
                batch = audio_windows[i : i + self.batch_size]
                batch_size = len(batch)

                # Process batch through AST model
                inputs = self.audio_processor(
                    batch, sampling_rate=self.target_sr, return_tensors="pt"
                )
                _dtype = next(self.audio_model.parameters()).dtype
                inputs = {k: v.to(device=self.device, dtype=_dtype) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = self.audio_model(**inputs).logits.float()
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                probs = probs.cpu().numpy()  # shape: (batch_size, num_labels)

                # Initialize per-window event lists
                batch_events: List[List[Dict]] = [[] for _ in range(batch_size)]

                # Vectorized per-category detection across entire batch
                for category, (indices, labels) in self._category_np_indices.items():
                    # Fancy-index: extract columns for matching labels
                    cat_probs = probs[:, indices]  # (batch_size, n_matching)
                    max_local_idx = np.argmax(cat_probs, axis=1)  # (batch_size,)
                    max_prob = np.max(cat_probs, axis=1)  # (batch_size,)

                    # Only create dicts for windows that exceed the threshold
                    above = max_prob > self.audio_event_threshold
                    for j in np.nonzero(above)[0]:
                        batch_events[int(j)].append(
                            {
                                "category": category,
                                "label": labels[int(max_local_idx[j])],
                                "confidence": float(max_prob[j]),
                            }
                        )

                all_results.extend(batch_events)

            return all_results

        except Exception as exc:
            print(f"WARN: Audio event detection failed: {exc}")
            return [[] for _ in audio_windows]

    def _detect_energy_changes(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect significant energy changes with improved algorithm."""
        # Calculate RMS energy with shorter frames for better temporal resolution
        frame_length = int(0.05 * sr)  # 50ms frames
        hop_length = int(0.025 * sr)  # 25ms hop

        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Smooth RMS to reduce noise
        rms_smooth = median_filter(rms, size=5)

        # Find peaks with adaptive threshold
        threshold = np.mean(rms_smooth) + 0.5 * np.std(rms_smooth)
        peaks, properties = find_peaks(
            rms_smooth,
            height=threshold,
            prominence=np.std(rms_smooth) * 0.3,
            distance=int(0.5 * sr / hop_length),  # At least 0.5s apart
        )

        events = []
        for peak in peaks:
            time = librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)
            events.append(
                {
                    "time": float(time),
                    "type": "energy_peak",
                    "energy": float(rms_smooth[peak]),
                    "prominence": float(
                        properties["prominences"][list(peaks).index(peak)]
                    ),
                }
            )

        return events

    def _apply_temporal_smoothing(
        self, segments: List[Dict], window: int = 3
    ) -> List[Dict]:
        """Apply temporal smoothing to reduce jitter in classifications.

        Uses a voting mechanism across neighboring segments for more stable results.
        """
        if len(segments) < window or window < 2:
            return segments

        smoothed = []

        for i, seg in enumerate(segments):
            # Get neighboring segments
            start_idx = max(0, i - window // 2)
            end_idx = min(len(segments), i + window // 2 + 1)
            neighbors = segments[start_idx:end_idx]

            # Count occurrences of each emotion/category
            if "emotion" in seg:
                key = "emotion"
            elif "category" in seg:
                key = "category"
            else:
                smoothed.append(seg)
                continue

            # Majority voting
            counts = defaultdict(float)
            for n in neighbors:
                if key in n:
                    counts[n[key]] += n.get("confidence", 1.0)

            if counts:
                # Get the most common value weighted by confidence
                most_common = max(counts.items(), key=lambda x: x[1])
                seg_copy = seg.copy()
                seg_copy[key] = most_common[0]
                smoothed.append(seg_copy)
            else:
                smoothed.append(seg)

        return smoothed

    def _merge_consecutive_segments(
        self, segments: List[Dict], max_gap: float = 1.5
    ) -> List[Dict]:
        """Merge consecutive segments with improved algorithm.

        Args:
            segments: List of segment dictionaries
            max_gap: Maximum gap in seconds to merge segments
        """
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x["start"])

        merged = [segments[0].copy()]

        for current in segments[1:]:
            last = merged[-1]

            # Determine if segments are of the same type
            same_type = False
            if "emotion" in current and "emotion" in last:
                same_type = current["emotion"] == last["emotion"]
            elif "category" in current and "category" in last:
                same_type = current["category"] == last["category"]

            # Check if close together and same type
            gap = current["start"] - last["end"]

            if same_type and gap < max_gap:
                # Merge: extend the end time and update confidence
                last["end"] = current["end"]

                # Weighted average of confidence based on duration
                if "confidence" in current and "confidence" in last:
                    last_duration = last["end"] - last["start"]
                    current_duration = current["end"] - current["start"]
                    total_duration = last_duration + current_duration

                    last["confidence"] = (
                        last["confidence"] * last_duration
                        + current["confidence"] * current_duration
                    ) / total_duration
            else:
                # Filter out very short segments (likely noise)
                if (last["end"] - last["start"]) >= self.min_segment_duration:
                    pass  # Keep it
                else:
                    merged.pop()  # Remove it

                merged.append(current.copy())

        # Filter last segment
        if (
            merged
            and (merged[-1]["end"] - merged[-1]["start"]) < self.min_segment_duration
        ):
            merged.pop()

        return merged

    def analyze(self, audio_path: str) -> Tuple[Dict, np.ndarray, int]:
        """Analyze audio file and return timeline of events.

        Returns:
            Tuple of (results_dict, audio_waveform, sample_rate).
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Extract windows
        self._debug("DEBUG: Extracting audio windows")
        windows = self._extract_windows(audio, sr)
        self._debug(f"DEBUG: Processing {len(windows)} window(s)")

        # Analyze each window using batched inference
        emotion_segments: List[Dict] = []
        audio_event_segments: List[Dict] = []

        speech_segments = self._detect_voice_activity(audio, sr)

        if windows:
            window_arrays = [window for _, _, window in windows]
            audio_event_results = self._detect_audio_events(window_arrays)

            speech_windows: List[np.ndarray] = []
            speech_mask: List[bool] = []
            for i, (start, end, _) in enumerate(windows):
                events = audio_event_results[i] if i < len(audio_event_results) else []
                should_analyze = self._should_analyze_emotion(
                    start, end, events, speech_segments
                )
                speech_mask.append(should_analyze)
                if should_analyze:
                    speech_windows.append(window_arrays[i])

            detected_emotions = (
                self._detect_emotion(speech_windows) if speech_windows else []
            )
            emotion_results: List[Optional[Dict]] = [None] * len(windows)
            speech_idx = 0
            for i, should_analyze in enumerate(speech_mask):
                if should_analyze:
                    if speech_idx < len(detected_emotions):
                        emotion_results[i] = detected_emotions[speech_idx]
                    speech_idx += 1
        else:
            emotion_results: List[Optional[Dict]] = []
            audio_event_results = []

        for i, (start, end, _window) in enumerate(windows):
            if (i + 1) % 10 == 0:
                self._debug(f"DEBUG: Processed {i + 1}/{len(windows)} windows")

            emotion = emotion_results[i] if i < len(emotion_results) else None
            if emotion and emotion.get("confidence", 0.0) >= self.emotion_threshold:
                emotion_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "emotion": emotion["emotion"],
                        "confidence": emotion["confidence"],
                        "scores": emotion.get("all_scores", {}),
                    }
                )

            events = audio_event_results[i] if i < len(audio_event_results) else []
            for event in events:
                audio_event_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "category": event["category"],
                        "label": event["label"],
                        "confidence": event["confidence"],
                    }
                )

        # Merge consecutive segments
        self._debug("DEBUG: Merging consecutive segments")
        emotion_segments = self._apply_temporal_smoothing(emotion_segments)
        emotion_segments = self._merge_consecutive_segments(emotion_segments)
        audio_event_segments = self._merge_consecutive_segments(audio_event_segments)

        # Detect energy changes
        self._debug("DEBUG: Detecting energy changes")
        energy_events = self._detect_energy_changes(audio, sr)

        # Compile results
        results = {
            "file": str(audio_path),
            "duration": float(len(audio) / sr),
            "sample_rate": int(sr),
            "analysis_parameters": {
                "window_size": self.window_size,
                "hop_size": self.hop_size,
            },
            "timeline": {
                "emotions": emotion_segments,
                "audio_events": audio_event_segments,
                "energy_peaks": energy_events,
            },
            "summary": {
                "total_emotion_segments": len(emotion_segments),
                "total_audio_events": len(audio_event_segments),
                "total_energy_peaks": len(energy_events),
            },
        }

        return results, audio, sr

    def export_to_json(self, results: Dict, output_path: Union[str, Path]):
        """Export results to JSON file."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self._debug(f"DEBUG: Results exported to {output_path_obj}")


def _assign_ids_and_export_segments(
    results: Dict,
    audio_path: Union[str, Path],
    classification_path: Path,
    *,
    energy_window: float = 1.0,
    debug: bool = False,
    preloaded_audio: Optional[np.ndarray] = None,
    preloaded_sr: Optional[int] = None,
) -> None:
    """Assign IDs to timeline entries and export corresponding audio snippets."""

    classification_path.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path)

    if preloaded_audio is not None and preloaded_sr is not None:
        audio = preloaded_audio
        sr = preloaded_sr
    else:
        if not audio_path.exists():
            print(f"Warning: Audio file for segment export not found: {audio_path}")
            return

        sr = int(results.get("sample_rate", 16000))
        try:
            with gray_debug_output(debug):
                audio, sr = librosa.load(str(audio_path), sr=sr, mono=True)
        except Exception as exc:
            print(f"WARN: Failed to load audio for segment export: {exc}")
            return

    total_samples = len(audio)
    duration = float(results.get("duration", total_samples / sr))

    def clamp_interval(
        start: float, end: float, minimum: float = 0.25
    ) -> Tuple[float, float]:
        start = max(0.0, float(start))
        end = min(duration, float(end))
        if end - start < minimum:
            end = min(duration, start + minimum)
        return start, end

    def write_clip(
        kind: str, segment: Dict, start: float, end: float, idx: int
    ) -> Optional[str]:
        segment_dir = classification_path / kind
        segment_dir.mkdir(parents=True, exist_ok=True)

        start_idx = max(0, int(round(start * sr)))
        end_idx = min(total_samples, int(round(end * sr)))
        if end_idx <= start_idx:
            return None

        clip_path = segment_dir / f"{idx}.wav"
        try:
            sf.write(str(clip_path), audio[start_idx:end_idx], sr)
        except Exception as exc:
            print(f"Warning: Failed to write segment {kind}#{idx}: {exc}")
            return None

        segment["clip_start"] = start
        segment["clip_end"] = end
        rel_path = clip_path.relative_to(classification_path)
        segment["clip_path"] = str(rel_path)
        return str(rel_path)

    timeline = results.get("timeline", {})

    # Emotions
    for idx, segment in enumerate(timeline.get("emotions", []), start=1):
        segment["id"] = idx
        start, end = clamp_interval(segment.get("start", 0.0), segment.get("end", 0.0))
        write_clip("emotions", segment, start, end, idx)

    # Audio events
    for idx, segment in enumerate(timeline.get("audio_events", []), start=1):
        segment["id"] = idx
        start, end = clamp_interval(segment.get("start", 0.0), segment.get("end", 0.0))
        write_clip("audio_events", segment, start, end, idx)

    # Energy peaks (window around the peak time)
    half_window = energy_window / 2.0
    for idx, segment in enumerate(timeline.get("energy_peaks", []), start=1):
        segment["id"] = idx
        peak_time = float(segment.get("time", 0.0))
        start, end = clamp_interval(peak_time - half_window, peak_time + half_window)
        write_clip("energy_peaks", segment, start, end, idx)


def _run_timeline_classification(
    audio_path: Union[str, Path],
    classification_dir: Union[str, Path],
    *,
    device: Optional[str] = None,
    batch_size: int = 32,
    window_size: float = 2.0,
    hop_size: float = 2.0,
    target_sample_rate: int = 16000,
    emotion_threshold: float = 0.12,
    audio_event_threshold: float = 0.35,
    min_segment_duration: float = 0.5,
    emotion_temperature: float = 0.7,
    emotion_prob_smoothing: int = 5,
    emotion_confidence_gamma: float = 1.35,
    min_speech_overlap_ratio: float = 0.15,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: int = 250,
    vad_min_silence_duration_ms: int = 100,
    energy_window: float = 1.0,
    output_filename: str = "timeline_classification.json",
    debug: bool = False,
) -> Dict:
    """Convenience helper to analyze audio and persist timeline classification results.

    Args:
        audio_path: Path to the source audio file.
        classification_dir: Directory under which results should be written.
        device: Optional device override for the classifier (defaults to auto selection).
        batch_size: Optional batch size override for analysis.
        output_filename: Name of the JSON file to emit within the classification directory.

    Returns:
        A dictionary containing the analysis results and metadata about the written file.
    """

    info_print("Performing timeline audio classification")

    classifier = _AudioClassifier(
        device=device,
        batch_size=batch_size,
        window_size=window_size,
        hop_size=hop_size,
        target_sample_rate=target_sample_rate,
        emotion_threshold=emotion_threshold,
        audio_event_threshold=audio_event_threshold,
        min_segment_duration=min_segment_duration,
        emotion_temperature=emotion_temperature,
        emotion_prob_smoothing=emotion_prob_smoothing,
        emotion_confidence_gamma=emotion_confidence_gamma,
        min_speech_overlap_ratio=min_speech_overlap_ratio,
        vad_threshold=vad_threshold,
        vad_min_speech_duration_ms=vad_min_speech_duration_ms,
        vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        debug=debug,
    )
    results, audio_waveform, audio_sr = classifier.analyze(str(audio_path))

    classification_path = Path(classification_dir)
    classification_path.mkdir(parents=True, exist_ok=True)

    output_path = classification_path / output_filename
    _assign_ids_and_export_segments(
        results,
        results.get("file", audio_path),
        classification_path,
        energy_window=energy_window,
        debug=debug,
        preloaded_audio=audio_waveform,
        preloaded_sr=audio_sr,
    )
    classifier.export_to_json(results, output_path)

    return {"results": results, "output_path": str(output_path)}


def handle(
    input_file: str,
    output_folder: str,
    config: "TimelineConfig | None" = None,
    *,
    debug: bool = False,
) -> Dict:
    """Perform timeline-based audio classification.

    This is the standardized entry point for the timeline classification module.

    Args:
        input_file: Path to the input audio file.
        output_folder: Directory where output files will be written.
        config: TimelineConfig instance with classification parameters, or None for defaults.
        debug: If True, emit verbose debug output.

    Returns:
        Dictionary with classification results including emotion and audio event timelines.
    """
    audio_path = Path(input_file)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    working_dir = Path(output_folder)
    classification_dir = working_dir / "classification"

    result = _run_timeline_classification(
        audio_path=audio_path,
        classification_dir=classification_dir,
        device=config.device if config else None,
        batch_size=config.batch_size if config else 32,
        window_size=config.window_size if config else 2.0,
        hop_size=config.hop_size if config else 2.0,
        target_sample_rate=config.target_sample_rate if config else 16000,
        emotion_threshold=config.emotion_threshold if config else 0.12,
        audio_event_threshold=config.audio_event_threshold if config else 0.35,
        min_segment_duration=config.min_segment_duration if config else 0.5,
        emotion_temperature=config.emotion_temperature if config else 0.7,
        emotion_prob_smoothing=config.emotion_prob_smoothing if config else 5,
        emotion_confidence_gamma=config.emotion_confidence_gamma if config else 1.35,
        min_speech_overlap_ratio=config.min_speech_overlap_ratio if config else 0.15,
        vad_threshold=config.vad_threshold if config else 0.5,
        vad_min_speech_duration_ms=config.vad_min_speech_duration_ms if config else 250,
        vad_min_silence_duration_ms=config.vad_min_silence_duration_ms
        if config
        else 100,
        energy_window=config.energy_window if config else 1.0,
        debug=debug,
    )

    return result
