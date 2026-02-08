"""Chapter identification using embeddings and an optional local LLM."""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.signal import find_peaks
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.logging import debug_print, gray_debug_output

# Import config type for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import ScenesConfig

warnings.filterwarnings("ignore")

try:
    import nltk
except ModuleNotFoundError:
    nltk = None  # type: ignore

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass

    # Newer NLTK releases split punkt tables into a separate package.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass


__all__ = ["handle"]


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class Chapter:
    title: str
    start: float
    end: float
    summary: str
    full_text: str


def _paths_equal(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _resolve_segment_source(
    path_or_working_dir: str,
    maybe_working_dir: Optional[str],
    *,
    debug: bool,
) -> Tuple[Path, Optional[Path]]:
    explicit_path = Path(path_or_working_dir)
    working_dir = Path(maybe_working_dir) if maybe_working_dir else None

    if explicit_path.is_dir() and working_dir is None:
        working_dir = explicit_path
        explicit_path = working_dir / "transcription" / "transcription.json"

    if working_dir is None and explicit_path.is_file():
        parent = explicit_path.parent
        if parent.name in {"ctc", "transcription"}:
            working_dir = parent.parent
        else:
            working_dir = parent

    forced_path = working_dir / "ctc" / "alignment.json" if working_dir else None
    transcript_path = (
        working_dir / "transcription" / "transcription.json" if working_dir else None
    )

    selected: Optional[Path] = None

    if forced_path and forced_path.is_file():
        if explicit_path.exists() and _paths_equal(explicit_path, forced_path):
            debug_print(
                f"Using forced alignment segments from {forced_path}",
                debug=debug,
            )
            selected = forced_path
        elif (
            transcript_path
            and transcript_path.is_file()
            and explicit_path.exists()
            and _paths_equal(explicit_path, transcript_path)
        ):
            debug_print(
                f"Forced alignment available at {forced_path}; overriding transcript {explicit_path}",
                debug=debug,
            )
            debug_print(
                f"Using forced alignment segments from {forced_path}",
                debug=debug,
            )
            selected = forced_path
        elif not explicit_path.exists() or explicit_path.is_dir():
            debug_print(
                f"Using forced alignment segments from {forced_path}",
                debug=debug,
            )
            selected = forced_path

    if selected is None and explicit_path.exists() and explicit_path.is_file():
        selected = explicit_path

    if selected is None and transcript_path and transcript_path.is_file():
        debug_print(
            f"Falling back to transcription segments from {transcript_path}",
            debug=debug,
        )
        selected = transcript_path

    if selected is None:
        raise FileNotFoundError(
            f"No transcript-style segments found near {explicit_path}"
        )

    return selected, working_dir


class TitleGenerator:
    """Lightweight LLM-based title generator for YouTube-style chapter titles."""

    _instance: Optional["TitleGenerator"] = None
    _model_name: str = ""

    def __init__(
        self,
        model_name: str,
        *,
        debug: bool = False,
    ) -> None:
        if not model_name:
            raise ValueError("TitleGenerator requires a model_name")
        self.model_name = model_name
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load()

    @classmethod
    def get_instance(cls, model_name: str, *, debug: bool = False) -> "TitleGenerator":
        """Get or create a singleton instance to avoid reloading the model."""
        if not model_name:
            raise ValueError("TitleGenerator.get_instance requires a model_name")
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = cls(model_name=model_name, debug=debug)
            cls._model_name = model_name
        return cls._instance

    def _load(self) -> None:
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        debug_print(
            f"Loading title generator '{self.model_name}' on {self.device}...",
            debug=self.debug,
        )

        with gray_debug_output(self.debug):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.model.to(torch.device(self.device))  # type: ignore[arg-type, union-attr]
            self.model.eval()

        debug_print("Title generator ready", debug=self.debug)

    def generate_title(self, text: str, max_chars: int = 50) -> str:
        """Generate a concise YouTube-style chapter title from transcript text."""
        # Truncate input to avoid context overflow
        truncated = text[:1500] if len(text) > 1500 else text

        system_prompt = (
            "You create YouTube video chapter titles. "
            "Rules: 3-6 words, describe the SPECIFIC topic discussed, avoid generic phrases. "
            "Examples: 'What Are AI Agents', 'Building Your First Agent', 'Agent Collaboration Patterns'. "
            "Output ONLY the title, nothing else."
        )

        user_prompt = f"Create a chapter title for:\n\n{truncated}"

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(  # type: ignore[union-attr]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nTitle:"

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt")  # type: ignore[misc]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            output_ids = self.model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=32,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        title = self.tokenizer.decode(generated, skip_special_tokens=True).strip()  # type: ignore[union-attr]

        # Clean up the title
        title = self._clean_title(title, max_chars)
        return title

    def _clean_title(self, title: str, max_chars: int) -> str:
        """Clean and normalize the generated title."""
        # Remove common unwanted prefixes
        prefixes_to_remove = [
            "Title:",
            "Chapter:",
            "Chapter Title:",
            "Here's",
            "Here is",
            "The title is",
            "A good title would be",
            "**",
            "##",
            "Sure,",
        ]
        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix) :].strip()

        # Remove quotes
        title = title.strip("\"'`*#")

        # Take only the first line if multiple lines
        title = title.split("\n")[0].strip()

        # If title has a colon, take the part after it (usually more specific)
        if ":" in title:
            parts = title.split(":", 1)
            # Use the longer, more descriptive part
            after_colon = parts[1].strip()
            if len(after_colon) > 10:
                title = after_colon
            elif len(parts[0].strip()) > len(after_colon):
                title = parts[0].strip()

        # Remove trailing punctuation except for question marks
        while title and title[-1] in ".,;:!":
            title = title[:-1]

        # Truncate if too long, keeping whole words
        if len(title) > max_chars:
            words = title[:max_chars].rsplit(" ", 1)
            title = words[0] if len(words) > 1 else title[:max_chars]

        # Ensure title is not empty
        if not title or len(title) < 3:
            return "Discussion"

        return title


class LocalLLM:
    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        debug: bool = False,
    ) -> None:
        if not model_name:
            raise ValueError("LocalLLM requires a model_name")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.debug = debug
        self.device = self._resolve_device()
        self._load()

    def _resolve_device(self) -> str:
        requested = os.getenv("AUDIO_SCENES_LLM_DEVICE")
        if requested in {"cpu", "cuda"}:
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if not self.model_name:
            raise ValueError("LocalLLM requires a model name")

        trust_remote_code = os.getenv("AUDIO_SCENES_TRUST_REMOTE_CODE", "1") != "0"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        message = f"Loading local LLM '{self.model_name}' on {self.device}..."
        debug_print(message, debug=self.debug)
        with gray_debug_output(self.debug):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code,
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            self.model.to(torch.device(self.device))  # type: ignore[arg-type, union-attr]

        debug_print("Local LLM ready", debug=self.debug)

    def _apply_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"{system_prompt}\n\n{user_prompt}\n"

    def _generate(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> str:
        with gray_debug_output(self.debug):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def summarize(
        self,
        *,
        text: str,
        start: float,
        end: float,
        global_context: str = "",
        context_before: str = "",
        context_after: str = "",
    ) -> Dict[str, str]:
        system_prompt = (
            "You craft original, editorial-quality chapter metadata for longform audio or video transcripts."
            " Respond with compact JSON containing only the keys 'title' and 'summary'."
            " Title requirements: 4-8 words, specific to this segment, no numbering or generic filler, and avoid reusing transcript phrasing verbatim."
            " Summary requirements: two or three crisp sentences that highlight the most important ideas, include one concrete takeaway or insight, and avoid copying sentences directly from the transcript."
            " Keep the tone informative, confident, and audience-friendly."
        )

        start_ts = ChapterIdentifier.format_timestamp_static(start)
        end_ts = ChapterIdentifier.format_timestamp_static(end)

        context_sections: List[str] = []
        if global_context:
            context_sections.append(
                "Overall program context:\n" + global_context.strip()
            )
        if context_before:
            context_sections.append(
                "Context before segment:\n" + context_before.strip()
            )
        if context_after:
            context_sections.append("Context after segment:\n" + context_after.strip())

        user_prompt_parts = context_sections + [
            "Transcript segment:\n" + text.strip(),
            f"Start: {start_ts}",
            f"End: {end_ts}",
            "Return valid JSON, nothing else.",
        ]
        user_prompt = "\n\n".join(part for part in user_prompt_parts if part.strip())

        prompt = self._apply_chat_template(system_prompt, user_prompt)
        raw = self._generate(prompt)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("LLM did not respond with JSON")

        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Unexpected LLM payload type")

        return {
            "title": str(payload.get("title", "")).strip(),
            "summary": str(payload.get("summary", "")).strip(),
        }

    def summarize_context(
        self,
        *,
        text: str,
        instruction: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        system_prompt = instruction
        user_prompt = text.strip() + "\n\nReturn only the summary."
        prompt = self._apply_chat_template(system_prompt, user_prompt)
        return self._generate(prompt, max_new_tokens=max_new_tokens)


class ChapterIdentifier:
    def __init__(
        self,
        *,
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        title_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        min_chapter_length: float = 120.0,
        boundary_percentile: float = 75.0,
        max_new_tokens: int = 256,
        max_ctx_chars: int = 6000,
        global_chunk_chars: int = 4000,
        context_snippet_chars: int = 1200,
        global_chunk_tokens: int = 160,
        global_summary_tokens: int = 220,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_llm: bool = True,
        debug: bool = False,
    ) -> None:
        self.debug = debug

        debug_print(
            f"Loading sentence transformer '{embed_model}'...", debug=self.debug
        )
        embed_trust_remote = os.getenv("AUDIO_SCENES_TRUST_REMOTE_EMBED", "1") != "0"
        with gray_debug_output(self.debug):
            self.embedder = SentenceTransformer(
                embed_model,
                trust_remote_code=embed_trust_remote,
            )
        debug_print("Sentence transformer ready", debug=self.debug)

        self.stop_words = set(stopwords.words("english"))
        self.min_chapter_length = min_chapter_length
        self.boundary_percentile = boundary_percentile
        self.max_ctx_chars = max_ctx_chars
        self.global_chunk_chars = global_chunk_chars
        self.context_snippet_chars = context_snippet_chars
        self.global_chunk_tokens = global_chunk_tokens
        self.global_summary_tokens = global_summary_tokens
        self.global_context_summary: str = ""
        self.global_keywords: List[str] = []

        self.llm: Optional[LocalLLM] = None
        if enable_llm:
            try:
                self.llm = LocalLLM(
                    model_name=llm_model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    debug=self.debug,
                )
            except Exception as exc:
                print(
                    f"WARN: Failed to load local LLM ({exc}); falling back to heuristics."
                )

        # Load lightweight title generator
        self.title_generator: Optional[TitleGenerator] = None
        if enable_llm:
            try:
                self.title_generator = TitleGenerator.get_instance(
                    model_name=title_model,
                    debug=self.debug,
                )
            except Exception as exc:
                print(
                    f"WARN: Failed to load title generator ({exc}); using fallback titles."
                )

    def load_transcript(self, filepath: str) -> List[Segment]:
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "segments" in data:
            raw_segments = data["segments"]
        elif isinstance(data, list):
            raw_segments = data
        else:
            raise ValueError(
                "Transcript must be a list or {'segments': [...]} structure"
            )

        segments: List[Segment] = []
        previous_end = 0.0

        def _coerce_time(value: object, default: float) -> float:
            if value is None:
                return default
            try:
                return float(str(value))
            except (TypeError, ValueError):
                return default

        for entry in raw_segments:
            text = str(entry.get("text", "")).strip()

            if not text and isinstance(entry.get("words"), list):
                tokens = [
                    str(word.get("word", "")).strip()
                    for word in entry["words"]
                    if isinstance(word, dict) and str(word.get("word", "")).strip()
                ]
                text = " ".join(tokens).strip()

            if not text and isinstance(entry.get("transcript"), str):
                text = entry["transcript"].strip()

            if not text:
                continue

            start_default = previous_end
            start = _coerce_time(entry.get("start"), start_default)
            end = _coerce_time(entry.get("end"), start)

            if end < start:
                end = start

            previous_end = end
            segments.append(Segment(text=text, start=start, end=end))

        debug_print(
            f"Loaded {len(segments)} usable segments from {filepath}",
            debug=self.debug,
        )

        return segments

    def _compute_dissimilarity(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.shape[0] < 2:
            return np.array([])

        sims = np.sum(embeddings[1:] * embeddings[:-1], axis=1)
        scores = 1.0 - sims
        if scores.size < 3:
            return scores

        kernel = np.ones(3, dtype=np.float32) / 3.0
        return np.convolve(scores, kernel, mode="same")

    def _select_peaks(
        self,
        dissimilarity: np.ndarray,
        segments: Sequence[Segment],
        max_chapters: Optional[int],
    ) -> List[int]:
        if dissimilarity.size == 0:
            return []

        avg_duration = np.mean([max(seg.end - seg.start, 1.0) for seg in segments])
        min_distance = max(
            1, int(math.ceil(self.min_chapter_length / max(avg_duration, 1.0)))
        )
        threshold = np.percentile(dissimilarity, self.boundary_percentile)

        peaks, properties = find_peaks(
            dissimilarity,
            height=threshold,
            distance=min_distance,
            prominence=0.05,
        )

        if max_chapters and max_chapters > 1 and peaks.size > max_chapters - 1:
            heights = properties.get("peak_heights")
            if heights is not None:
                order = np.argsort(heights)[-(max_chapters - 1) :]
                peaks = np.sort(peaks[order])

        return peaks.tolist()

    def _merge_short_spans(
        self, boundaries: List[int], segments: Sequence[Segment]
    ) -> List[int]:
        if len(boundaries) <= 2:
            return boundaries

        merged = [boundaries[0]]
        for idx in range(1, len(boundaries)):
            start_idx = merged[-1]
            end_idx = boundaries[idx]
            slice_segments = segments[start_idx:end_idx]
            if not slice_segments:
                continue

            duration = slice_segments[-1].end - slice_segments[0].start
            if duration < self.min_chapter_length and idx < len(boundaries) - 1:
                continue

            merged.append(end_idx)

        if merged[-1] != boundaries[-1]:
            merged.append(boundaries[-1])

        return merged

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.max_ctx_chars:
            return text

        sentences = sent_tokenize(text)
        selected: List[str] = []
        total = 0
        for sentence in sentences:
            if total + len(sentence) > self.max_ctx_chars:
                break
            selected.append(sentence)
            total += len(sentence)
        return " ".join(selected)

    def _limit_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text

        sentences = sent_tokenize(text)
        selected: List[str] = []
        total = 0
        for sentence in sentences:
            if total + len(sentence) > limit:
                break
            selected.append(sentence)
            total += len(sentence)

        if selected:
            return " ".join(selected)
        return text[:limit]

    def _summarize_block(self, text: str, instruction: str, max_tokens: int) -> str:
        trimmed = self._limit_text(text, self.max_ctx_chars)
        if self.llm:
            try:
                result = self.llm.summarize_context(
                    text=trimmed,
                    instruction=instruction,
                    max_new_tokens=max_tokens,
                )
                if result.strip():
                    return result.strip()
            except Exception as exc:
                debug_print(
                    f"LLM context summarisation failed ({exc}); using fallback heuristics.",
                    debug=self.debug,
                )
        return self._fallback_summary(trimmed)

    def _compute_global_keywords(
        self, texts: Sequence[str], limit: int = 12
    ) -> List[str]:
        if not texts:
            return []

        vectorizer = TfidfVectorizer(max_features=limit * 2, stop_words="english")
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        # Convert sparse matrix to array for sum operation
        weights = np.asarray(matrix.toarray().sum(axis=0)).ravel()  # type: ignore[union-attr]
        terms = vectorizer.get_feature_names_out()
        order = np.argsort(weights)[::-1]
        keywords: List[str] = []
        for idx in order:
            term = str(terms[idx])
            if term.isdigit():
                continue
            keywords.append(term)
            if len(keywords) >= limit:
                break
        return keywords

    def _build_global_context(self, segments: Sequence[Segment]) -> None:
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        self.global_keywords = self._compute_global_keywords(texts)

        if not texts:
            self.global_context_summary = ""
            return

        chunk_texts: List[str] = []
        current: List[str] = []
        current_len = 0
        for text in texts:
            length = len(text)
            if current and current_len + length > self.global_chunk_chars:
                chunk_texts.append(" ".join(current))
                current = [text]
                current_len = length
            else:
                current.append(text)
                current_len += length

        if current:
            chunk_texts.append(" ".join(current))

        if not chunk_texts:
            self.global_context_summary = ""
            return

        chunk_instruction = (
            "You summarise transcript chunks to retain key topics, decisions, and speakers."
            " Provide 2-3 sentences highlighting the most important ideas in neutral tone."
        )
        chunk_summaries: List[str] = []
        for chunk in chunk_texts:
            summary = self._summarize_block(
                chunk, chunk_instruction, self.global_chunk_tokens
            )
            if summary:
                chunk_summaries.append(summary.strip())

        if not chunk_summaries:
            self.global_context_summary = self._fallback_summary(" ".join(texts))
            return

        combined_basis = " ".join(chunk_summaries)
        final_instruction = (
            "You combine chunk summaries into a single overview of the entire programme."
            " Deliver 3 sentences covering overarching themes, recurring challenges, and outcomes."
        )
        final_summary = self._summarize_block(
            combined_basis,
            final_instruction,
            self.global_summary_tokens,
        )

        if final_summary:
            self.global_context_summary = final_summary.strip()
        else:
            self.global_context_summary = chunk_summaries[0]

    def _collect_neighbor_context(
        self,
        segments: Sequence[Segment],
        start_idx: int,
        end_idx: int,
    ) -> Tuple[str, str]:
        before_parts: List[str] = []
        before_len = 0
        idx = start_idx - 1
        while idx >= 0 and before_len < self.context_snippet_chars:
            text = segments[idx].text.strip()
            if text:
                before_parts.append(text)
                before_len += len(text)
            idx -= 1
        before_text = " ".join(reversed(before_parts))

        after_parts: List[str] = []
        after_len = 0
        idx = end_idx
        total_segments = len(segments)
        while idx < total_segments and after_len < self.context_snippet_chars:
            text = segments[idx].text.strip()
            if text:
                after_parts.append(text)
                after_len += len(text)
            idx += 1
        after_text = " ".join(after_parts)

        return (
            self._limit_text(before_text, self.context_snippet_chars),
            self._limit_text(after_text, self.context_snippet_chars),
        )

    def _extract_key_phrase(self, text: str) -> str:
        """Extract a meaningful phrase from the first question or statement."""
        sentences = sent_tokenize(text)
        if not sentences:
            return ""

        # Look for a question in the first few sentences - often a good topic indicator
        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if sentence.endswith("?") and len(sentence) > 15:
                # Extract the core of the question as a title
                # Remove question words and clean up
                cleaned = sentence.rstrip("?").strip()
                words = cleaned.split()
                # Skip leading question words
                skip_words = {
                    "what",
                    "who",
                    "where",
                    "when",
                    "why",
                    "how",
                    "is",
                    "are",
                    "do",
                    "does",
                    "can",
                    "could",
                    "would",
                    "should",
                    "so",
                }
                start_idx = 0
                for i, word in enumerate(words):
                    if word.lower() not in skip_words:
                        start_idx = i
                        break
                if start_idx < len(words):
                    phrase = " ".join(words[start_idx:])
                    # Limit to reasonable title length
                    phrase_words = phrase.split()[:7]
                    if len(phrase_words) >= 2:
                        return " ".join(
                            w.capitalize() if i == 0 else w
                            for i, w in enumerate(phrase_words)
                        )

        # Fall back to first meaningful sentence, extract key noun phrases
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 10:
            words = first_sentence.split()[:8]
            # Clean up and capitalize
            cleaned_words = [
                w.strip(".,!?:;\"'") for w in words if w.strip(".,!?:;\"'")
            ]
            if len(cleaned_words) >= 2:
                return " ".join(
                    w.capitalize() if i == 0 else w
                    for i, w in enumerate(cleaned_words[:6])
                )

        return ""

    def _fallback_title(self, text: str) -> str:
        # First, try to extract a meaningful phrase from the text
        phrase_title = self._extract_key_phrase(text)
        if phrase_title and len(phrase_title) > 10:
            return phrase_title

        # Identify key topics using noun-like word patterns
        tokens = [w.lower() for w in word_tokenize(text) if w.isalnum()]
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 3]
        if not tokens:
            return "Overview"

        # Count word frequency within this chapter
        word_freq: Dict[str, int] = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

        # Prioritize words that appear in global keywords (topic-relevant)
        scored_words: List[Tuple[str, float]] = []
        for word, freq in word_freq.items():
            score = freq
            # Boost words that match global keywords
            if self.global_keywords and word in self.global_keywords:
                score *= 2.0
            # Slight boost for longer words (often more specific)
            if len(word) > 6:
                score *= 1.2
            scored_words.append((word, score))

        # Sort by score descending
        scored_words.sort(key=lambda x: x[1], reverse=True)
        top_words = [w for w, _ in scored_words[:4]]

        if not top_words:
            return "Overview"

        # Try to form a more coherent title
        # Look for common patterns in the text that include these words
        text_lower = text.lower()
        for word in top_words[:2]:
            # Find phrases containing this key word
            pattern = rf"\b(\w+\s+)?{re.escape(word)}(\s+\w+)?(\s+\w+)?\b"
            matches = re.findall(pattern, text_lower)
            if matches:
                # Get the first good match
                for match in matches[:3]:
                    phrase = " ".join(p.strip() for p in match if p.strip())
                    phrase_words = phrase.split()
                    # Filter out stop words from the phrase
                    phrase_words = [
                        w for w in phrase_words if w not in self.stop_words or w == word
                    ]
                    if 2 <= len(phrase_words) <= 5:
                        # Capitalize appropriately
                        title = " ".join(w.capitalize() for w in phrase_words)
                        if len(title) > 8:
                            return title

        # Last resort: use top keywords but try to make it readable
        # Add a connecting word if we have topic-specific terms
        if len(top_words) >= 2:
            # Format as "Topic1 and Topic2" or similar
            return f"{top_words[0].capitalize()} and {top_words[1].capitalize()}"

        return top_words[0].capitalize() if top_words else "Overview"

    def _fallback_summary(self, text: str, global_context: str = "") -> str:
        sentences = sent_tokenize(text)
        if not sentences:
            summary_text = text.strip()[:200]
        else:
            summary_text = " ".join(sentences[:2]).strip()

        if global_context:
            context_sentences = sent_tokenize(global_context)
            if context_sentences:
                summary_text = f"{summary_text} Overall: {context_sentences[0]}"

        return summary_text.strip()

    def _generate_chapter(
        self,
        text: str,
        start: float,
        end: float,
        context_before: str,
        context_after: str,
    ) -> Chapter:
        truncated = self._truncate_text(text)
        title = ""
        summary = ""

        # Try to generate title using the lightweight title generator
        if self.title_generator:
            try:
                title = self.title_generator.generate_title(truncated)
                debug_print(f"Generated title: {title}", debug=self.debug)
            except Exception as exc:
                debug_print(
                    f"Title generation failed ({exc}); using fallback.",
                    debug=self.debug,
                )
                title = ""

        # Try full LLM for summary if available
        if self.llm:
            try:
                payload = self.llm.summarize(
                    text=truncated,
                    start=start,
                    end=end,
                    global_context=self.global_context_summary,
                    context_before=context_before,
                    context_after=context_after,
                )
                if not title:
                    title = payload.get("title", "").strip()
                summary = payload.get("summary", "").strip()
            except Exception as exc:
                debug_print(
                    f"LLM summarisation failed ({exc}); using fallback heuristics.",
                    debug=self.debug,
                )

        if not title:
            title = self._fallback_title(truncated)
        if not summary:
            summary = self._fallback_summary(truncated, self.global_context_summary)

        return Chapter(
            title=title, start=start, end=end, summary=summary, full_text=text
        )

    def identify_chapters(
        self,
        segments: List[Segment],
        *,
        max_chapters: Optional[int] = None,
    ) -> List[Chapter]:
        if not segments:
            return []

        self._build_global_context(segments)
        texts = [segment.text for segment in segments]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        dissimilarity = self._compute_dissimilarity(embeddings)
        peaks = self._select_peaks(dissimilarity, segments, max_chapters)

        boundaries = [0] + [peak + 1 for peak in peaks] + [len(segments)]
        boundaries = sorted(set(boundaries))
        boundaries = self._merge_short_spans(boundaries, segments)

        chapters: List[Chapter] = []
        for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
            slice_segments = segments[start_idx:end_idx]
            if not slice_segments:
                continue

            full_text = " ".join(
                seg.text.strip() for seg in slice_segments if seg.text.strip()
            )
            if not full_text:
                continue

            start_time = slice_segments[0].start
            end_time = slice_segments[-1].end
            context_before, context_after = self._collect_neighbor_context(
                segments,
                start_idx,
                end_idx,
            )
            chapters.append(
                self._generate_chapter(
                    full_text,
                    start_time,
                    end_time,
                    context_before,
                    context_after,
                )
            )

        return chapters

    @staticmethod
    def format_timestamp_static(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        return f"{minutes:02d}:{secs:02d}.{millis:03d}"

    def export_chapters(self, chapters: List[Chapter], filepath: str) -> None:
        payload = {
            "summary": self.global_context_summary,
            "chapters": [
                {
                    "chapter_title": chapter.title,
                    "chapter_overview": chapter.summary,
                    "text": chapter.full_text,
                    "start_time": chapter.start,
                    "end_time": chapter.end,
                }
                for chapter in chapters
            ],
        }

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def print_chapters(self, chapters: List[Chapter]) -> None:
        for index, chapter in enumerate(chapters, start=1):
            start_ts = self.format_timestamp_static(chapter.start)
            end_ts = self.format_timestamp_static(chapter.end)
            print(f"[{index:02d}] {start_ts} -> {end_ts} | {chapter.title}")
            print(f"    {chapter.summary}\n")


def handle(
    input_file: str,
    output_folder: str,
    config: "ScenesConfig | None" = None,
    *,
    debug: bool = False,
):
    """Main entry point for chapter/scene detection.

    Args:
        input_file: Path to transcript JSON or working directory.
        output_folder: Path to output directory.
        config: ScenesConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Dict containing summary and chapters, or empty list on error.
    """
    return _chapters(input_file, output_folder, config=config, debug=debug)


def _chapters(
    path_or_working_dir: str,
    output_folder: Optional[str] = None,
    config: "ScenesConfig | None" = None,
    *,
    debug: bool = False,
):
    print("INFO: Generating chapter summaries")

    try:
        segments_path, working_dir = _resolve_segment_source(
            path_or_working_dir,
            output_folder,
            debug=debug,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return []

    # Build ChapterIdentifier with config values
    identifier_kwargs: Dict[str, Any] = {"debug": debug}
    if config:
        identifier_kwargs.update(
            {
                "embed_model": config.embed_model,
                "llm_model": config.llm_model,
                "title_model": config.title_model,
                "min_chapter_length": config.min_chapter_length,
                "boundary_percentile": config.boundary_percentile,
                "max_new_tokens": config.max_new_tokens,
                "max_ctx_chars": config.max_ctx_chars,
                "global_chunk_chars": config.global_chunk_chars,
                "context_snippet_chars": config.context_snippet_chars,
                "global_chunk_tokens": config.global_chunk_tokens,
                "global_summary_tokens": config.global_summary_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "enable_llm": config.enable_llm,
            }
        )

    try:
        identifier = ChapterIdentifier(**identifier_kwargs)
    except Exception as exc:
        print(f"ERROR: Unable to create ChapterIdentifier: {exc}")
        return []

    try:
        segments = identifier.load_transcript(str(segments_path))
    except FileNotFoundError:
        print(f"ERROR: Transcript not found: {segments_path}")
        return []
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return []
    except Exception as exc:
        print(f"ERROR: Failed to parse transcript: {exc}")
        return []

    if not segments:
        print("WARN: Transcript is empty; no chapters generated")
        return []

    try:
        chapter_objects = identifier.identify_chapters(segments)
    except Exception as exc:
        print(f"ERROR: Chapter identification failed: {exc}")
        return []

    chapter_results = [
        {
            "chapter_title": chapter.title,
            "chapter_overview": chapter.summary,
            "text": chapter.full_text,
            "start_time": chapter.start,
            "end_time": chapter.end,
        }
        for chapter in chapter_objects
    ]

    output_payload = {
        "summary": identifier.global_context_summary,
        "chapters": chapter_results,
    }

    scene_root = (
        Path(output_folder)
        if output_folder is not None
        else (working_dir if working_dir is not None else segments_path.parent)
    )
    scene_dir = scene_root / "chapters"
    scene_dir.mkdir(parents=True, exist_ok=True)
    output_path = scene_dir / "chapters.json"

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"ERROR: Failed to write chapters to {output_path}: {exc}")

    if debug:
        identifier.print_chapters(chapter_objects)

    return output_payload
