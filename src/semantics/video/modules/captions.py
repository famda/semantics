from __future__ import annotations

import json
import os
import shutil
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import cv2
from PIL import Image
from tqdm import tqdm

from .utils.logging import debug_print, gray_debug_output
import torch

if TYPE_CHECKING:
    from config import CaptionsConfig

# Silence noisy warnings from dependencies as early as possible (before transformers import)
warnings.filterwarnings("ignore", category=SyntaxWarning)

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, logging as hf_logging

# Keep transformers logs quiet
hf_logging.set_verbosity_error()

__all__ = ["handle"]

_FLORENCE_SINGLETON: dict[tuple[str, str, str], "Florence2Analyzer"] = {}


def handle(
    input_file: str,
    output_folder: str,
    config: "CaptionsConfig | None" = None,
    *,
    frame_indices: Optional[Iterable] = None,
    debug: bool = False,
):
    """Main entry point for caption extraction.

    Args:
        input_file: Path to input video file.
        output_folder: Path to output directory.
        config: CaptionsConfig instance or None for defaults.
        frame_indices: List of frame indices to process.
        debug: Enable verbose debug output.

    Returns:
        Tuple of (frame_results, captions_folder).
    """
    return _extract_captions(
        input_file,
        frame_indices or [],
        output_folder,
        model_id=config.model_id if config else "microsoft/Florence-2-large-ft",
        precision=config.precision if config else "fp16",
        default_queries=config.default_queries if config else "person. car. dog. cat. bicycle. chair. book. phone. text.",
        run_ocr=config.run_ocr if config else False,
        run_objects=config.run_objects if config else False,
        run_visual_grounding=config.run_visual_grounding if config else False,
        debug=debug,
    )


def _extract_captions(
    video_file,
    frame_indices,
    output_folder,
    *,
    model_id: str = "microsoft/Florence-2-large-ft",
    precision: str = "fp16",
    default_queries: str = "person. car. dog. cat. bicycle. chair. book. phone. text.",
    run_ocr: bool = False,
    run_objects: bool = False,
    run_visual_grounding: bool = False,
    debug: bool = False,
):
    output_folder = os.path.join(output_folder, "captions")

    normalized_indices: List[int] = []
    seen: set[int] = set()
    source_iterable: Iterable[Any]
    if isinstance(frame_indices, Iterable) and not isinstance(frame_indices, (str, bytes)):
        source_iterable = frame_indices
    else:
        source_iterable = [frame_indices]

    for value in source_iterable:
        number = None
        if isinstance(value, (int, float)):
            try:
                number = int(value)
            except Exception:
                number = None
        else:
            try:
                number = int(float(value))
            except Exception:
                number = None

        if number is None:
            continue
        if number < 0 or number in seen:
            continue
        seen.add(number)
        normalized_indices.append(number)

    normalized_indices.sort()

    if not normalized_indices:
        print("ERROR: No frame indices provided for caption generation")
        return [], output_folder

    print(f"INFO: Extracting captions for {len(normalized_indices)} frame(s)")
    debug_print(f"Frame indices: {normalized_indices}", debug=debug)

    if os.path.isdir(output_folder):
        print("INFO: Cleaning existing captions folder")
        try:
            shutil.rmtree(output_folder)
        except Exception as e:
            print(f"Warning: Failed to clean captions folder: {e}")

    os.makedirs(output_folder, exist_ok=True)
    json_path = os.path.join(output_folder, "captions.json")
    frame_results = []

    # Pre-initialize analyzer once (avoids downloading mid-loop)
    with gray_debug_output(debug):
        _ = _get_analyzer(model_id=model_id, precision=precision, default_queries=default_queries)

    def _progress_iter(it: Iterable, desc: Optional[str] = None, unit: Optional[str] = None):
        if debug:
            kwargs = {}
            if desc is not None:
                kwargs["desc"] = desc
            if unit is not None:
                kwargs["unit"] = unit
            try:
                iterable = tqdm(it, colour="#888888", **kwargs)
            except TypeError:
                iterable = tqdm(it, **kwargs)

            @contextmanager
            def _ctx():
                with gray_debug_output(True):
                    try:
                        yield
                    finally:
                        close_fn = getattr(iterable, "close", None)
                        if callable(close_fn):
                            close_fn()

            return iterable, _ctx()
        return it, nullcontext()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file: {video_file}")
        return [], output_folder

    video_abs_path = os.path.abspath(video_file)

    iterable, progress_ctx = _progress_iter(normalized_indices, desc="Captions", unit="frame")

    try:
        with progress_ctx:
            for frame_number in iterable:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))
                ret, frame_img = cap.read()
                if not ret:
                    continue

                with gray_debug_output(debug):
                    gen = _generate_captions(
                        frame_img,
                        model_id=model_id,
                        precision=precision,
                        default_queries=default_queries,
                        run_ocr=run_ocr,
                        run_objects=run_objects,
                        run_visual_grounding=run_visual_grounding,
                    )

                frame_reference = f"{video_abs_path}#frame_{int(frame_number):08d}"

                frame_results.append(
                    {
                        "frame": int(frame_number),
                        "frame_path": frame_reference,
                        "caption": gen.get("caption", ""),
                        "caption_detailed": gen.get("caption_detailed", ""),
                        "caption_more_detailed": gen.get("caption_more_detailed", ""),
                        "detections": gen.get("detections", []),
                    }
                )
    finally:
        cap.release()

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(frame_results, f, ensure_ascii=False, indent=4)
        debug_print(f"INFO: Saved captions report to {json_path}", debug=debug)
    except Exception as e:
        print(f"Warning: Failed to save captions report: {e}")

    return frame_results, output_folder

class Florence2Analyzer:
    """Lightweight wrapper to run Florence-2 tasks and parse results."""

    def __init__(self, model_id: str = "microsoft/Florence-2-large-ft", precision: str = "fp16"):
        # Device and dtype selection mirroring backup logic
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision == "fp16":
            # Prefer bfloat16 on CPU; float16 works well on CUDA
            if self.device == "cpu":
                self.dtype = getattr(torch, "bfloat16", torch.float32)
            else:
                self.dtype = torch.float16
        else:
            self.dtype = getattr(torch, "bfloat16", torch.float32)

        # Load model (try torch_dtype first, fall back to dtype for older transformers)
        # Force eager attention to avoid SDPA attribute checks on some builds
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="eager",
                dtype=self.dtype,
            ).to(self.device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.device)

        # Eval mode for faster inference
        try:
            self.model.eval()
            if hasattr(self.model, "language_model") and self.model.language_model is not None:
                self.model.language_model.eval()
        except Exception:
            pass

        # Load processor (simple load as in original script)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Default queries for phrase grounding/open-vocabulary
        self.default_queries = "person. car. dog. cat. bicycle. chair. book. phone. text."

        # Align model generation behavior with backup (greedy, no beams)
        try:
            if hasattr(self.model, "config"):
                # Enable cache at config level; some wrappers rely on this
                self.model.config.use_cache = True
            if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "config"):
                self.model.language_model.config.use_cache = True

            gen_cfg = getattr(self.model, "generation_config", None)
            tokenizer = getattr(self.processor, "tokenizer", None)
            if gen_cfg is not None:
                gen_cfg.do_sample = False
                gen_cfg.num_beams = 1
                if tokenizer is not None:
                    if getattr(gen_cfg, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
                        gen_cfg.pad_token_id = tokenizer.pad_token_id
                    if getattr(gen_cfg, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                        gen_cfg.eos_token_id = tokenizer.eos_token_id
                    if getattr(gen_cfg, "bos_token_id", None) is None and getattr(tokenizer, "bos_token_id", None) is not None:
                        gen_cfg.bos_token_id = tokenizer.bos_token_id

            lm = getattr(self.model, "language_model", None)
            lm_gen_cfg = getattr(lm, "generation_config", None)
            if lm_gen_cfg is not None:
                lm_gen_cfg.do_sample = False
                lm_gen_cfg.num_beams = 1
        except Exception:
            # Non-fatal if config knobs are not available on specific builds
            pass

    def run_task(self, image: Image.Image, task_prompt: str, text_input: Optional[str] = None) -> Dict[str, Any]:
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Greedy generation config (match backup, avoid beam search and SDPA cache dependency)
        gc = getattr(self.model, "generation_config", GenerationConfig())
        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token_id = getattr(gc, "eos_token_id", None) or (getattr(tokenizer, "eos_token_id", None) if tokenizer else None)
        pad_token_id = getattr(gc, "pad_token_id", None) or (getattr(tokenizer, "pad_token_id", None) if tokenizer else None)
        bos_token_id = getattr(gc, "bos_token_id", None) or (getattr(tokenizer, "bos_token_id", None) if tokenizer else None)
        forced_bos_token_id = getattr(gc, "forced_bos_token_id", None)
        forced_eos_token_id = getattr(gc, "forced_eos_token_id", None)
        decoder_start_token_id = getattr(gc, "decoder_start_token_id", None)
        no_repeat_ngram_size = getattr(gc, "no_repeat_ngram_size", None)

        # Tune max_new_tokens per task to keep OCR fast while preserving caption quality
        is_ocr_task = task_prompt in ("<OCR>", "<OCR_WITH_REGION>")
        max_new = 256 if is_ocr_task else 512

        gen_cfg = GenerationConfig(
            do_sample=False,
            num_beams=1,
            use_cache=False,
            max_new_tokens=max_new,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
        )
        # Ensure a valid value to satisfy transformers validation
        if hasattr(gen_cfg, "early_stopping"):
            try:
                gen_cfg.early_stopping = False
            except Exception:
                pass
        if forced_bos_token_id is not None:
            gen_cfg.forced_bos_token_id = forced_bos_token_id
        if forced_eos_token_id is not None:
            gen_cfg.forced_eos_token_id = forced_eos_token_id
        if decoder_start_token_id is not None:
            gen_cfg.decoder_start_token_id = decoder_start_token_id
        if no_repeat_ngram_size is not None:
            gen_cfg.no_repeat_ngram_size = no_repeat_ngram_size

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        cleaned_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image.size,
        )
        return {"parsed": parsed, "raw": generated_text, "clean": cleaned_text}

    # ---- Parsers (adapted from backup) ----
    @staticmethod
    def _flatten_points(box: Any) -> Optional[List[float]]:
        # Accept shapes:
        # - [x1,y1,x2,y2]
        # - [x1,y1, x2,y2, x3,y3, x4,y4]
        # - [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        if isinstance(box, (list, tuple)):
            if len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
                return [float(v) for v in box]
            # nested points
            if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in box):
                flat: List[float] = []
                for p in box:
                    try:
                        flat.extend([float(p[0]), float(p[1])])
                    except Exception:
                        return None
                return flat if len(flat) >= 4 else None
        return None

    @staticmethod
    def _to_rect_box(box: Any) -> Optional[List[float]]:
        flat = Florence2Analyzer._flatten_points(box)
        if not flat or len(flat) < 4:
            return None
        xs = flat[0::2]
        ys = flat[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _is_valid_box(box: Any) -> bool:
        return Florence2Analyzer._to_rect_box(box) is not None

    @staticmethod
    def _unwrap_task_data(data: Any) -> Any:
        if isinstance(data, dict):
            keys = [k for k in data.keys() if isinstance(k, str) and k.startswith('<') and k.endswith('>')]
            if len(keys) == 1:
                return data.get(keys[0])
        return data

    @staticmethod
    def extract_caption(result: Dict[str, Any]) -> str:
        if isinstance(result, dict):
            clean = result.get("clean")
            if isinstance(clean, str) and clean.strip():
                return clean.strip()
        data = result.get("parsed") if isinstance(result, dict) else result
        data = Florence2Analyzer._unwrap_task_data(data)
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            for key in ("caption", "detailed_caption", "more_detailed_caption", "text"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            for key in ("caption", "text"):
                val = data.get(key)
                if isinstance(val, list) and val:
                    return str(val[0]).strip()
        raw = result.get("raw") if isinstance(result, dict) else None
        return raw.strip() if isinstance(raw, str) else ""

    @staticmethod
    def extract_od(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = result.get("parsed") if isinstance(result, dict) else result
        data = Florence2Analyzer._unwrap_task_data(data)
        out: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            labels = data.get("labels") or data.get("classes") or data.get("class_names") or data.get("captions")
            bboxes = data.get("bboxes") or data.get("boxes") or data.get("bbox")
            if isinstance(labels, list) and isinstance(bboxes, list) and len(labels) == len(bboxes):
                for lab, box in zip(labels, bboxes):
                    if Florence2Analyzer._is_valid_box(box):
                        out.append({"task": "object_detection", "label": str(lab), "bounding_box": list(box)})
            elif isinstance(data.get("objects"), list):
                for o in data["objects"]:
                    lab = o.get("label") or o.get("class") or o.get("caption")
                    box = o.get("bbox") or o.get("bounding_box") or o.get("box")
                    if lab is not None and Florence2Analyzer._is_valid_box(box):
                        out.append({"task": "object_detection", "label": str(lab), "bounding_box": list(box)})
            elif isinstance(bboxes, list) and not labels:
                for box in bboxes:
                    if Florence2Analyzer._is_valid_box(box):
                        out.append({"task": "object_detection", "label": "", "bounding_box": list(box)})
        return out

    @staticmethod
    def extract_ocr(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = result.get("parsed") if isinstance(result, dict) else result
        data = Florence2Analyzer._unwrap_task_data(data)
        entries: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            texts = data.get("text") or data.get("texts") or data.get("labels") or data.get("words")
            boxes = (
                data.get("bboxes")
                or data.get("boxes")
                or data.get("polygons")
                or data.get("quad_boxes")
                or data.get("quad_bboxes")
            )
            if isinstance(texts, list) and isinstance(boxes, list):
                for t, b in zip(texts, boxes):
                    rect = Florence2Analyzer._to_rect_box(b)
                    if isinstance(t, str) and rect is not None:
                        entries.append({"task": "ocr_with_region", "text": t, "bounding_box": rect})
            elif isinstance(texts, str):
                rect = Florence2Analyzer._to_rect_box(boxes)
                if rect is not None:
                    entries.append({"task": "ocr_with_region", "text": texts, "bounding_box": rect})
            if isinstance(data.get("regions"), list):
                for r in data["regions"]:
                    txt = r.get("text")
                    b = r.get("bbox") or r.get("bounding_box")
                    rect = Florence2Analyzer._to_rect_box(b)
                    if isinstance(txt, str) and rect is not None:
                        entries.append({"task": "ocr_with_region", "text": txt, "bounding_box": rect})
        # Also handle list of region dicts directly
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict):
                    txt = r.get("text")
                    b = r.get("bbox") or r.get("bounding_box") or r.get("box")
                    rect = Florence2Analyzer._to_rect_box(b)
                    if isinstance(txt, str) and rect is not None:
                        entries.append({"task": "ocr_with_region", "text": txt, "bounding_box": rect})
        return [e for e in entries if e.get("text")]

    @staticmethod
    def extract_phrase_grounding(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = result.get("parsed") if isinstance(result, dict) else result
        data = Florence2Analyzer._unwrap_task_data(data)
        out: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            labels = data.get("labels") or data.get("phrases") or data.get("text") or data.get("queries")
            bboxes = data.get("bboxes") or data.get("boxes") or data.get("bbox")
            if isinstance(labels, list) and isinstance(bboxes, list):
                for lab, box in zip(labels, bboxes):
                    if Florence2Analyzer._is_valid_box(box):
                        out.append({"task": "visual_grounding", "label": str(lab), "bounding_box": list(box)})
            elif isinstance(data.get("objects"), list):
                for o in data["objects"]:
                    lab = o.get("label") or o.get("phrase") or o.get("text")
                    box = o.get("bbox") or o.get("bounding_box") or o.get("box")
                    if lab is not None and Florence2Analyzer._is_valid_box(box):
                        out.append({"task": "visual_grounding", "label": str(lab), "bounding_box": list(box)})
        return out


def _get_analyzer(
    *,
    model_id: str = "microsoft/Florence-2-large-ft",
    precision: str = "fp16",
    default_queries: str = "person. car. dog. cat. bicycle. chair. book. phone. text.",
) -> Florence2Analyzer:
    key = (model_id, precision, default_queries)
    if key not in _FLORENCE_SINGLETON:
        analyzer = Florence2Analyzer(model_id=model_id, precision=precision)
        analyzer.default_queries = default_queries
        _FLORENCE_SINGLETON[key] = analyzer
    return _FLORENCE_SINGLETON[key]


def _generate_captions(
    frame_img,
    *,
    model_id: str = "microsoft/Florence-2-large-ft",
    precision: str = "fp16",
    default_queries: str = "person. car. dog. cat. bicycle. chair. book. phone. text.",
    run_ocr: bool = False,
    run_objects: bool = False,
    run_visual_grounding: bool = False,
) -> Dict[str, Any]:
    """Internal: Generate captions and detections for a given frame image (numpy array, BGR)."""
    if frame_img is None:
        return {"caption": "", "caption_detailed": "", "caption_more_detailed": "", "detections": []}

    # Convert OpenCV BGR to PIL RGB
    rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)

    try:
        analyzer = _get_analyzer(model_id=model_id, precision=precision, default_queries=default_queries)
    except Exception:
        # Graceful fallback when model isn't available at runtime
        return {"caption": "", "caption_detailed": "", "caption_more_detailed": "", "detections": []}

    # Core caption tasks (respecting backup style prompts)
    res_caption = analyzer.run_task(image_pil, "<CAPTION>")
    res_caption_det = analyzer.run_task(image_pil, "<DETAILED_CAPTION>")
    res_caption_more = analyzer.run_task(image_pil, "<MORE_DETAILED_CAPTION>")

    # Detection-oriented tasks
    res_od = None
    if run_objects:
        res_od = analyzer.run_task(image_pil, "<OD>")

    # OCR with region; fallback to <OCR> when regions are not returned
    ocr_entries: List[Dict[str, Any]] = []
    if run_ocr:
        res_ocr = analyzer.run_task(image_pil, "<OCR_WITH_REGION>")
        ocr_entries = Florence2Analyzer.extract_ocr(res_ocr)
        if not ocr_entries:
            res_ocr_simple = analyzer.run_task(image_pil, "<OCR>")
            ocr_entries = Florence2Analyzer.extract_ocr(res_ocr_simple)
            # If simple OCR returns only text without regions, synthesize a full-image box
            if not ocr_entries:
                # Try to extract plain text from the OCR responses
                simple_txt = Florence2Analyzer.extract_caption(res_ocr_simple)
                if not isinstance(simple_txt, str) or not simple_txt.strip():
                    simple_txt = Florence2Analyzer.extract_caption(res_ocr)
                if isinstance(simple_txt, str) and simple_txt.strip():
                    w, h = image_pil.size
                    ocr_entries = [{
                        "task": "ocr_with_region",
                        "text": simple_txt.strip(),
                        "bounding_box": [0.0, 0.0, float(max(0, w - 1)), float(max(0, h - 1))],
                    }]

    # Visual grounding (phrase grounding) with default queries
    res_vg = None
    if run_visual_grounding:
        res_vg = analyzer.run_task(image_pil, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=analyzer.default_queries)

    caption = Florence2Analyzer.extract_caption(res_caption)
    caption_detailed = Florence2Analyzer.extract_caption(res_caption_det)
    caption_more_detailed = Florence2Analyzer.extract_caption(res_caption_more)

    detections: List[Dict[str, Any]] = []
    if res_od is not None:
        detections.extend(Florence2Analyzer.extract_od(res_od))
    if ocr_entries:
        detections.extend(ocr_entries)
    if res_vg is not None:
        detections.extend(Florence2Analyzer.extract_phrase_grounding(res_vg))

    return {
        "caption": caption,
        "caption_detailed": caption_detailed,
        "caption_more_detailed": caption_more_detailed,
        "detections": detections,
    }
