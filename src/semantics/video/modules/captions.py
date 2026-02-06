from __future__ import annotations

import json
import os
import shutil
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils.logging import debug_print, gray_debug_output
import torch

if TYPE_CHECKING:
    from config import CaptionsConfig

# Silence noisy warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, logging as hf_logging

# Keep transformers logs quiet
hf_logging.set_verbosity_error()

__all__ = ["handle"]

_FLORENCE_SINGLETON: dict[tuple[str, str, str], "Florence2Analyzer"] = {}
BATCH_SIZE = 8  # Safe default for VRAM usage (Florence-Large)


def handle(
    input_file: str,
    output_folder: str,
    config: "CaptionsConfig | None" = None,
    *,
    frame_indices: Optional[Iterable] = None,
    debug: bool = False,
):
    """Main entry point for caption extraction."""
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

    # Normalize and sort indices
    normalized_indices: List[int] = []
    seen: set[int] = set()
    source_iterable: Iterable[Any] = frame_indices if isinstance(frame_indices, Iterable) and not isinstance(frame_indices, (str, bytes)) else [frame_indices]

    for value in source_iterable:
        try:
            val = float(value)
            number = int(val)
            if number >= 0 and number not in seen:
                seen.add(number)
                normalized_indices.append(number)
        except (ValueError, TypeError):
            continue

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

    # Initialize analyzer
    with gray_debug_output(debug):
        analyzer = _get_analyzer(model_id=model_id, precision=precision, default_queries=default_queries)

    video_abs_path = os.path.abspath(video_file)
    
    # --- OPTIMIZATION: Process in Batches ---
    
    # 1. Open Video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file: {video_file}")
        return [], output_folder

    total_frames_count = len(normalized_indices)
    pbar_desc = "Captions"
    
    # Setup progress bar
    if debug:
        pbar = tqdm(total=total_frames_count, desc=pbar_desc, unit="frame", colour="#888888")
    else:
        pbar = nullcontext()

    try:
        with pbar as pb:
            # Chunk indices into batches
            for i in range(0, total_frames_count, BATCH_SIZE):
                batch_indices = normalized_indices[i : i + BATCH_SIZE]
                batch_images: List[np.ndarray] = []
                valid_batch_indices: List[int] = []

                # Smart Video Reading (avoid seeking if close)
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                for target_idx in batch_indices:
                    # If we need to go backward or skip a huge amount, use Seek
                    # If gap is small (< 64 frames), just grab() through them.
                    # grab() is much faster than set() because it doesn't clear decoder buffers.
                    gap = target_idx - current_pos
                    
                    if gap < 0 or gap > 64:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
                        current_pos = target_idx
                    elif gap > 0:
                        # Skip intermediate frames
                        for _ in range(int(gap)):
                            cap.grab()
                        current_pos = target_idx

                    ret, frame_img = cap.read()
                    current_pos += 1 # read() advances pos
                    
                    if ret:
                        batch_images.append(frame_img)
                        valid_batch_indices.append(target_idx)
                
                if not batch_images:
                    continue

                # Run Batch Inference
                with gray_debug_output(debug):
                    batch_results = _generate_captions_batch(
                        analyzer,
                        batch_images,
                        run_ocr=run_ocr,
                        run_objects=run_objects,
                        run_visual_grounding=run_visual_grounding,
                    )

                # Collect Results
                for idx, (frame_num, gen) in enumerate(zip(valid_batch_indices, batch_results)):
                    frame_reference = f"{video_abs_path}#frame_{int(frame_num):08d}"
                    frame_results.append({
                        "frame": int(frame_num),
                        "frame_path": frame_reference,
                        "caption": gen.get("caption", ""),
                        "caption_detailed": gen.get("caption_detailed", ""),
                        "caption_more_detailed": gen.get("caption_more_detailed", ""),
                        "detections": gen.get("detections", []),
                    })
                
                if debug:
                    pb.update(len(valid_batch_indices))

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
    """Lightweight wrapper to run Florence-2 tasks and parse results (Batched)."""

    def __init__(self, model_id: str = "microsoft/Florence-2-large-ft", precision: str = "fp16"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision == "fp16":
            if self.device == "cpu":
                self.dtype = getattr(torch, "bfloat16", torch.float32)
            else:
                self.dtype = torch.float16
        else:
            self.dtype = getattr(torch, "bfloat16", torch.float32)

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

        try:
            self.model.eval()
            if hasattr(self.model, "language_model") and self.model.language_model is not None:
                self.model.language_model.eval()
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.default_queries = "person. car. dog. cat. bicycle. chair. book. phone. text."

        # Configure generation settings once
        self._setup_generation_config()

    def _setup_generation_config(self):
        try:
            if hasattr(self.model, "config"):
                self.model.config.use_cache = True
            
            gen_cfg = getattr(self.model, "generation_config", None)
            tokenizer = getattr(self.processor, "tokenizer", None)
            
            if gen_cfg is not None:
                gen_cfg.do_sample = False
                gen_cfg.num_beams = 1
                if tokenizer:
                    if getattr(gen_cfg, "pad_token_id", None) is None: gen_cfg.pad_token_id = tokenizer.pad_token_id
                    if getattr(gen_cfg, "eos_token_id", None) is None: gen_cfg.eos_token_id = tokenizer.eos_token_id
                    if getattr(gen_cfg, "bos_token_id", None) is None: gen_cfg.bos_token_id = tokenizer.bos_token_id
        except Exception:
            pass

    def run_task_batch(self, images: List[Image.Image], task_prompt: str, text_input: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run a task on a batch of images."""
        if not images:
            return []

        # Prepare prompts: One per image
        prompt_str = task_prompt if text_input is None else task_prompt + text_input
        prompts = [prompt_str] * len(images)

        # Batch tokenization and image processing
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Dynamic max tokens based on task
        is_ocr_task = task_prompt in ("<OCR>", "<OCR_WITH_REGION>")
        max_new = 256 if is_ocr_task else 512

        # Configure generation
        gc = getattr(self.model, "generation_config", GenerationConfig())
        gen_cfg = GenerationConfig(
            do_sample=False,
            num_beams=1,
            use_cache=False, # Cache not needed for greedy, saves VRAM
            max_new_tokens=max_new,
            eos_token_id=gc.eos_token_id,
            pad_token_id=gc.pad_token_id,
            bos_token_id=gc.bos_token_id,
        )
        
        # Suppress warnings/fix config
        if hasattr(gen_cfg, "early_stopping"): gen_cfg.early_stopping = False
        if gc.forced_bos_token_id is not None: gen_cfg.forced_bos_token_id = gc.forced_bos_token_id
        if gc.forced_eos_token_id is not None: gen_cfg.forced_eos_token_id = gc.forced_eos_token_id
        
        # Batch Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
            )

        # Decode all at once
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        cleaned_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Post-process individually (CPU bound, fast)
        results = []
        for i, (gen_text, clean_text) in enumerate(zip(generated_texts, cleaned_texts)):
            parsed = self.processor.post_process_generation(
                gen_text,
                task=task_prompt,
                image_size=images[i].size,
            )
            results.append({"parsed": parsed, "raw": gen_text, "clean": clean_text})
        
        return results

    # ---- Parsers (Static methods unchanged) ----
    @staticmethod
    def _flatten_points(box: Any) -> Optional[List[float]]:
        if isinstance(box, (list, tuple)):
            if len(box) >= 4 and all(isinstance(v, (int, float)) for v in box[:4]):
                return [float(v) for v in box]
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
                data.get("bboxes") or data.get("boxes") or data.get("polygons") or data.get("quad_boxes")
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


def _generate_captions_batch(
    analyzer: Florence2Analyzer,
    images: List[np.ndarray], # BGR numpy arrays
    *,
    run_ocr: bool = False,
    run_objects: bool = False,
    run_visual_grounding: bool = False,
) -> List[Dict[str, Any]]:
    """Generate captions and detections for a BATCH of images."""
    if not images:
        return []

    # Batch Convert OpenCV BGR to PIL RGB
    pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images]
    
    # Run Batch Tasks
    # 1. Base Captions
    res_caption = analyzer.run_task_batch(pil_images, "<CAPTION>")
    res_caption_det = analyzer.run_task_batch(pil_images, "<DETAILED_CAPTION>")
    res_caption_more = analyzer.run_task_batch(pil_images, "<MORE_DETAILED_CAPTION>")
    
    # 2. Object Detection
    res_od = [None] * len(images)
    if run_objects:
        res_od = analyzer.run_task_batch(pil_images, "<OD>")

    # 3. OCR (Logic preserved: Try Region -> Fallback to Simple)
    ocr_results = [[] for _ in range(len(images))]
    if run_ocr:
        # First attempt: OCR with Region
        res_ocr_region = analyzer.run_task_batch(pil_images, "<OCR_WITH_REGION>")
        
        # Check which ones failed (empty) and might need fallback
        indices_needing_fallback = []
        fallback_images = []
        
        for idx, res in enumerate(res_ocr_region):
            entries = Florence2Analyzer.extract_ocr(res)
            if entries:
                ocr_results[idx] = entries
            else:
                indices_needing_fallback.append(idx)
                fallback_images.append(pil_images[idx])
        
        # Fallback attempt: Simple OCR
        if fallback_images:
            res_ocr_simple = analyzer.run_task_batch(fallback_images, "<OCR>")
            
            for i, res_simple in enumerate(res_ocr_simple):
                original_idx = indices_needing_fallback[i]
                entries = Florence2Analyzer.extract_ocr(res_simple)
                
                # Synthesize box if text exists but no region
                if not entries:
                    txt = Florence2Analyzer.extract_caption(res_simple)
                    if txt:
                        w, h = fallback_images[i].size
                        entries = [{
                            "task": "ocr_with_region",
                            "text": txt,
                            "bounding_box": [0.0, 0.0, float(max(0, w - 1)), float(max(0, h - 1))],
                        }]
                
                if entries:
                    ocr_results[original_idx] = entries

    # 4. Visual Grounding
    res_vg = [None] * len(images)
    if run_visual_grounding:
        res_vg = analyzer.run_task_batch(pil_images, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=analyzer.default_queries)

    # Compile results
    batch_out = []
    for i in range(len(images)):
        detections = []
        if res_od[i]:
            detections.extend(Florence2Analyzer.extract_od(res_od[i]))
        if ocr_results[i]:
            detections.extend(ocr_results[i])
        if res_vg[i]:
            detections.extend(Florence2Analyzer.extract_phrase_grounding(res_vg[i]))

        batch_out.append({
            "caption": Florence2Analyzer.extract_caption(res_caption[i]),
            "caption_detailed": Florence2Analyzer.extract_caption(res_caption_det[i]),
            "caption_more_detailed": Florence2Analyzer.extract_caption(res_caption_more[i]),
            "detections": detections,
        })
        
    return batch_out