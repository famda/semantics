import json
import os
import shutil
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Iterable, List, Optional, Dict, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

try:
    import easyocr
except ImportError:  # optional dependency
    easyocr = None

from global_helpers import select_frame_indices
from .utils.logging import debug_print, gray_debug_output

if TYPE_CHECKING:
    from config import OcrConfig

__all__ = ["handle"]


# -----------------------------------------------------------------------------
# Singleton cache for EasyOCR Reader to avoid reloading the model on each call
# Key: (tuple of languages, use_gpu)
# Value: easyocr.Reader instance
# -----------------------------------------------------------------------------
_EASYOCR_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool], "easyocr.Reader"] = {}


def _get_easyocr_reader(lang_list: List[str], use_gpu: bool, debug: bool = False) -> "easyocr.Reader":
    """Get or create a cached EasyOCR Reader instance.
    
    Args:
        lang_list: List of language codes to support.
        use_gpu: Whether to use GPU acceleration.
        debug: Whether to show verbose output during initialization.
        
    Returns:
        A cached or newly created EasyOCR Reader.
    """
    cache_key = (tuple(sorted(lang_list)), use_gpu)
    if cache_key not in _EASYOCR_READER_CACHE:
        with gray_debug_output(debug):
            _EASYOCR_READER_CACHE[cache_key] = easyocr.Reader(lang_list, gpu=use_gpu)
    return _EASYOCR_READER_CACHE[cache_key]


def _extract_text(
    video_file,
    output_folder,
    frames_json,
    confidence_threshold: float = 70,
    fps: Optional[float] = None,
    save_images: bool = True,
    lang: str = 'en',
    debug: bool = False,
):

    print("INFO: Extracting OCR data from video")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device: {device}")

    output_folder = os.path.join(output_folder, "ocr")
    if os.path.isdir(output_folder):
        print("INFO: Cleaning existing OCR folder")
        try:
            shutil.rmtree(output_folder)
        except Exception as exc:
            print(f"Warning: Failed to clean OCR folder: {exc}")
    os.makedirs(output_folder, exist_ok=True)
    json_results_path = os.path.join(output_folder, "ocr.json")

    try:
        with open(frames_json, "r", encoding="utf-8") as metadata_file:
            frames_payload = json.load(metadata_file)
    except FileNotFoundError:
        print(f"ERROR: Frames metadata file not found at {frames_json}")
        return []
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse frames metadata ({exc})")
        return []
    except Exception as exc:
        print(f"ERROR: Unable to load frames metadata: {exc}")
        return []

    frames_metadata = frames_payload.get("frames", []) or []
    fps_arg = fps if (isinstance(fps, (int, float)) and fps > 0) else None
    selected_indices = select_frame_indices(frames_metadata, fps_arg)
    debug_print(f"Selected {len(selected_indices)} frames for OCR", debug=debug)

    if not selected_indices:
        print("ERROR: No frames available for OCR")
        return []

    if easyocr is None:
        print("ERROR: The easyocr package is not installed; skipping OCR module.")
        return []

    lang_list = [entry.strip() for entry in lang.split(',') if entry.strip()]
    if not lang_list:
        lang_list = ['en']

    # Use cached reader to avoid reloading the model on each call
    reader = _get_easyocr_reader(lang_list, use_gpu=device.type == "cuda", debug=debug)

    box_annotator = label_annotator = None
    if save_images:
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_scale=0.6,
            text_thickness=1,
            text_padding=2,
        )

    def _progress_iter(it: Iterable, desc: Optional[str] = None, unit: Optional[str] = None):
        if debug:
            kwargs = {}
            if desc is not None:
                kwargs["desc"] = desc
            if unit is not None:
                kwargs["unit"] = unit
            try:
                iterator = tqdm(it, colour="#888888", **kwargs)
            except TypeError:
                iterator = tqdm(it, **kwargs)

            @contextmanager
            def _ctx():
                with gray_debug_output(True):
                    try:
                        yield
                    finally:
                        close_fn = getattr(iterator, "close", None)
                        if callable(close_fn):
                            close_fn()

            return iterator, _ctx()
        return it, nullcontext()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file: {video_file}")
        return []

    extracted_data = []
    video_abs_path = os.path.abspath(video_file)
    iterable, progress_ctx = _progress_iter(selected_indices, desc="OCR", unit="frame")

    try:
        with progress_ctx:
            for frame_index in iterable:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
                ok, frame_img = cap.read()
                if not ok or frame_img is None:
                    continue

                full_text, high_conf_text, avg_high_conf, text_details_list = _extract_text_from_frame(
                    reader,
                    frame_img,
                    confidence_threshold=confidence_threshold,
                )

                if avg_high_conf is None or avg_high_conf < confidence_threshold:
                    continue

                frame_reference = f"{video_abs_path}#frame_{int(frame_index):08d}"
                entry = {
                    "frame": int(frame_index),
                    "frame_path": frame_reference,
                    "full_text_ocr": full_text,
                    "high_confidence_text": high_conf_text,
                    "high_confidence_text_avg_conf": avg_high_conf,
                    "text_details": text_details_list,
                }
                extracted_data.append(entry)

                if save_images and text_details_list and box_annotator and label_annotator:
                    xyxy = []
                    confidences = []
                    labels = []
                    class_ids = []

                    for detail in text_details_list:
                        x_min = detail['bounding_box']['x1']
                        y_min = detail['bounding_box']['y1']
                        x_max = detail['bounding_box']['x2']
                        y_max = detail['bounding_box']['y2']
                        xyxy.append([x_min, y_min, x_max, y_max])
                        confidences.append(detail['confidence'] / 100.0)
                        labels.append(f"{detail['text']} {detail['confidence']:.1f}%")
                        class_ids.append(0)

                    detections = sv.Detections(
                        xyxy=np.array(xyxy),
                        confidence=np.array(confidences),
                        class_id=np.array(class_ids),
                    )

                    annotated_frame = frame_img.copy()
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    output_filename = f"{int(frame_index):08d}.png"
                    output_image_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_image_path, annotated_frame)
    finally:
        cap.release()

    if extracted_data:
        try:
            with open(json_results_path, 'w', encoding='utf-8') as json_file:
                json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)
            debug_print(f"INFO: Saved OCR report to {json_results_path}", debug=debug)
        except Exception as exc:
            print(f"Warning: Failed to save OCR report: {exc}")

    return extracted_data


def _extract_text_from_frame(reader, frame, confidence_threshold=80):
    """Internal: Extract text from a single frame."""

    if frame is None:
        return "", "", None, []

    conf_thresh_float = confidence_threshold / 100.0
    results = reader.readtext(frame)

    full_text_parts = []
    high_confidence_parts = []
    text_details = []
    high_conf_scores = [] # Store scores in 0-100 format

    for detection in results:
        # Handle potential differences in EasyOCR output format slightly
        if len(detection) == 3:
            bbox, text, conf = detection
        elif len(detection) == 2: # Sometimes simple list output
                bbox, text = detection
                conf = 0.0 # Default confidence if not provided
                print(f"Warning: OCR result for '{text}' missing confidence.")
        else:
            print(f"Warning: Skipping unexpected OCR result format: {detection}")
            continue

        conf_percent = conf * 100.0
        text = text.strip()

        if text:
            full_text_parts.append(text)
            # Ensure bbox is a list/numpy array of points
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(p, list) for p in bbox):
                points = np.array(bbox, dtype=np.int32)
            else:
                print(f"Warning: Skipping detection '{text}' due to unexpected bbox format: {bbox}")
                continue

            x_coords = points[:, 0]
            y_coords = points[:, 1]
            x1 = int(np.min(x_coords))
            y1 = int(np.min(y_coords))
            x2 = int(np.max(x_coords))
            y2 = int(np.max(y_coords))
            text_details.append({"text": text, "confidence": round(conf_percent, 2), "bounding_box": {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}})

            if conf_percent >= confidence_threshold:
                high_confidence_parts.append(text)
                high_conf_scores.append(conf_percent)

    full_extracted_text = " ".join(full_text_parts).strip()
    high_confidence_text = " ".join(high_confidence_parts).strip()

    avg_high_conf = None
    if high_conf_scores:
        avg_high_conf = round(sum(high_conf_scores) / len(high_conf_scores), 2)

    return full_extracted_text, high_confidence_text, avg_high_conf, text_details


# ---------------------------------------------------------------------------
# Public entry point (follows CLI module pattern)
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "OcrConfig | None" = None,
    *,
    frame_indices: Optional[List[int]] = None,
    debug: bool = False,
) -> Optional[str]:
    """Unified OCR entry point for the video CLI.

    Parameters
    ----------
    input_file:
        Path to the video file.
    output_folder:
        Destination directory for OCR results.
    config:
        Optional pydantic OcrConfig with parameters like confidence_threshold, lang, etc.
    frame_indices:
        Specific frame indexes to process.  If *None*, all frames from
        ``segments.json`` are processed.
    debug:
        When *True*, extra diagnostic output is printed.

    Returns
    -------
    str | None
        Path to the generated ``ocr.json`` file on success, *None* otherwise.
    """
    # Pull values from config or use defaults
    confidence = config.confidence_threshold if config else 70.0
    language = config.lang if config else "en"
    save_images = config.save_images if config else True

    # Build the path to the segments.json (expected by extract_text)
    frames_json = os.path.join(output_folder, "frames", "segments", "segments.json")
    if not os.path.isfile(frames_json):
        print(f"ERROR: segments.json not found at {frames_json}; run --from-segments first")
        return None

    result_data = _extract_text(
        video_file=input_file,
        output_folder=output_folder,
        frames_json=frames_json,
        confidence_threshold=confidence,
        fps=None,
        save_images=save_images,
        lang=language,
        debug=debug,
    )

    if result_data:
        ocr_json_path = os.path.join(output_folder, "ocr", "ocr.json")
        print(f"INFO: Saved OCR results to {ocr_json_path}")
        return ocr_json_path
    return None