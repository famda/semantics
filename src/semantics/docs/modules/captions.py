"""Image captioning using BLIP and Qwen3-VL.

Reads the images manifest produced by the images module and generates
captions for each extracted image:
- ``caption`` and ``caption_detailed`` via Salesforce BLIP (fast, basic)
- ``caption_very_detailed`` via Qwen3-VL (slower, rich descriptions)
- ``entities`` via NER on the detailed caption (optional)

Quality presets control which phases run:
- ``speed``    – BLIP only (~0.5 s / image)
- ``balanced`` – BLIP + Qwen3-VL 256 tokens (~12 s / image)
- ``quality``  – BLIP + Qwen3-VL 512 tokens (~21 s / image) **default**

Updates images.json with the new fields.
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch
from PIL import Image

from .utils.chunking import extract_document_context
from .utils.logging import debug_print, gray_debug_output, info_print

if TYPE_CHECKING:
    from ..config import CaptionsConfig

__all__ = ["handle"]

warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

# Quality preset → VLM max_new_tokens mapping (0 = skip VLM)
_QUALITY_TOKENS = {"speed": 0, "balanced": 256, "quality": 512}

# ---------------------------------------------------------------------------
# BLIP (basic captions)
# ---------------------------------------------------------------------------

_BLIP_MODEL = None
_BLIP_PROCESSOR = None


def _load_blip(model_id: str, precision: str):
    global _BLIP_MODEL, _BLIP_PROCESSOR
    if _BLIP_MODEL is not None:
        return _BLIP_MODEL, _BLIP_PROCESSOR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and precision == "fp16" else torch.float32

    _BLIP_PROCESSOR = BlipProcessor.from_pretrained(model_id)
    _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    _BLIP_MODEL.eval()
    return _BLIP_MODEL, _BLIP_PROCESSOR


def _unload_blip():
    global _BLIP_MODEL, _BLIP_PROCESSOR
    del _BLIP_MODEL, _BLIP_PROCESSOR
    _BLIP_MODEL = None
    _BLIP_PROCESSOR = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _caption_blip(
    model, processor, image: Image.Image, *, text_prompt: str | None = None, max_tokens: int = 100,
) -> str:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if text_prompt:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device, dtype)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device, dtype)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens)

    return processor.decode(out[0], skip_special_tokens=True).strip()


def _caption_blip_batch(
    model, processor, images: list[Image.Image],
    *, text_prompt: str | None = None, max_tokens: int = 100,
) -> list[str]:
    """Batch-caption multiple images in a single forward pass."""
    if not images:
        return []

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if text_prompt:
        texts = [text_prompt] * len(images)
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(device, dtype)
    else:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device, dtype)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_tokens)

    return [processor.decode(o, skip_special_tokens=True).strip() for o in out]


# ---------------------------------------------------------------------------
# Qwen3-VL (very detailed captions) — uses shared VLM from utils/vlm.py
# ---------------------------------------------------------------------------

_VLM_PROMPT_WITH_CONTEXT = (
    "You are captioning an image extracted from a document about: {context}\n\n"
    "Describe what is shown in this image. Include all visual elements, text, "
    "diagrams, connections, labels, colors, and layout. "
    "If it is a diagram or architecture chart, describe the components "
    "and their relationships.\n\n"
    "IMPORTANT: Start directly with the description of the content. "
    "Do NOT begin with phrases like 'This is', 'This image is', "
    "'This image displays', 'This image shows', or similar introductions."
)

_VLM_PROMPT_NO_CONTEXT = (
    "Describe what is shown in this image. Include all visual elements, text, "
    "diagrams, connections, labels, colors, and layout. "
    "If it is a diagram or architecture chart, describe the components "
    "and their relationships.\n\n"
    "IMPORTANT: Start directly with the description of the content. "
    "Do NOT begin with phrases like 'This is', 'This image is', "
    "'This image displays', 'This image shows', or similar introductions."
)

# Regex patterns to strip common intro prefixes the model may still generate
import re as _re

_INTRO_PATTERNS = _re.compile(
    r"^(?:"
    r"This is a detailed[,.]?\s*(?:high-resolution )?(?:image|description|photograph|picture|diagram|screenshot)?\s*(?:of |showing |depicting |that shows |which shows )?|" 
    r"(?:This|The) image (?:is |displays |shows |depicts |presents |contains |features )?(?:a |an |the )?(?:detailed )?(?:image |picture |photograph |screenshot |diagram )?(?:of |showing |depicting )?|" 
    r"(?:This|The) (?:is |shows |displays |depicts |presents )?(?:a |an |the )?(?:detailed )?(?:image |picture |photograph |screenshot |diagram )?(?:of |showing |depicting )?"
    r")",
    _re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    """Remove generic intro phrases from the start of a caption."""
    cleaned = _INTRO_PATTERNS.sub("", text, count=1).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or text


def _caption_vlm(
    model, processor, image: Image.Image, *, max_tokens: int = 512,
    prompt_text: str = "",
) -> str:
    prompt = prompt_text or _VLM_PROMPT_NO_CONTEXT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
        )

    generated = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
    raw = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    # Strip <think>…</think> reasoning blocks that Qwen3 may still emit
    import re as _re
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    return _strip_intro(raw)


# ---------------------------------------------------------------------------
# NER on captions
# ---------------------------------------------------------------------------

_NER_PIPELINE = None

_FALSE_POSITIVE_ENTITIES = frozenset({
    ".", "..", "...", ",", "!", "?",
    "the", "a", "an", "and", "or", "but", "to", "of", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "it", "its", "they", "them", "their",
    "image", "diagram", "figure", "chart", "table",
})

_MIN_ENTITY_LENGTH = 2


def _load_ner(model_name: str):
    global _NER_PIPELINE
    if _NER_PIPELINE is not None:
        return _NER_PIPELINE

    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    _NER_PIPELINE = pipeline(
        "ner", model=model_name, device=device, aggregation_strategy="simple",
    )
    return _NER_PIPELINE


def _extract_entities(ner_pipe: Any, text: str, confidence: float) -> list[dict]:
    if not text:
        return []
    results = ner_pipe(text)
    seen: set[str] = set()
    entities: list[dict] = []
    for ent in results:
        word = (ent.get("word") or "").strip().strip("##").strip()
        if len(word) < _MIN_ENTITY_LENGTH:
            continue
        if word.lower() in _FALSE_POSITIVE_ENTITIES:
            continue
        score = float(ent.get("score", 0))
        if score < confidence:
            continue
        key = f"{ent['entity_group']}:{word.lower()}"
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "text": word,
            "type": ent["entity_group"],
            "confidence": round(score, 4),
        })
    return entities


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def handle(
    input_file: str,
    output_folder: str,
    config: "CaptionsConfig | None" = None,
    *,
    debug: bool = False,
) -> Optional[dict]:
    """Caption extracted images using BLIP + Qwen3-VL with optional NER.

    Reads images/images.json, generates captions for each image, and
    updates the manifest with ``caption``, ``caption_detailed``,
    ``caption_very_detailed``, and optionally ``entities`` fields.

    Args:
        input_file: Path to the input document file (unused, kept for API parity).
        output_folder: Path to the output directory.
        config: CaptionsConfig instance or None for defaults.
        debug: Enable verbose debug output.

    Returns:
        Updated images manifest or None if no images found.
    """
    # Extract config
    quality = config.quality if config else "quality"
    blip_model_id = config.model_id if config else "Salesforce/blip-image-captioning-large"
    precision = config.precision if config else "fp16"
    vlm_model_id = config.detailed_model_id if config else "Qwen/Qwen3-VL-2B-Instruct"
    vlm_max_tokens_override = config.detailed_max_tokens if config else 0
    run_ner = config.run_ner if config else True
    ner_model = config.ner_model if config else "Jean-Baptiste/roberta-large-ner-english"
    ner_confidence = config.ner_confidence if config else 0.6

    # Resolve VLM max tokens: explicit override > quality preset
    vlm_max_tokens = vlm_max_tokens_override or _QUALITY_TOKENS.get(quality, 512)
    use_vlm = vlm_max_tokens > 0

    info_print(f"quality: {quality}")
    debug_print(f"BLIP model: {blip_model_id}", debug=debug)
    if use_vlm:
        debug_print(f"VLM model: {vlm_model_id} (max_tokens={vlm_max_tokens})", debug=debug)

    images_dir = os.path.join(output_folder, "images")
    manifest_path = os.path.join(images_dir, "images.json")

    if not os.path.isfile(manifest_path):
        info_print("No images manifest found (run with -im first)")
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    image_entries = manifest.get("images", [])
    if not image_entries:
        info_print("No images to caption")
        return manifest

    # ----- Phase 1: BLIP basic captions (batched) -----
    info_print(f"Captioning {len(image_entries)} image(s) with BLIP")

    with gray_debug_output(debug):
        blip_model, blip_proc = _load_blip(model_id=blip_model_id, precision=precision)

    # Load all images upfront for batching
    _blip_images: list[Image.Image] = []
    _blip_indices: list[int] = []  # indices into image_entries
    for idx, entry in enumerate(image_entries):
        filename = entry.get("filename")
        if not filename:
            continue
        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            continue
        try:
            pil_img = Image.open(img_path).convert("RGB")
            _blip_images.append(pil_img)
            _blip_indices.append(idx)
        except Exception:
            continue

    # Process in batches of up to 8 images
    _BLIP_BATCH_SIZE = 8
    with gray_debug_output(debug):
        for batch_start in range(0, len(_blip_images), _BLIP_BATCH_SIZE):
            batch_imgs = _blip_images[batch_start:batch_start + _BLIP_BATCH_SIZE]
            batch_idxs = _blip_indices[batch_start:batch_start + _BLIP_BATCH_SIZE]

            # Basic captions (unconditional)
            captions_basic = _caption_blip_batch(blip_model, blip_proc, batch_imgs)
            # Detailed captions (conditional)
            captions_detailed = _caption_blip_batch(
                blip_model, blip_proc, batch_imgs,
                text_prompt="a detailed photograph of",
            )

            for i, entry_idx in enumerate(batch_idxs):
                image_entries[entry_idx]["caption"] = captions_basic[i]
                image_entries[entry_idx]["caption_detailed"] = captions_detailed[i]

    for idx in _blip_indices:
        entry = image_entries[idx]
        debug_print(
            f"[BLIP {idx+1}/{len(image_entries)}] {entry.get('filename','')}: {entry.get('caption','')[:60]}",
            debug=debug,
        )

    info_print(f"BLIP captions complete for {len(_blip_indices)} image(s)")

    if not use_vlm:
        # Speed mode — skip VLM and NER, save and return
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        info_print(f"Captioned {len(image_entries)} image(s) → images/images.json")
        return manifest

    # Free BLIP before loading VLM
    with gray_debug_output(debug):
        _unload_blip()

    # Get document context for better captioning (reads structure.json if available)
    doc_context = extract_document_context(output_folder, max_chars=300)
    if doc_context:
        debug_print(f"Document context: {doc_context[:80]}...", debug=debug)

    # ----- Phase 2: Qwen3-VL very detailed captions -----
    info_print(f"Generating detailed captions with {vlm_model_id}")

    from .utils.vlm import load_vlm
    with gray_debug_output(debug):
        vlm_model, vlm_proc = load_vlm(model_id=vlm_model_id)

    # Pre-compute the prompt text once (same for all images)
    vlm_prompt_text = (
        _VLM_PROMPT_WITH_CONTEXT.format(context=doc_context)
        if doc_context
        else _VLM_PROMPT_NO_CONTEXT
    )

    for idx, entry in enumerate(image_entries, 1):
        filename = entry.get("filename")
        if not filename:
            continue
        img_path = os.path.join(images_dir, filename)
        if not os.path.isfile(img_path):
            continue
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        with gray_debug_output(debug):
            entry["caption_very_detailed"] = _caption_vlm(
                vlm_model, vlm_proc, pil_img, max_tokens=vlm_max_tokens,
                prompt_text=vlm_prompt_text,
            )
        debug_print(
            f"[VLM {idx}/{len(image_entries)}] {filename}: {entry['caption_very_detailed'][:80]}",
            debug=debug,
        )

    info_print(f"Detailed captions complete for {len(image_entries)} image(s)")

    # Note: VLM stays loaded (shared cache) for potential reuse by overview/classify.
    # It will be unloaded when the process ends or unload_vlm() is called explicitly.

    # ----- Phase 3: NER on detailed captions -----
    if run_ner:
        info_print("Extracting entities from detailed captions")
        with gray_debug_output(debug):
            ner_pipe = _load_ner(ner_model)

        for idx, entry in enumerate(image_entries, 1):
            caption_text = entry.get("caption_very_detailed", "")
            if not caption_text:
                entry["entities"] = []
                continue

            with gray_debug_output(debug):
                entities = _extract_entities(ner_pipe, caption_text, ner_confidence)
            entry["entities"] = entities
            if entities:
                debug_print(
                    f"[NER {idx}/{len(image_entries)}] {len(entities)} entities: "
                    + ", ".join(e["text"] for e in entities[:5]),
                    debug=debug,
                )

        info_print(f"Extracted entities from {len(image_entries)} caption(s)")

    # ----- Save -----
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    info_print(f"Captioned {len(image_entries)} image(s) → images/images.json")
    return manifest
