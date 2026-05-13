"""Shared VLM (Vision-Language Model) loading and text generation.

Provides a cached model loader so that overview, classify, and captions
modules can share the same Qwen3-VL instance without reloading.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

__all__ = ["load_vlm", "unload_vlm", "generate_text"]

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_VLM_MODEL: Any = None
_VLM_PROCESSOR: Any = None
_VLM_MODEL_ID: str | None = None


def load_vlm(model_id: str) -> tuple[Any, Any]:
    """Load and cache the VLM model + processor.

    If the model is already loaded with the same *model_id*, returns the
    cached instances immediately.
    """
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_MODEL_ID

    if _VLM_MODEL is not None and _VLM_MODEL_ID == model_id:
        return _VLM_MODEL, _VLM_PROCESSOR

    # Different model requested — unload first
    if _VLM_MODEL is not None:
        unload_vlm()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    _VLM_MODEL = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    _VLM_MODEL.eval()
    _VLM_PROCESSOR = AutoProcessor.from_pretrained(model_id)
    _VLM_MODEL_ID = model_id
    return _VLM_MODEL, _VLM_PROCESSOR


def unload_vlm() -> None:
    """Release the cached VLM model and free GPU memory."""
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_MODEL_ID
    del _VLM_MODEL, _VLM_PROCESSOR
    _VLM_MODEL = None
    _VLM_PROCESSOR = None
    _VLM_MODEL_ID = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_text(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> str:
    """Generate text using the VLM in text-only mode (no image).

    Constructs a single-turn chat message and returns the decoded output.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

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
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    generated = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
