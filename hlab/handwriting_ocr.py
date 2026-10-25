"""
hlab.handwriting_ocr
--------------------
Read *handwritten* text from an image using Microsoft's TrOCR model
(a transformer trained specifically on handwriting).

TrOCR reads one line at a time, so for a multi-line page we first split the
image into horizontal line-bands (by looking at where the ink is), then read
each line and join them.

This is the heavy option: the model is ~1.3 GB (downloaded once, then cached)
and runs on the CPU, so a page of several lines takes a few seconds.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocess import load_gray, ink_mask

MODEL_NAME = "microsoft/trocr-base-handwritten"
_cache = {}


def is_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch          # noqa: F401
        return True
    except Exception:
        return False


def _load():
    """Load (and cache) the TrOCR processor + model. Downloads on first use."""
    if "model" not in _cache:
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        _cache["proc"] = AutoProcessor.from_pretrained(MODEL_NAME)
        _cache["model"] = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).eval()
    return _cache["proc"], _cache["model"]


def _read_line(pil_image: Image.Image) -> str:
    import torch
    proc, model = _load()
    pixel_values = proc(images=pil_image.convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=64)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _segment_lines(gray: np.ndarray):
    """Return a list of (y0, y1) row-bands that contain a line of writing."""
    mask = ink_mask(gray)
    H, W = mask.shape
    row_ink = mask.sum(axis=1)
    has_ink = row_ink > max(1, 0.004 * W)

    bands, start = [], None
    for i, v in enumerate(has_ink):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append([start, i])
            start = None
    if start is not None:
        bands.append([start, H])

    # Merge bands separated by a tiny vertical gap (parts of the same line).
    merged = []
    gap = max(2, int(0.012 * H))
    for b in bands:
        if merged and b[0] - merged[-1][1] <= gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)

    # Drop specks and add a little vertical padding.
    pad = max(2, int(0.006 * H))
    out = []
    for y0, y1 in merged:
        if y1 - y0 < 0.01 * H:
            continue
        out.append((max(0, y0 - pad), min(H, y1 + pad)))
    return out


def read_handwriting(image_path: str) -> str:
    """Read handwritten text from an image (handles multiple lines)."""
    gray = load_gray(image_path)
    bands = _segment_lines(gray)
    if len(bands) <= 1:
        return _read_line(Image.fromarray(gray)) or "(No readable text found.)"

    lines = []
    for y0, y1 in bands:
        text = _read_line(Image.fromarray(gray[y0:y1, :]))
        if text:
            lines.append(text)
    return "\n".join(lines) or "(No readable text found.)"
