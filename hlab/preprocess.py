"""
hlab.preprocess
---------------
Shared image clean-up used by every feature in the app. Keeping it in one
place means the digit recogniser, the writer-ID and the signature checker all
"see" images the same way.

Everything here works on plain numpy arrays / Pillow images and uses OpenCV +
scikit-image, so it runs offline.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Loading / binarising
# ---------------------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    """Read an image file as an 8-bit grayscale numpy array (paper light, ink dark)."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:                                   # fall back to Pillow
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return img


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Return a boolean map where True = ink. Uses Otsu thresholding and
    auto-corrects light-on-dark images so ink is always the minority class.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = bw > 0
    if mask.mean() > 0.5:                             # more "ink" than paper => flip
        mask = ~mask
    return mask


def bounding_box(mask: np.ndarray):
    """Tight (r0, r1, c0, c1) box around the ink, or None if the mask is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1), int(c0), int(c1)


# ---------------------------------------------------------------------------
# Deskew (straighten slanted writing) via image moments
# ---------------------------------------------------------------------------
def deskew(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Shear the image so the dominant slant becomes vertical. Background = white."""
    if mask is None:
        mask = ink_mask(gray)
    m = cv2.moments(mask.astype(np.uint8))
    if abs(m["mu02"]) < 1e-2:
        return gray
    skew = m["mu11"] / m["mu02"]
    h, w = gray.shape
    M = np.float32([[1, skew, -0.5 * h * skew], [0, 1, 0]])
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# ---------------------------------------------------------------------------
# Canonical signature image (fixed size, centred, deskewed)
# ---------------------------------------------------------------------------
def normalize_signature(path: str, size=(128, 256), margin: int = 8,
                        do_deskew: bool = False):
    """
    Turn any signature photo into a canonical, comparable image.

    Note: deskew is OFF by default here. A signature's slant is a personal
    trait that helps expose forgers, so for verification we deliberately keep
    it. (Deskew is still used for OCR, where straightening helps.)

    Returns a dict with:
      gray   : uint8 HxW canonical grayscale (paper=255, ink dark)
      mask   : bool   HxW ink map of the canonical image
      aspect : float  width/height of the original ink (a real style cue)
      density: float  fraction of the cropped box that was ink
    """
    H, W = size
    gray = load_gray(path)
    mask = ink_mask(gray)
    box = bounding_box(mask)
    if box is None:
        blank = np.full((H, W), 255, np.uint8)
        return {"gray": blank, "mask": np.zeros((H, W), bool),
                "aspect": 1.0, "density": 0.0}

    r0, r1, c0, c1 = box
    crop = gray[r0:r1 + 1, c0:c1 + 1]
    crop_mask = mask[r0:r1 + 1, c0:c1 + 1]
    ch, cw = crop.shape
    aspect = cw / ch
    density = float(crop_mask.mean())

    if do_deskew:
        crop = deskew(crop, crop_mask)

    # Scale to fit inside (H-2margin, W-2margin) keeping aspect ratio.
    avail_h, avail_w = H - 2 * margin, W - 2 * margin
    scale = min(avail_h / ch, avail_w / cw)
    new_h, new_w = max(1, int(ch * scale)), max(1, int(cw * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((H, W), 255, np.uint8)
    y0 = (H - new_h) // 2
    x0 = (W - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return {"gray": canvas, "mask": ink_mask(canvas),
            "aspect": float(aspect), "density": density}


# ---------------------------------------------------------------------------
# MNIST-style preprocessing for a hand-drawn digit
# ---------------------------------------------------------------------------
def preprocess_digit(image) -> np.ndarray:
    """
    Convert a drawing (white strokes on black, any size) into a 28x28 float
    array matching how MNIST digits are framed: scaled to a 20px box and
    shifted so the ink's centre of mass sits in the middle. This alignment is
    what makes a model trained on MNIST work on real mouse-drawn digits.

    Returns a (28, 28) float32 array in 0..1, or None if nothing was drawn.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    mask = arr > 60                                   # bright pen = ink
    box = bounding_box(mask)
    if box is None:
        return None
    r0, r1, c0, c1 = box
    digit = arr[r0:r1 + 1, c0:c1 + 1].astype(np.float32)

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    digit = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = digit
    canvas /= 255.0

    # Shift so the centre of mass is at the image centre (like MNIST).
    cy, cx = _center_of_mass(canvas)
    canvas = _shift(canvas, 14 - cx, 14 - cy)
    return canvas


def _center_of_mass(img: np.ndarray):
    total = img.sum()
    if total <= 0:
        return 14.0, 14.0
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return (xs * img).sum() / total, (ys * img).sum() / total


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)
