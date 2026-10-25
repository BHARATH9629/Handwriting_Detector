"""
hlab.features
-------------
Turns a signature / handwriting image into a fixed-length numeric "fingerprint"
using well-established image descriptors (no deep learning needed):

  * HOG  — Histogram of Oriented Gradients: captures stroke shape & direction
  * LBP  — Local Binary Patterns: captures pen texture / line quality
  * Geometry — aspect, ink density, layout, connected components
  * Hu moments — 7 shape descriptors that ignore position, scale and rotation

These are the classic, reliable building blocks of offline signature
verification systems.
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

from .preprocess import normalize_signature

# Canonical size every signature is squeezed to before measuring.
CANON = (128, 256)

# HOG settings — fixed so the output length never changes. Finer cells (12px)
# capture more stroke detail, which helps separate skilled forgeries.
_HOG_KW = dict(orientations=9, pixels_per_cell=(12, 12),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)

# LBP settings.
_LBP_P, _LBP_R, _LBP_BINS = 8, 1, 10


def _geometry(norm) -> np.ndarray:
    """Global layout/shape numbers from a normalized signature."""
    mask = norm["mask"]
    H, W = mask.shape
    ink = mask.astype(np.uint8)

    ratio = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs):
        cx, cy = xs.mean() / W, ys.mean() / H
    else:
        cx = cy = 0.5

    n_components = cv2.connectedComponents(ink)[0] - 1
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    col_profile = mask.sum(axis=0) / H               # ink per column
    row_profile = mask.sum(axis=1) / W               # ink per row

    hu = cv2.HuMoments(cv2.moments(ink)).ravel()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)  # compress huge dynamic range

    geo = np.array([
        norm["aspect"], norm["density"], ratio, cx, cy,
        min(n_components, 50) / 50.0,
        min(len(contours), 50) / 50.0,
        col_profile.mean(), col_profile.std(),
        row_profile.mean(), row_profile.std(),
    ], dtype=np.float32)
    return np.concatenate([geo, hu.astype(np.float32)])


def base_features(path: str) -> np.ndarray:
    """Full fingerprint vector for one image (fixed length)."""
    norm = normalize_signature(path, size=CANON)
    gray = norm["gray"]

    hog_vec = hog(gray, **_HOG_KW).astype(np.float32)

    lbp = local_binary_pattern(gray, _LBP_P, _LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=_LBP_BINS, range=(0, _LBP_BINS),
                               density=True)
    lbp_hist = lbp_hist.astype(np.float32)

    geo = _geometry(norm)
    return np.concatenate([hog_vec, lbp_hist, geo])
