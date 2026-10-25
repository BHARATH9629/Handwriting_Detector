"""
hlab.ocr
--------
Read text out of an image using the free Tesseract engine. If Tesseract is not
installed, every function degrades gracefully with clear instructions instead
of crashing.
"""

from __future__ import annotations

import os
import shutil
import cv2
from PIL import Image

from .preprocess import load_gray

# Where Tesseract typically lives on Windows if it isn't on PATH.
_COMMON_PATHS = [
    os.environ.get("TESSERACT_CMD", ""),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 r"Programs\Tesseract-OCR\tesseract.exe"),
]



def _configure_tesseract():
    """Point pytesseract at the tesseract.exe, searching PATH then common spots."""
    import pytesseract
    exe = shutil.which("tesseract")
    if exe is None:
        for p in _COMMON_PATHS:
            if p and os.path.exists(p):
                exe = p
                break
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    return pytesseract

INSTALL_HELP = (
    "Tesseract OCR engine is not installed yet.\n\n"
    "To turn on this feature:\n"
    "  1. Download the free Windows installer from:\n"
    "     https://github.com/UB-Mannheim/tesseract/wiki\n"
    "  2. Run it (default options are fine).\n"
    "  3. Reopen HandwritingLab.\n\n"
    "The other three features work without it."
)


def tesseract_available() -> bool:
    try:
        pytesseract = _configure_tesseract()
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def read_text(image_path: str) -> str:
    if not tesseract_available():
        return INSTALL_HELP

    pytesseract = _configure_tesseract()
    gray = load_gray(image_path)
    # Tesseract does its own binarisation; we just upscale small images a bit,
    # which noticeably improves accuracy. (No deskew — that's for signatures.)
    longest = max(gray.shape)
    if longest < 1000:
        f = 1000 / longest
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(Image.fromarray(gray))
    return text.strip() or "(No readable text found.)"
