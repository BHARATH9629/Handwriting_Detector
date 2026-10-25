"""
hlab.letters
------------
Recognise a single hand-drawn letter (A-Z plus the lowercase letters that
actually look different) with a CNN trained on EMNIST 'balanced'. Trained by
train_letters.py.

Why not all 26 lowercase letters? When a letter is drawn on its own, many
lowercase shapes are identical to their capital (c/C, o/O, s/S, w/W, …) — even
a person can't tell them apart without surrounding text. EMNIST 'balanced'
keeps only the 11 lowercase letters that are genuinely distinct
(a b d e f g h n q r t); the rest are reported as capitals.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import preprocess_digit
from .digits import DigitCNN

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "letter_cnn.pt")

# The 47 EMNIST 'balanced' classes, in label order.
BALANCED = (list("0123456789")
            + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + list("abdefghnqrt"))
LETTER_START = 10                       # classes 10..46 are the letters
LETTERS = BALANCED[LETTER_START:]       # 37 letter characters we can output
N_CLASSES = len(BALANCED)               # 47


class LetterCNN(DigitCNN):
    """Same architecture as the digit network but with 47 EMNIST outputs."""

    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, N_CLASSES)


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Letter model not found. Train it once with:  python train_letters.py")
    ckpt = torch.load(path, map_location="cpu")
    model = LetterCNN()
    model.load_state_dict(ckpt["state"])
    model.eval()
    model._mean, model._std = ckpt["mean"], ckpt["std"]
    return model


def predict(model, image):
    """
    Recognise a drawn letter -> (letter, confidence, letter_probs) or
    (None, 0, None). Only letter classes are considered (digits ignored).
    """
    arr = preprocess_digit(image)
    if arr is None:
        return None, 0.0, None
    norm = (arr - model._mean) / model._std
    t = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1).numpy().ravel()
    letter_probs = probs[LETTER_START:]            # drop the 10 digit classes
    i = int(letter_probs.argmax())
    return LETTERS[i], float(letter_probs[i]), letter_probs
