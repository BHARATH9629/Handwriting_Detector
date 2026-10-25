"""
hlab.calc
---------
Handwriting calculator. Reads a hand-drawn arithmetic expression like "7+3"
and works out the answer.

Pipeline:
  1. split the drawing into separate ink blobs, left to right
  2. classify each blob as a digit 0-9 or an operator + - × /
     (a 14-class CNN trained by train_calc.py)
  3. assemble the expression and evaluate it *safely* (no Python eval())
"""

from __future__ import annotations

import os
import ast
import operator
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .preprocess import preprocess_digit
from .digits import DigitCNN

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "calc_cnn.pt")

# 14 classes: ten digits, then four operators.
CALC_CHARS = list("0123456789") + ["+", "-", "×", "/"]
N_CLASSES = len(CALC_CHARS)


class CalcCNN(DigitCNN):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, N_CLASSES)


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Calculator model not found. Train it once with:  python train_calc.py")
    ckpt = torch.load(path, map_location="cpu")
    model = CalcCNN()
    model.load_state_dict(ckpt["state"])
    model.eval()
    model._mean, model._std = ckpt["mean"], ckpt["std"]
    return model


def _classify(model, arr28):
    norm = (arr28 - model._mean) / model._std
    t = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1).numpy().ravel()
    i = int(probs.argmax())
    return CALC_CHARS[i], float(probs[i])


# The calculator model is great at telling digits from operators, but the
# dedicated MNIST digit model (99.1%) reads the actual digit more accurately.
# So we use this model only to confirm "this is a digit", then defer to it.
_DIGIT_MODEL = "unset"


def _digit_model():
    global _DIGIT_MODEL
    if _DIGIT_MODEL == "unset":
        try:
            from .digits import load_model
            _DIGIT_MODEL = load_model()
        except Exception:
            _DIGIT_MODEL = None
    return _DIGIT_MODEL


# ---------------------------------------------------------------------------
# Safe arithmetic evaluation (only numbers and + - * /)
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def safe_eval(expr: str):
    return _eval(ast.parse(expr, mode="eval").body)


# ---------------------------------------------------------------------------
def recognize_expression(model, image, min_area_frac: float = 0.003):
    """
    Read and evaluate a drawn expression.
    Returns dict with: display, result, error, boxes, tokens.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    mask = (arr > 60).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = arr.shape
    boxes = [(int(x), int(y), int(w), int(h))
             for (x, y, w, h, a) in stats[1:] if a >= min_area_frac * H * W]
    boxes.sort(key=lambda b: b[0])

    tokens = []
    for (x, y, w, h) in boxes:
        crop = Image.fromarray(arr[y:y + h, x:x + w])
        a28 = preprocess_digit(crop)
        if a28 is None:
            continue
        ch, _ = _classify(model, a28)
        if ch.isdigit():                       # refine digits with the better model
            dm = _digit_model()
            if dm is not None:
                from .digits import predict as digit_predict
                dd, _, _ = digit_predict(dm, crop)
                if dd is not None:
                    ch = str(dd)
        tokens.append(ch)

    if not tokens:
        return {"display": "", "result": None, "error": "nothing drawn",
                "boxes": boxes, "tokens": tokens}

    # Build an evaluable string: × -> *, group consecutive digits into numbers.
    expr = "".join("*" if t == "×" else t for t in tokens)
    display = " ".join(tokens)
    try:
        value = safe_eval(expr)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        elif isinstance(value, float):
            value = round(value, 4)
        return {"display": display, "result": value, "error": None,
                "boxes": boxes, "tokens": tokens}
    except ZeroDivisionError:
        return {"display": display, "result": None, "error": "can't divide by zero",
                "boxes": boxes, "tokens": tokens}
    except Exception:
        return {"display": display, "result": None,
                "error": "couldn't read a valid sum", "boxes": boxes, "tokens": tokens}
