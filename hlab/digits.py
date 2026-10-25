"""
hlab.digits
-----------
A small but real convolutional neural network (PyTorch) for recognising a
single hand-drawn digit 0-9. Trained on MNIST by train_digits.py.

The drawing is framed exactly like MNIST (see preprocess.preprocess_digit)
before it reaches the network, which is what makes mouse-drawn digits work.
"""

from __future__ import annotations

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .preprocess import preprocess_digit

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "digit_cnn.pt")

# Normalisation constants of the MNIST dataset.
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class DigitCNN(nn.Module):
    """Compact LeNet-style CNN: ~99% test accuracy on MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))          # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))          # 14 -> 7
        x = self.drop(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def load_model(path: str = MODEL_PATH):
    """Load the trained network in eval mode. Raises if it hasn't been trained."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Digit model not found. Train it once with:  python train_digits.py")
    model = DigitCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _to_tensor(arr28: np.ndarray) -> torch.Tensor:
    norm = (arr28 - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)


def predict(model, image):
    """
    Recognise a drawn digit. Returns (digit, confidence_0_to_1, all_probs)
    or (None, 0.0, None) if the canvas is empty.
    """
    arr = preprocess_digit(image)
    if arr is None:
        return None, 0.0, None
    with torch.no_grad():
        logits = model(_to_tensor(arr))
        probs = F.softmax(logits, dim=1).numpy().ravel()
    digit = int(probs.argmax())
    return digit, float(probs[digit]), probs


def predict_sequence(model, image, min_area_frac: float = 0.004):
    """
    Read a whole multi-digit number from a drawing.

    Finds each separate blob of ink (left to right), recognises it, and joins
    the results. Returns (number_string, [(digit, confidence, (x, y, w, h)), …]).
    Digits must not touch each other to be separated.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    mask = (arr > 60).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = arr.shape
    boxes = []
    for i in range(1, n):                              # skip background label 0
        x, y, w, h, area = stats[i]
        if area < min_area_frac * H * W:               # ignore tiny specks
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: b[0])                     # left to right

    results = []
    for (x, y, w, h) in boxes:
        crop = arr[y:y + h, x:x + w]
        digit, conf, _ = predict(model, Image.fromarray(crop))
        if digit is not None:
            results.append((digit, conf, (x, y, w, h)))
    number = "".join(str(d) for d, _, _ in results)
    return number, results
