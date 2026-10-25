"""
train_calc.py
-------------
Train the 14-class calculator CNN (digits 0-9 + operators + - × /) and save it
to models/calc_cnn.pt.

Digits come from MNIST (real handwriting). Operators are drawn synthetically
here (there's no standard hand-drawn operator dataset), with lots of random
variation so the model generalises to your handwriting.

Run once:   python train_calc.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image, ImageDraw

from hlab.calc import CalcCNN, MODEL_PATH, CALC_CHARS
from hlab.preprocess import preprocess_digit

RNG = np.random.default_rng(0)
S = 280                                   # synthetic canvas size (matches the app pad)


def _canvas():
    img = Image.new("L", (S, S), 0)
    return img, ImageDraw.Draw(img)


def _jit(p, amt=10):
    return (p[0] + RNG.integers(-amt, amt + 1), p[1] + RNG.integers(-amt, amt + 1))


def draw_operator(kind):
    """Draw one randomised operator and return a 28x28 array via the app pipeline."""
    img, d = _canvas()
    cx, cy = S // 2 + int(RNG.integers(-25, 26)), S // 2 + int(RNG.integers(-25, 26))
    half = int(RNG.integers(60, 101))
    w = int(RNG.integers(10, 20))
    if kind == "+":
        d.line([_jit((cx - half, cy)), _jit((cx + half, cy))], fill=255, width=w)
        d.line([_jit((cx, cy - half)), _jit((cx, cy + half))], fill=255, width=w)
    elif kind == "-":
        d.line([_jit((cx - half, cy)), _jit((cx + half, cy))], fill=255, width=w)
    elif kind == "×":
        d.line([_jit((cx - half, cy - half)), _jit((cx + half, cy + half))], fill=255, width=w)
        d.line([_jit((cx - half, cy + half)), _jit((cx + half, cy - half))], fill=255, width=w)
    elif kind == "/":
        d.line([_jit((cx - half, cy + half)), _jit((cx + half, cy - half))], fill=255, width=w)
    return preprocess_digit(img)


def load_digits(per_class=5000):
    from sklearn.datasets import fetch_openml
    print("Loading MNIST digits…")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, parser="auto")
    X = X.reshape(-1, 28, 28).astype(np.float32) / 255.0
    y = y.astype(np.int64)
    keep_X, keep_y = [], []
    for d in range(10):
        idx = np.where(y == d)[0][:per_class]
        keep_X.append(X[idx]); keep_y.append(y[idx])
    return np.concatenate(keep_X), np.concatenate(keep_y)


def build_dataset(ops_per_class=8000):
    Xd, yd = load_digits()
    print(f"digits: {len(Xd)}; generating {ops_per_class} of each operator…")
    Xo, yo = [], []
    for label, kind in enumerate(["+", "-", "×", "/"], start=10):
        for _ in range(ops_per_class):
            a = draw_operator(kind)
            if a is not None:
                Xo.append(a); yo.append(label)
    Xo = np.array(Xo, dtype=np.float32)
    yo = np.array(yo, dtype=np.int64)
    X = np.concatenate([Xd, Xo])
    y = np.concatenate([yd, yo])
    return X, y


def evaluate(model, dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


def main(epochs: int = 8, batch_size: int = 128, lr: float = 1e-3):
    torch.manual_seed(0)
    X, y = build_dataset()
    mean, std = float(X.mean()), float(X.std())
    X = (X - mean) / std
    X = X[:, None, :, :]

    # shuffle + split 90/10
    perm = RNG.permutation(len(X))
    X, y = X[perm], y[perm]
    cut = int(len(X) * 0.9)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    print(f"train {len(Xtr)}, test {len(Xte)}, {len(CALC_CHARS)} classes, "
          f"mean {mean:.4f}, std {std:.4f}")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                          batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                         batch_size=512)

    model = CalcCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        print(f"epoch {epoch}/{epochs}  loss {running/len(Xtr):.4f}  "
              f"test-acc {evaluate(model, test_dl)*100:.2f}%  ({time.time()-t0:.0f}s)")

    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")
    print(f"FINAL TEST ACCURACY: {evaluate(model, test_dl)*100:.2f}%")


if __name__ == "__main__":
    main()
