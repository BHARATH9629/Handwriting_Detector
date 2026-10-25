"""
train_digits.py
---------------
Train the digit CNN on MNIST and save it to models/digit_cnn.pt.

Run once:   python train_digits.py
MNIST (~15 MB) is downloaded automatically the first time and cached.
On a CPU this takes a few minutes; the saved model then loads instantly.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from hlab.digits import DigitCNN, MODEL_PATH, MNIST_MEAN, MNIST_STD


def load_mnist():
    from sklearn.datasets import fetch_openml
    print("Loading MNIST (downloads ~15 MB the first time)…")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, parser="auto")
    X = X.reshape(-1, 28, 28).astype(np.float32) / 255.0
    X = (X - MNIST_MEAN) / MNIST_STD
    y = y.astype(np.int64)
    X = X[:, None, :, :]                              # add channel dim
    # Standard MNIST split: first 60k train, last 10k test.
    return (X[:60000], y[:60000]), (X[60000:], y[60000:])


def main(epochs: int = 6, batch_size: int = 128, lr: float = 1e-3):
    torch.manual_seed(0)
    (Xtr, ytr), (Xte, yte) = load_mnist()

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=512)

    model = DigitCNN()
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
        acc = evaluate(model, test_dl)
        print(f"epoch {epoch}/{epochs}  loss {running/len(Xtr):.4f}  "
              f"test-acc {acc*100:.2f}%  ({time.time()-t0:.0f}s)")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")
    print(f"FINAL TEST ACCURACY: {evaluate(model, test_dl)*100:.2f}%")


def evaluate(model, dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


if __name__ == "__main__":
    main()
