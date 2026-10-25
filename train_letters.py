"""
train_letters.py
----------------
Train the letter CNN on EMNIST 'balanced' (digits + A-Z + 11 distinct
lowercase) and save it to models/letter_cnn.pt.

Run once:   python train_letters.py
EMNIST (~535 MB) downloads automatically the first time into datasets/emnist.
On a CPU this takes several minutes.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision

from hlab.letters import LetterCNN, MODEL_PATH, N_CLASSES


def load_emnist():
    root = "datasets/emnist"
    print("Loading EMNIST 'balanced' (downloads ~535 MB the first time)…")
    tr = torchvision.datasets.EMNIST(root, split="balanced", train=True, download=True)
    te = torchvision.datasets.EMNIST(root, split="balanced", train=False, download=True)

    def prep(ds):
        # EMNIST images are stored transposed; transpose to upright.
        X = ds.data.numpy().astype(np.float32) / 255.0
        X = np.transpose(X, (0, 2, 1))
        y = ds.targets.numpy().astype(np.int64)
        return X, y

    Xtr, ytr = prep(tr)
    Xte, yte = prep(te)
    mean, std = float(Xtr.mean()), float(Xtr.std())
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return (Xtr[:, None], ytr), (Xte[:, None], yte), mean, std


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
    (Xtr, ytr), (Xte, yte), mean, std = load_emnist()
    print(f"train {len(Xtr)}, test {len(Xte)}, {N_CLASSES} classes, "
          f"mean {mean:.4f}, std {std:.4f}")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                          batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                         batch_size=512)

    model = LetterCNN()
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
    print(f"FINAL TEST ACCURACY (47 classes): {evaluate(model, test_dl)*100:.2f}%")


if __name__ == "__main__":
    main()
