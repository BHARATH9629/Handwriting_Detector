"""
hlab.writer_id
--------------
"Who wrote this?" — enrol a few handwriting samples per person, then match an
unknown sample to the closest person.

It reuses the global embedding learned for signatures (a compact, denoised
representation of handwriting style) and compares samples by cosine similarity.
"""

from __future__ import annotations

import os
import json
import numpy as np

from .signature import load_model, embed_paths

WRITERS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "writers.json")
os.makedirs(os.path.dirname(WRITERS_PATH), exist_ok=True)


def _load() -> dict:
    if os.path.exists(WRITERS_PATH):
        with open(WRITERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(d: dict) -> None:
    with open(WRITERS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f)


def enroll_writer(name: str, image_paths: list[str]) -> int:
    """Store one style-embedding per sample image under a person's name."""
    model = load_model()
    embs = embed_paths(image_paths, model)
    store = _load()
    vectors = store.get(name, [])
    vectors.extend(embs.tolist())
    store[name] = vectors
    _save(store)
    return len(image_paths)


def list_writers() -> dict:
    return {name: len(vs) for name, vs in _load().items()}


def forget_all_writers() -> None:
    _save({})


def _cos(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return B @ a


def identify_writer(image_path: str):
    """
    Returns (best_name, score_0_to_1, ranked_list) or (None, 0, []) if nobody
    is enrolled. Score is the best cosine similarity to that person's samples.
    """
    store = _load()
    if not store:
        return None, 0.0, []
    model = load_model()
    q = embed_paths([image_path], model)[0]

    ranked = []
    for name, vectors in store.items():
        sims = _cos(q, np.asarray(vectors, dtype=np.float32))
        ranked.append((name, float(sims.max())))
    ranked.sort(key=lambda x: x[1], reverse=True)
    # Map cosine (-1..1) to a friendlier 0..1 confidence.
    best_name, best = ranked[0]
    return best_name, max(0.0, best), ranked
