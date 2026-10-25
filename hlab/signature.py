"""
hlab.signature
--------------
Writer-independent offline signature verification.

How it works
============
1. Every signature becomes a 3808-d fingerprint (hlab.features.base_features).
2. A global StandardScaler + PCA (learned once from many signers) squeezes that
   into a compact, denoised "embedding".
3. To judge a *questioned* signature against a set of *genuine reference*
   signatures, we build a comparison vector (how the questioned one differs
   from the references) and feed it to a classifier trained to tell genuine
   from forged — across signers it has never seen.

This file holds the runtime API (embed / compare / verify) and the model
loader. The model itself is trained & measured by evaluate_signature.py.
"""

from __future__ import annotations

import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "signature_model.joblib")


# ---------------------------------------------------------------------------
# Embedding: raw fingerprint -> compact vector using the global scaler + PCA
# ---------------------------------------------------------------------------
def embed(raw: np.ndarray, model: dict) -> np.ndarray:
    """raw: (n, 3808) or (3808,) -> embeddings (n, d) or (d,)."""
    single = raw.ndim == 1
    X = raw.reshape(1, -1) if single else raw
    Z = model["pca"].transform(model["scaler"].transform(X))
    return Z[0] if single else Z


def embed_paths(paths, model: dict) -> np.ndarray:
    from .features import base_features
    raw = np.stack([base_features(p) for p in paths])
    return embed(raw, model)


# ---------------------------------------------------------------------------
# Comparison vector: questioned embedding vs a set of reference embeddings
# ---------------------------------------------------------------------------
def _cos(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return B_n @ a_n


def comparison_vector(query_emb: np.ndarray, ref_embs: np.ndarray) -> np.ndarray:
    """
    Build one fixed-length vector describing how `query_emb` relates to the
    reference set `ref_embs` (k, d). Works for any number of references >= 1.
    """
    centroid = ref_embs.mean(axis=0)
    diff = np.abs(query_emb - centroid)                       # d

    cos_refs = _cos(query_emb, ref_embs)
    euc_refs = np.linalg.norm(ref_embs - query_emb, axis=1)
    cos_c = float(_cos(query_emb, centroid[None, :])[0])
    euc_c = float(np.linalg.norm(query_emb - centroid))

    stats = np.array([
        cos_refs.min(), cos_refs.mean(), cos_refs.max(), cos_refs.std(),
        euc_refs.min(), euc_refs.mean(), euc_refs.max(), euc_refs.std(),
        cos_c, euc_c,
    ], dtype=np.float32)
    return np.concatenate([diff.astype(np.float32), stats])


# ---------------------------------------------------------------------------
# Runtime model + verdict
# ---------------------------------------------------------------------------
def load_model(path: str = MODEL_PATH) -> dict:
    import joblib
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Signature model not found. Train it once with: "
            "python evaluate_signature.py")
    return joblib.load(path)


def genuine_probability(genuine_paths, questioned_path, model: dict) -> float:
    """Probability (0..1) that the questioned signature is genuine."""
    ref_embs = embed_paths(genuine_paths, model)
    q_emb = embed_paths([questioned_path], model)[0]
    cmp = comparison_vector(q_emb, ref_embs).reshape(1, -1)
    return float(model["clf"].predict_proba(cmp)[0, 1])


def verify(genuine_paths, questioned_path, model: dict | None = None):
    """
    Compare a questioned signature against genuine references.
    Returns (probability_0_to_1, verdict_text).
    """
    if not genuine_paths:
        return 0.0, "Add at least one genuine signature first."
    if model is None:
        model = load_model()

    p = genuine_probability(genuine_paths, questioned_path, model)
    thr = model.get("threshold", 0.5)

    if p >= thr + 0.10:
        verdict = "LIKELY GENUINE"
    elif p >= thr - 0.10:
        verdict = "BORDERLINE — check manually"
    else:
        verdict = "LIKELY FORGERY"
    return p, verdict
