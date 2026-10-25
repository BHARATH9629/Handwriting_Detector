"""
evaluate_signature.py
---------------------
Train the writer-independent signature verifier on the CEDAR dataset and
measure it honestly on signers it never saw during training.

Run:   python evaluate_signature.py

It will:
  1. extract fingerprints for every CEDAR signature (cached to disk),
  2. learn a global embedding (StandardScaler + PCA) on TRAIN signers,
  3. train a classifier on genuine-vs-forged comparisons from TRAIN signers,
  4. report ROC-AUC, Equal-Error-Rate and False-Accept / False-Reject rates
     on the held-out TEST signers,
  5. save the model bundle to models/signature_model.joblib.

Reported numbers come only from TEST signers, so they reflect real-world use
on people the system was never trained on.
"""

import os
import glob
import re
import warnings
import numpy as np

warnings.filterwarnings("ignore")          # keep the training log readable

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from hlab.features import base_features
from hlab.signature import comparison_vector, MODEL_PATH

DATA = "datasets"
FEATURE_CACHE = os.path.join(DATA, "cedar_features.npz")
REFS_PER_SIGNER = 10          # genuine signatures used as the reference set
PCA_DIM = 0.95                # keep 95% of the variance (sklearn picks the dim)
SAMPLINGS = 8                 # random reference subsets per signer (more pairs)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
def collect_paths():
    """Return {signer_id: {'genuine': [...], 'forged': [...]}}."""
    signers = {}
    for kind, folder, prefix in [("genuine", "full_org", "original"),
                                 ("forged", "full_forg", "forgeries")]:
        for path in glob.glob(os.path.join(DATA, folder, f"{prefix}_*.png")):
            m = re.search(rf"{prefix}_(\d+)_(\d+)\.png", os.path.basename(path))
            if not m:
                continue
            sid = int(m.group(1))
            signers.setdefault(sid, {"genuine": [], "forged": []})[kind].append(path)
    return signers


def build_feature_cache(signers):
    """Extract (and cache) base features for every image."""
    if os.path.exists(FEATURE_CACHE):
        print("Using cached features:", FEATURE_CACHE)
        d = np.load(FEATURE_CACHE, allow_pickle=True)
        return d["feats"].item()

    print("Extracting fingerprints for all CEDAR images (one-time, ~8 min)…")
    feats = {}
    total = sum(len(v["genuine"]) + len(v["forged"]) for v in signers.values())
    done = 0
    for sid, groups in signers.items():
        feats[sid] = {"genuine": [], "forged": []}
        for kind in ("genuine", "forged"):
            for p in groups[kind]:
                feats[sid][kind].append(base_features(p))
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{total}")
        feats[sid]["genuine"] = np.array(feats[sid]["genuine"], dtype=np.float32)
        feats[sid]["forged"] = np.array(feats[sid]["forged"], dtype=np.float32)
    np.savez_compressed(FEATURE_CACHE, feats=np.array(feats, dtype=object))
    print("Cached ->", FEATURE_CACHE)
    return feats


def make_pairs(feats, signer_ids, scaler, pca, samplings=4):
    """Build comparison vectors + labels (1=genuine, 0=forged) for given signers."""
    X, y = [], []
    for sid in signer_ids:
        gen = pca.transform(scaler.transform(feats[sid]["genuine"]))
        forg = pca.transform(scaler.transform(feats[sid]["forged"]))
        n = len(gen)
        for _ in range(samplings):
            idx = RNG.permutation(n)
            ref_idx = idx[:REFS_PER_SIGNER]
            test_idx = idx[REFS_PER_SIGNER:]
            refs = gen[ref_idx]
            for q in gen[test_idx]:                      # genuine queries
                X.append(comparison_vector(q, refs)); y.append(1)
            for q in forg:                               # forged queries
                X.append(comparison_vector(q, refs)); y.append(0)
    return np.array(X, dtype=np.float32), np.array(y)


def new_classifier():
    """Gradient-boosted trees — best EER in our sweep over RF/ExtraTrees/SVM."""
    return HistGradientBoostingClassifier(max_iter=800, learning_rate=0.08,
                                          l2_regularization=1.0, random_state=0)


def eer_threshold(y_true, scores):
    """Equal-Error-Rate and the probability threshold where FAR == FRR."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2, float(thr[i])


def main():
    signers = collect_paths()
    feats = build_feature_cache(signers)

    ids = sorted(feats.keys())
    RNG.shuffle(ids)
    split = int(len(ids) * 0.72)
    train_ids, test_ids = ids[:split], ids[split:]
    print(f"\n{len(ids)} signers -> {len(train_ids)} train / {len(test_ids)} test (unseen)")

    # 1) Global embedding learned on TRAIN signers only.
    train_raw = np.vstack([np.vstack([feats[s]["genuine"], feats[s]["forged"]])
                           for s in train_ids])
    scaler = StandardScaler().fit(train_raw)
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(scaler.transform(train_raw))
    print(f"Embedding: {train_raw.shape[1]} -> {pca.n_components_} dims "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance kept)")

    # 2) Train classifier on TRAIN comparisons.
    Xtr, ytr = make_pairs(feats, train_ids, scaler, pca, samplings=SAMPLINGS)
    clf = new_classifier()
    clf.fit(Xtr, ytr)
    print(f"Trained on {len(ytr)} comparisons "
          f"({ytr.sum()} genuine / {len(ytr)-ytr.sum()} forged)")

    # 3) Evaluate on TEST signers (one fixed reference set each).
    cmps, labels = [], []
    for sid in test_ids:
        gen = pca.transform(scaler.transform(feats[sid]["genuine"]))
        forg = pca.transform(scaler.transform(feats[sid]["forged"]))
        refs = gen[:REFS_PER_SIGNER]
        for q in gen[REFS_PER_SIGNER:]:
            cmps.append(comparison_vector(q, refs)); labels.append(1)
        for q in forg:
            cmps.append(comparison_vector(q, refs)); labels.append(0)
    scores = clf.predict_proba(np.array(cmps))[:, 1]
    labels = np.array(labels)

    auc = roc_auc_score(labels, scores)
    eer, thr = eer_threshold(labels, scores)
    pred = (scores >= thr).astype(int)
    genuine_mask = labels == 1
    frr = float((pred[genuine_mask] == 0).mean())     # genuine wrongly rejected
    far = float((pred[~genuine_mask] == 1).mean())    # forgery wrongly accepted
    acc = float((pred == labels).mean())

    print("\n================  HELD-OUT TEST RESULTS  ================")
    print(f"  ROC AUC ................ {auc*100:.2f}%")
    print(f"  Equal Error Rate ...... {eer*100:.2f}%")
    print(f"  Accuracy @ EER thr .... {acc*100:.2f}%")
    print(f"  False Accept Rate ..... {far*100:.2f}%  (forgeries slipping through)")
    print(f"  False Reject Rate ..... {frr*100:.2f}%  (genuine wrongly flagged)")
    print(f"  decision threshold .... {thr:.3f}")
    print("========================================================\n")

    # 4) Retrain on ALL signers for the shipped model, keep test threshold.
    Xall, yall = make_pairs(feats, ids, scaler, pca, samplings=SAMPLINGS)
    clf_final = new_classifier().fit(Xall, yall)

    import joblib
    bundle = {"scaler": scaler, "pca": pca, "clf": clf_final,
              "threshold": thr,
              "metrics": {"auc": auc, "eer": eer, "far": far, "frr": frr,
                          "accuracy": acc},
              "refs_per_signer": REFS_PER_SIGNER, "feat_len": train_raw.shape[1]}
    joblib.dump(bundle, MODEL_PATH)
    print("Saved ->", MODEL_PATH)


if __name__ == "__main__":
    main()
