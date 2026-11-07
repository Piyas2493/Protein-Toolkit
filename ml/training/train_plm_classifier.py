"""
Train the EC classifier on frozen ESM-1b embeddings.

Input : ml/training/esm1b_shards/ (from embed_esm1b_dataset) and enzyme_features.csv
        (only its ID/Label columns, so the split is identical to the XGBoost model's)
Output: ml/models/enzyme_classifier_plm.pkl and benchmarks/results/plm_classifier.{json,md}

Protocol, chosen so the held-out test set is touched exactly once:
  1. The cluster-aware split (ml/splits.py) gives train and test; no test cluster
     appears in train.
  2. Inside TRAIN only, a cluster-grouped validation split picks the classifier
     (logistic regression vs a one-hidden-layer MLP, each with its own early
     stopping / regularisation) by validation accuracy.
  3. The same validation split fits one temperature that rescales the chosen
     model's log-probabilities so confidence matches accuracy.
  4. The chosen model is refitted on all of train, and evaluated once on test.

Usage:
    python -m ml.training.train_plm_classifier --weights /path/to/esm1b_t33_650M_UR50S.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation import brier_score, calibration_bins, expected_calibration_error
from ml.plm_classifier import PLMEnzymeWrapper
from ml.splits import held_out_split

RESULTS = ROOT / "benchmarks" / "results"


def load_embeddings(shards: Path, ids: np.ndarray) -> np.ndarray:
    pos = {i: k for k, i in enumerate(ids)}
    out = np.full((len(ids), 1280), np.nan, dtype=np.float32)
    for p in sorted(shards.glob("part_*.npz")):
        d = np.load(p, allow_pickle=False)
        for sid, row in zip(d["ids"], d["emb"]):
            if sid in pos:
                out[pos[sid]] = row
    missing = int(np.isnan(out[:, 0]).sum())
    if missing:
        raise SystemExit(f"{missing} sequences have no embedding yet; finish embed_esm1b_dataset first.")
    return out


def nll(logp: np.ndarray, y: np.ndarray, T: float) -> float:
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    return float(-np.mean(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(axis=1))))


def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    return float(minimize_scalar(lambda T: nll(logp, y, T), bounds=(0.2, 5.0), method="bounded").x)


def fit_mlp(Xf, yf, Xv, yv, classes, max_epochs=60, seed=0):
    """Train an MLP epoch by epoch; return (model at best validation epoch, best epoch, best acc)."""
    import copy
    m = MLPClassifier(hidden_layer_sizes=(1024,), alpha=1e-2, batch_size=256,
                      learning_rate_init=1e-3, random_state=seed)
    rng = np.random.default_rng(seed)
    best, best_ep, best_acc, stale = None, 0, -1.0, 0
    for ep in range(1, max_epochs + 1):
        order = rng.permutation(len(Xf))
        for lo in range(0, len(order), 4096):
            idx = order[lo:lo + 4096]
            m.partial_fit(Xf[idx], yf[idx], classes=classes)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best_acc:
            best, best_ep, best_acc, stale = copy.deepcopy(m), ep, acc, 0
        else:
            stale += 1
            if stale >= 6:
                break
    return best, best_ep, best_acc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="", help="stored in the model as a fallback path")
    parser.add_argument("--shards", default=str(ROOT / "ml" / "training" / "esm1b_shards"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = str(ROOT / "ml" / "models" / "enzyme_classifier_plm.pkl")

    t0 = time.time()
    df = pd.read_csv(ROOT / "ml" / "training" / "enzyme_features.csv", usecols=["ID", "Label"])
    train_idx, test_idx, groups = held_out_split(df)
    if groups is None:
        raise SystemExit("Needs the cluster file: python -m ml.training.cluster_sequences")
    classes = sorted(df["Label"].unique())
    y_all = df["Label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    E = load_embeddings(Path(args.shards), df["ID"].to_numpy())

    Xtr, ytr, gtr = E[train_idx], y_all[train_idx], groups[train_idx]
    Xte, yte = E[test_idx], y_all[test_idx]
    print(f"train {len(Xtr)}  test {len(Xte)}  classes {classes}", flush=True)

    fit_i, val_i = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xtr, ytr, gtr))
    scaler = StandardScaler().fit(Xtr[fit_i])
    Xf, Xv = scaler.transform(Xtr[fit_i]), scaler.transform(Xtr[val_i])
    yf, yv = ytr[fit_i], ytr[val_i]

    candidates = {}
    for C in (0.01, 0.1):
        lr = LogisticRegression(C=C, max_iter=400).fit(Xf, yf)
        candidates[f"logreg C={C}"] = {"model": lr, "val_acc": float((lr.predict(Xv) == yv).mean())}
        print(f"  logreg C={C}: val acc {candidates[f'logreg C={C}']['val_acc']:.4f}", flush=True)
    mlp, ep, acc = fit_mlp(Xf, yf, Xv, yv, np.arange(len(classes)))
    candidates[f"mlp-1024 ({ep} epochs)"] = {"model": mlp, "val_acc": acc, "epochs": ep}
    print(f"  mlp-1024: best epoch {ep}, val acc {acc:.4f}", flush=True)

    name = max(candidates, key=lambda k: candidates[k]["val_acc"])
    chosen = candidates[name]
    logp_val = np.log(np.clip(chosen["model"].predict_proba(Xv), 1e-12, 1))
    T = fit_temperature(logp_val, yv)
    print(f"chosen: {name}; temperature {T:.3f} (val NLL {nll(logp_val, yv, 1.0):.4f} -> {nll(logp_val, yv, T):.4f})", flush=True)

    # refit the chosen family on all of train, then evaluate once on test
    scaler = StandardScaler().fit(Xtr)
    Xa, Xt = scaler.transform(Xtr), scaler.transform(Xte)
    if name.startswith("logreg"):
        final = LogisticRegression(C=float(name.split("=")[1]), max_iter=400).fit(Xa, ytr)
    else:
        final = MLPClassifier(hidden_layer_sizes=(1024,), alpha=1e-2, batch_size=256,
                              learning_rate_init=1e-3, random_state=0)
        rng = np.random.default_rng(0)
        for _ in range(chosen["epochs"]):
            order = rng.permutation(len(Xa))
            for lo in range(0, len(order), 4096):
                idx = order[lo:lo + 4096]
                final.partial_fit(Xa[idx], ytr[idx], classes=np.arange(len(classes)))

    wrapper = PLMEnzymeWrapper(scaler, final, T, classes, weights_path=args.weights or None,
                               source=f"{name}; temperature {T:.3f}")
    proba = wrapper.predict_proba_embeddings(Xte)
    raw = final.predict_proba(Xt)
    pred = proba.argmax(axis=1)
    correct = (pred == yte).astype(float)
    conf = proba[np.arange(len(pred)), pred]
    raw_conf = raw[np.arange(len(pred)), raw.argmax(axis=1)]

    ece_cal = expected_calibration_error(calibration_bins(conf, correct), len(correct))
    ece_raw = expected_calibration_error(
        calibration_bins(raw_conf, (raw.argmax(axis=1) == yte).astype(float)), len(correct))
    recall = {classes[i]: float((pred[yte == i] == i).mean()) for i in range(len(classes))}
    ref = json.loads((RESULTS / "novelty_calibration.json").read_text()) \
        if (RESULTS / "novelty_calibration.json").exists() else {}
    out = {
        "chosen": name, "temperature": T, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "validation": {k: v["val_acc"] for k, v in candidates.items()},
        "test_accuracy": float(correct.mean()),
        "test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "mean_confidence": float(conf.mean()),
        "brier": brier_score(proba, yte, len(classes)),
        "ece_calibrated": ece_cal, "ece_uncalibrated": ece_raw,
        "per_class_recall": recall,
        "xgboost_reference": {"test_accuracy": ref.get("accuracy"), "ece": ref.get("ece_raw")},
        "minutes": (time.time() - t0) / 60,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, args.out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "plm_classifier.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "plm_classifier.md").write_text(to_markdown(out), encoding="utf-8")
    print(to_markdown(out))
    print(f"saved -> {args.out}")
    return 0


def to_markdown(o: dict) -> str:
    x = o["xgboost_reference"]
    rows = "\n".join(f"| {k} | {v:.1%} |" for k, v in o["validation"].items())
    pc = "\n".join(f"| {k} | {v:.0%} |" for k, v in o["per_class_recall"].items())
    xg = f"{x['test_accuracy']:.1%}" if x.get("test_accuracy") else "n/a"
    xe = f"{x['ece']:.3f}" if x.get("ece") else "n/a"
    return f"""# EC classifier on ESM-1b embeddings

Cluster-held-out split (same as the XGBoost model): {o['n_train']} train / {o['n_test']} test sequences.
Model chosen on a cluster-grouped validation split inside train; temperature fitted there; test used once.

| Model | Validation accuracy |
|---|---|
{rows}

Chosen: **{o['chosen']}**, temperature {o['temperature']:.3f}.

| Metric (held-out test) | ESM-1b classifier | XGBoost on sequence features |
|---|---|---|
| Accuracy | **{o['test_accuracy']:.1%}** | {xg} |
| Macro F1 | {o['test_macro_f1']:.3f} | |
| Mean confidence | {o['mean_confidence']:.3f} | |
| Expected calibration error (after temperature) | {o['ece_calibrated']:.3f} | {xe} |
| Expected calibration error (before temperature) | {o['ece_uncalibrated']:.3f} | |
| Brier score | {o['brier']:.3f} | |

| Class | Recall |
|---|---|
{pc}
"""


if __name__ == "__main__":
    sys.exit(main())
