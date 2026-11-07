"""
Novelty-aware confidence for the EC classifier.

A classifier's own probability says how sure it is, not how unfamiliar the
input is. On this toolkit's held-out data the same model is ~90% accurate on
sequences with a close relative in the training data and ~46% accurate on
sequences without one, while reporting 0.85 and 0.61 confidence. So every
prediction here also gets:

  nn_similarity   share of the query's 5-mers found in its closest training
                  family (the same measure ml/training/cluster_sequences.py
                  clusters on; ~0.05 is roughly 55% identity end to end)
  novelty         "novel" | "related" | "familiar" tier of that similarity
  adj_confidence  the model's confidence recalibrated on (confidence,
                  log similarity), i.e. P(this prediction is right) given how
                  novel the input is

The recalibration is a 3-parameter logistic fitted on the held-out sequences
(which no model ever trained on). `build` reports its quality out-of-fold,
grouped by sequence cluster, so the numbers are not fitted-on-itself.

Limits: similarity is a k-mer proxy, blind to remote homology and to
structure; the held-out set only spans the similarity range it spans, so
extrapolating far outside it (e.g. similarity ~0 or ~1) is unvalidated; and it
describes THIS model's training set, so it must be rebuilt when the model is
retrained.

Usage:
    python -m ml.novelty --build                    # needs the model, features, clusters
    python -m ml.novelty --check sequences.fasta    # how much of YOUR set leaks?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.training.cluster_sequences import kmers

ARTIFACT = ROOT / "ml" / "models" / "novelty.pkl"
RESULTS_DIR = ROOT / "benchmarks" / "results"
FLOOR = 1e-3        # added before log() so similarity 0 stays finite
TIERS = ("novel", "related", "familiar")


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class NoveltyIndex:
    """Inverted 5-mer index over training-cluster founders."""

    def __init__(self, founders: Dict[str, str]):
        self.founders = founders
        self._index = defaultdict(list)
        for name, seq in founders.items():
            for km in kmers(seq):
                self._index[km].append(name)

    def nearest(self, seq: str) -> Tuple[float, Optional[str]]:
        km = kmers(seq)
        hits = Counter()
        for k in km:
            for name in self._index.get(k, ()):
                hits[name] += 1
        if not hits:
            return 0.0, None
        name, shared = hits.most_common(1)[0]
        return shared / len(km), name


class NoveltyModel:
    def __init__(self, founders: Dict[str, str], coef, intercept: float,
                 edges, tier_stats=None):
        self.founders = founders
        self.coef = [float(c) for c in coef]
        self.intercept = float(intercept)
        self.edges = [float(e) for e in edges]       # novel < edges[0] <= related < edges[1] <= familiar
        self.tier_stats = tier_stats or {}
        self.index = NoveltyIndex(founders)

    def tier(self, similarity: float) -> str:
        if similarity < self.edges[0]:
            return "novel"
        return "familiar" if similarity >= self.edges[1] else "related"

    def adjust(self, confidence: float, similarity: float) -> float:
        z = (self.intercept + self.coef[0] * _logit(confidence)
             + self.coef[1] * math.log(similarity + FLOOR))
        return 1.0 / (1.0 + math.exp(-z))

    def assess(self, seq: str, confidence: float) -> dict:
        sim, nn = self.index.nearest(seq)
        return {"nn_similarity": round(sim, 4), "nn_id": nn,
                "novelty": self.tier(sim),
                "adj_confidence": round(self.adjust(confidence, sim), 4)}

    def save(self, path: Path = ARTIFACT) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"founders": self.founders, "coef": self.coef,
                     "intercept": self.intercept, "edges": self.edges,
                     "tier_stats": self.tier_stats}, str(path))

    @classmethod
    def load(cls, path: Path = ARTIFACT) -> "NoveltyModel":
        import joblib
        d = joblib.load(str(path))
        return cls(d["founders"], d["coef"], d["intercept"], d["edges"],
                   d.get("tier_stats"))


def load_if_built(path: Path = ARTIFACT) -> Optional[NoveltyModel]:
    """The built model, or None if it hasn't been built / can't be read."""
    try:
        return NoveltyModel.load(path) if Path(path).exists() else None
    except Exception:
        return None


# ---------------------------------------------------------------- building

def _fit(conf: np.ndarray, sim: np.ndarray, correct: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    X = np.c_[[_logit(c) for c in conf], np.log(sim + FLOOR)]
    lr = LogisticRegression(max_iter=1000).fit(X, correct)
    return lr.coef_[0], float(lr.intercept_[0])


def _ece(p: np.ndarray, correct: np.ndarray) -> float:
    from ml.evaluation import calibration_bins, expected_calibration_error
    return expected_calibration_error(
        calibration_bins(p, correct.astype(float)), len(p))


def build(features_csv: Path, sequences_csv: Path, model_path: Path,
          out: Path = ARTIFACT, results_dir: Path = RESULTS_DIR) -> dict:
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from ml.evaluation import _held_out_split
    from ml.model_loader import ModelLoader
    from ml.splits import held_out_split

    ids_df = pd.read_csv(features_csv, usecols=["ID", "Label"])
    train_idx, _, groups = held_out_split(ids_df)
    if groups is None:
        raise SystemExit("No cluster file: run `python -m ml.training.cluster_sequences` first.")
    seqs = pd.read_csv(sequences_csv).set_index("ID")["Sequence"]
    founders = {c: seqs[c] for c in set(groups[train_idx])}
    index = NoveltyIndex(founders)

    X_test, y_test, clusters, test_ids = _held_out_split(features_csv)
    model = ModelLoader.load(str(model_path))
    if hasattr(model, "feature_names_in_"):
        X_test = X_test.reindex(columns=model.feature_names_in_)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    pred = proba.argmax(axis=1)
    conf = proba[np.arange(len(pred)), pred]
    correct = np.array(classes)[pred] == y_test.to_numpy()
    sim = np.array([index.nearest(seqs[i])[0] for i in test_ids])

    edges = np.quantile(sim, [0.2, 0.8])
    coef, intercept = _fit(conf, sim, correct)

    # Out-of-fold (cluster-grouped) recalibration, so the reported quality is
    # honest rather than fitted-on-itself.
    oof = np.zeros(len(conf))
    for tr, te in GroupKFold(5).split(conf, correct, clusters):
        c, b = _fit(conf[tr], sim[tr], correct[tr])
        z = b + c[0] * np.array([_logit(x) for x in conf[te]]) + c[1] * np.log(sim[te] + FLOOR)
        oof[te] = 1 / (1 + np.exp(-z))

    tiers = np.where(sim < edges[0], "novel", np.where(sim >= edges[1], "familiar", "related"))
    tier_stats = {}
    for t in TIERS:
        m = tiers == t
        tier_stats[t] = {"n": int(m.sum()), "accuracy": float(correct[m].mean()),
                         "mean_raw_confidence": float(conf[m].mean()),
                         "mean_adj_confidence": float(oof[m].mean()),
                         "ece_raw": _ece(conf[m], correct[m]),
                         "ece_adj": _ece(oof[m], correct[m])}
    summary = {
        "n_heldout": int(len(conf)), "n_training_families": len(founders),
        "accuracy": float(correct.mean()),
        "similarity_edges": [float(e) for e in edges],
        "auc_raw": float(roc_auc_score(correct, conf)),
        "auc_adjusted_oof": float(roc_auc_score(correct, oof)),
        "ece_raw": _ece(conf, correct), "ece_adjusted_oof": _ece(oof, correct),
        "brier_raw": float(np.mean((conf - correct) ** 2)),
        "brier_adjusted_oof": float(np.mean((oof - correct) ** 2)),
        "tiers": tier_stats,
    }

    NoveltyModel(founders, coef, intercept, edges, tier_stats).save(out)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "novelty_calibration.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "novelty_calibration.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(s: dict) -> str:
    rows = "\n".join(
        f"| {t} | {v['n']} | {v['accuracy']:.1%} | {v['mean_raw_confidence']:.2f} | "
        f"{v['mean_adj_confidence']:.2f} | {v['ece_raw']:.3f} | {v['ece_adj']:.3f} |"
        for t, v in s["tiers"].items())
    return f"""# Novelty-aware confidence (EC classifier)

{s['n_heldout']} held-out sequences (no cluster shared with training), compared with
{s['n_training_families']} training families. Tiers split held-out sequences by similarity
to the nearest training family: novel < {s['similarity_edges'][0]:.4f} <= related
< {s['similarity_edges'][1]:.4f} <= familiar. Adjusted figures are out-of-fold
(grouped by sequence cluster).

| Metric | Raw confidence | Novelty-adjusted |
|---|---|---|
| Expected calibration error | {s['ece_raw']:.4f} | {s['ece_adjusted_oof']:.4f} |
| Brier (is the prediction right?) | {s['brier_raw']:.4f} | {s['brier_adjusted_oof']:.4f} |
| ROC AUC (is the prediction right?) | {s['auc_raw']:.3f} | {s['auc_adjusted_oof']:.3f} |

| Tier | n | Accuracy | Mean raw conf. | Mean adj. conf. | ECE raw | ECE adj. |
|---|---|---|---|---|---|---|
{rows}
"""


# ----------------------------------------------------------------- checking

def check_fasta(fasta: str, model: NoveltyModel) -> dict:
    """How familiar to this model's training data is each sequence in a FASTA?"""
    from core.fasta import FASTAParser
    from core.validator import SequenceValidator

    rows = []
    for rec in FASTAParser.load(fasta):
        seq = SequenceValidator.clean(rec.sequence)
        if not seq:
            continue
        sim, nn = model.index.nearest(seq)
        rows.append({"header": rec.header, "nn_similarity": round(sim, 4),
                     "nn_id": nn, "novelty": model.tier(sim)})
    counts = Counter(r["novelty"] for r in rows)
    return {"rows": rows, "counts": {t: counts.get(t, 0) for t in TIERS}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check", metavar="FASTA")
    parser.add_argument("--model", default=str(ROOT / "ml" / "models" / "enzyme_classifier_xgb.pkl"))
    parser.add_argument("--artifact", default=str(ARTIFACT))
    args = parser.parse_args(argv)

    if args.build:
        s = build(ROOT / "ml" / "training" / "enzyme_features.csv",
                  ROOT / "ml" / "training" / "enzyme_sequences.csv",
                  Path(args.model), Path(args.artifact))
        print(_markdown(s))
        return 0
    if args.check:
        model = load_if_built(Path(args.artifact))
        if model is None:
            print("No novelty model: run `python -m ml.novelty --build` first.",
                  file=sys.stderr)
            return 2
        out = check_fasta(args.check, model)
        n = max(len(out["rows"]), 1)
        for r in out["rows"]:
            print(f"{r['novelty']:9} sim={r['nn_similarity']:.4f}  nearest={r['nn_id']}  {r['header'][:50]}")
        print()
        for t in TIERS:
            st = model.tier_stats.get(t, {})
            acc = f"{st['accuracy']:.0%}" if st else "?"
            print(f"{t:9}: {out['counts'][t]:5d} ({out['counts'][t] / n:.0%})   "
                  f"model accuracy on held-out {t} sequences: {acc}")
        if out["counts"]["familiar"] / n > 0.5:
            print("\nMost of these sequences have a close relative in this model's "
                  "training data, so accuracy measured on them will be inflated "
                  "relative to genuinely new families (66% on the cluster-held-out set).")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
