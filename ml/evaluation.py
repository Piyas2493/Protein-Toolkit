"""
Model calibration evaluation.

Accuracy alone doesn't say whether a model's confidence can be trusted.
A classifier that's "80% confident" should be right about 80% of the
time it says that — most small bioinformatics tools never check this.
This reproduces the exact held-out test split used at training time
(same CSV, same random_state) and reports:

    - Expected Calibration Error (ECE): mean gap between predicted
      confidence and actual accuracy, weighted by bin size.
    - Brier score: mean squared error between the predicted probability
      vector and the one-hot true label (lower is better; 0 = perfect).
    - A reliability diagram (predicted confidence vs. empirical
      accuracy per bin), saved as a PNG.

Usage:
    python -m ml.evaluation --model ml/models/enzyme_classifier_xgb.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.model_loader import ModelLoader
from ml.splits import held_out_split


def _held_out_split(csv_path: Path):
    """Reproduces the exact split used by the training scripts."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("ID", "Label")]
    X = df[feature_cols].astype(np.float32)
    _, test_idx, _ = held_out_split(df)
    return X.iloc[test_idx], df["Label"].iloc[test_idx]


def calibration_bins(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> List[Tuple[float, float, float, int]]:
    """Returns (bin_lo, mean_confidence, empirical_accuracy, count) per
    non-empty bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else \
                 (confidences >= lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        rows.append((
            lo,
            float(confidences[in_bin].mean()),
            float(correct[in_bin].mean()),
            int(in_bin.sum()),
        ))
    return rows


def expected_calibration_error(bins: List[Tuple[float, float, float, int]],
                                total: int) -> float:
    return sum(
        (count / total) * abs(mean_conf - accuracy)
        for _, mean_conf, accuracy, count in bins
    )


def brier_score(probabilities: np.ndarray, y_true_idx: np.ndarray,
                 n_classes: int) -> float:
    one_hot = np.zeros((len(y_true_idx), n_classes), dtype=np.float32)
    one_hot[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def evaluate(model_path: str, csv_path: str, plot_path: str | None):
    model = ModelLoader.load(model_path)
    X_test, y_test = _held_out_split(Path(csv_path))

    if hasattr(model, "feature_names_in_"):
        X_test = X_test.reindex(columns=model.feature_names_in_)

    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)
    class_index = {c: i for i, c in enumerate(classes)}
    y_true_idx = y_test.map(class_index).to_numpy()

    predicted_idx = probabilities.argmax(axis=1)
    confidences = probabilities[np.arange(len(probabilities)), predicted_idx]
    correct = (predicted_idx == y_true_idx).astype(float)

    bins = calibration_bins(confidences, correct)
    ece = expected_calibration_error(bins, len(correct))
    brier = brier_score(probabilities, y_true_idx, len(classes))
    accuracy = float(correct.mean())

    print(f"Test accuracy          : {accuracy:.4f}")
    print(f"Brier score            : {brier:.4f} (0 = perfect, "
          f"{1 - 1 / len(classes):.4f} = uninformative)")
    print(f"Expected Calib. Error  : {ece:.4f} (0 = perfectly calibrated)")
    print()
    print("Reliability (confidence bin -> empirical accuracy):")
    for lo, mean_conf, acc, count in bins:
        print(f"  [{lo:.1f}-{lo + 0.1:.1f}]  "
              f"predicted={mean_conf:.3f}  actual={acc:.3f}  n={count}")

    if plot_path:
        _plot_reliability(bins, plot_path)
        print(f"\nReliability diagram -> {plot_path}")

    return {"accuracy": accuracy, "brier_score": brier, "ece": ece,
            "bins": bins}


def _plot_reliability(bins, plot_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_confs = [b[1] for b in bins]
    accuracies = [b[2] for b in bins]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_confs, accuracies, "o-", label="Model")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=str(ROOT / "ml" / "models" / "enzyme_classifier_xgb.pkl"),
    )
    parser.add_argument(
        "--csv", default=str(ROOT / "ml" / "training" / "enzyme_features.csv"),
    )
    parser.add_argument(
        "--plot", default=str(ROOT / "outputs" / "calibration_reliability.png"),
    )
    args = parser.parse_args(argv)
    evaluate(args.model, args.csv, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
