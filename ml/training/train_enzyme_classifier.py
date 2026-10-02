"""
Train (or retrain) the enzyme EC-classifier on the real UniProt dataset.

This is a thin wrapper around the existing `ml.trainer.ModelTrainer`.
It:
    1. Loads ml/training/enzyme_dataset.csv.
    2. Trains a 1000-tree RandomForest with 5-fold stratified CV.
    3. Reports accuracy / classification report / confusion matrix.
    4. Saves the new artifact to ml/models/enzyme_classifier.pkl,
       preserving the contract that `main.py` already expects.

Usage:
    python -m ml.training.train_enzyme_classifier \
        --csv ml/training/enzyme_dataset.csv \
        --out ml/models/enzyme_classifier.pkl
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.trainer import ModelTrainer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_csv = ROOT / "ml" / "training" / "enzyme_dataset.csv"
    default_out = ROOT / "ml" / "models" / "enzyme_classifier.pkl"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--keep-old", action="store_true",
                        help="Move existing .pkl to <out>.bak before saving.")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: dataset not found: {csv_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_old and out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.move(str(out_path), str(backup))
        print(f"Existing model moved to {backup}", file=sys.stderr)

    print(f"Training from {csv_path} ...", file=sys.stderr)
    trainer = ModelTrainer(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )
    result = trainer.train(str(csv_path))
    print(f"\nTraining accuracy: {result['accuracy']:.4f}",
          file=sys.stderr)
    print(f"CV mean accuracy : {result['cv_mean']:.4f} "
          f"(±{result['cv_std']:.4f})", file=sys.stderr)

    trainer.save(str(out_path))
    print(f"Saved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
