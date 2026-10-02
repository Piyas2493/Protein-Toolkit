"""
Train an XGBoost multiclass classifier on the extended enzyme dataset.

Input  : ml/training/enzyme_features.csv
         (built by `build_feature_dataset` from cached UniProt sequences)
Output : ml/models/enzyme_classifier_xgb.pkl
         (bare XGBClassifier; compatible with ml.predictor.EnzymePredictor
          because we fit on a labeled DataFrame so XGBoost stores
          `feature_names_in_`.)

Usage:
    python -m ml.training.train_xgb_enzyme_classifier
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


def _try_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError as e:
        print("ERROR: xgboost is not installed:", e, file=sys.stderr)
        sys.exit(2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_csv = ROOT / "ml" / "training" / "enzyme_features.csv"
    default_out = ROOT / "ml" / "models" / "enzyme_classifier_xgb.pkl"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--out", default=str(default_out))
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--keep-old", action="store_true")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: dataset not found: {csv_path}", file=sys.stderr)
        print("Run `python -m ml.training.fetch_enzyme_dataset` then "
              "`python -m ml.training.build_feature_dataset` first.",
              file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.keep_old and out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.move(str(out_path), str(backup))
        print(f"Existing model moved to {backup}", file=sys.stderr)

    print(f"Loading {csv_path} ...", file=sys.stderr)
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("ID", "Label")]
    X = df[feature_cols].astype(np.float32)

    le = LabelEncoder()
    y_enc = le.fit_transform(df["Label"])
    print(f"  rows={len(df)} features={len(feature_cols)} "
          f"classes={list(le.classes_)}", file=sys.stderr)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )

    XGBClassifier = _try_xgb()
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )

    print("\n5-fold stratified CV ...", file=sys.stderr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    print(f"  scores = {cv_scores}", file=sys.stderr)
    print(f"  mean   = {cv_scores.mean():.4f}  std={cv_scores.std():.4f}",
          file=sys.stderr)

    print("\nFitting final model on train split ...", file=sys.stderr)
    model.fit(X_train, y_train)

    print("\nTest set evaluation ...", file=sys.stderr)
    preds_enc = model.predict(X_test)
    preds = le.inverse_transform(preds_enc)
    y_test_lbl = le.inverse_transform(y_test)
    acc = accuracy_score(y_test_lbl, preds)
    print(f"  test accuracy = {acc:.4f}", file=sys.stderr)
    print("\nClassification report:", file=sys.stderr)
    print(classification_report(y_test_lbl, preds, zero_division=0),
          file=sys.stderr)
    print("Confusion matrix:", file=sys.stderr)
    print(confusion_matrix(y_test_lbl, preds, labels=list(le.classes_)),
          file=sys.stderr)

    # Save a sklearn-compatible wrapper that decodes string labels
    # internally. EnzymePredictor can call .predict / .predict_proba /
    # .classes_ on it just like on the RandomForest estimator.
    from ml.predictor import XGBoostEnzymeWrapper
    wrapper = XGBoostEnzymeWrapper(model, le, np.array(feature_cols, dtype=object))
    joblib.dump(wrapper, str(out_path))
    print(f"\nSaved {len(feature_cols)}-feature XGBoost -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
