"""
Train/test splitting shared by the training scripts and ml/evaluation.py,
so they always agree on what the held-out set is.

If a cluster file (ID -> cluster, from ml.training.cluster_sequences)
covers every row, the split keeps whole clusters together: no test
sequence has a near-duplicate in training. Otherwise it falls back to a
plain stratified random split — which lets homologs leak across the split
and inflates accuracy — and says so.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLUSTERS = ROOT / "ml" / "training" / "enzyme_clusters.csv"


def load_groups(ids: pd.Series,
                clusters_csv: Optional[Path] = DEFAULT_CLUSTERS
                ) -> Optional[np.ndarray]:
    """Cluster label per row, or None if there is no cluster file or it
    doesn't cover every ID."""
    if clusters_csv is None or not Path(clusters_csv).exists():
        return None
    mapping = pd.read_csv(clusters_csv).set_index("ID")["Cluster"]
    groups = ids.map(mapping)
    return None if groups.isna().any() else groups.to_numpy()


def held_out_split(df: pd.DataFrame,
                   clusters_csv: Optional[Path] = DEFAULT_CLUSTERS,
                   test_size: float = 0.2, seed: int = 42
                   ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """(train_idx, test_idx, groups) as positional indices into df.
    `groups` is the per-row cluster array when the split is cluster-aware
    (use it for grouped cross-validation), else None."""
    y = df["Label"].to_numpy()
    groups = load_groups(df["ID"], clusters_csv)

    if groups is not None:
        n_splits = round(1 / test_size)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, y, groups))
        return train_idx, test_idx, groups

    warnings.warn(
        "No sequence-cluster file covering these IDs: using a random "
        "stratified split, which lets near-duplicate sequences land in "
        "both train and test and inflates accuracy. Run "
        "`python -m ml.training.cluster_sequences` first.",
        stacklevel=2,
    )
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y)
    return train_idx, test_idx, None


def cv_splitter(groups: Optional[np.ndarray], n_splits: int = 5,
                seed: int = 42):
    """Cross-validation splitter matching held_out_split's leakage
    policy; call .split(X, y, groups) on the TRAIN subset."""
    if groups is not None:
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
