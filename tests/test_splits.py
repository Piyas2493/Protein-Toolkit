import numpy as np
import pandas as pd
import pytest

from ml.splits import held_out_split
from ml.training.cluster_sequences import cluster


def _frame(n_clusters=40, per_cluster=5):
    ids = [f"s{i}" for i in range(n_clusters * per_cluster)]
    labels = [f"EC{(i // per_cluster) % 3 + 1}" for i in range(len(ids))]
    return pd.DataFrame({"ID": ids, "Label": labels})


def test_clustered_split_keeps_clusters_whole(tmp_path):
    df = _frame()
    clusters = tmp_path / "clusters.csv"
    pd.DataFrame({"ID": df["ID"], "Cluster": np.arange(len(df)) // 5}
                 ).to_csv(clusters, index=False)

    train, test, groups = held_out_split(df, clusters)

    assert set(groups[train]).isdisjoint(groups[test])
    assert len(train) + len(test) == len(df) and len(test) > 0


def test_missing_cluster_file_falls_back_with_warning(tmp_path):
    with pytest.warns(UserWarning, match="near-duplicate"):
        _, _, groups = held_out_split(_frame(), tmp_path / "absent.csv")
    assert groups is None


def test_near_duplicate_sequences_share_a_cluster():
    base = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
    seqs = {"a": base, "b": base[:-3] + "AAA",           # ~96% identical
            "c": "WWPHHCCYYMMNNDDEEQQGGRRKKFFTTSSVVLLIIAAPPWWHHCCYYMMNNDDEEQQGG"}
    groups = cluster(seqs)
    assert groups["a"] == groups["b"] != groups["c"]
