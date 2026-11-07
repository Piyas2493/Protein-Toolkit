"""
Does the triage gate's recall guarantee survive on the EC classifier's own
held-out data?

The gate promises: with probability >= 1 - delta, it keeps at least
`target_recall` of the correctly-classified sequences. That only means
something if it holds for sequences the gate was NOT calibrated on. Here the
held-out rows are split by CLUSTER into a calibration part and a deployment
part (so deployment homologs never appear in calibration), a gate is fitted on
the former, and its realized recall is measured on the latter, over many
random splits. Two gates are compared:

  row-level      treats every calibration row as an independent draw
  cluster-aware  one random positive per cluster (ml/selective_gate.py)

Two realized recalls are reported for each, because they differ:
  cluster-averaged  every cluster counts equally (what the cluster-aware gate
                    bounds)
  row-weighted      every sequence counts equally (large clusters dominate)

"Guarantee held" = realized recall >= target in that split. A valid 95% gate
holds in >= 95% of splits for the quantity it bounds. The deployment part is a
finite sample of clusters, so its own recall wobbles around the population
value; that noise is included, not removed.

Usage:
    python -m ml.training.cluster_sequences
    python -m ml.evaluation --gate-calibration outputs/gate_calib.csv
    python -m benchmarks.gate_coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.selective_gate import RiskControlledGate


def coverage(scores, correct, clusters, cal_fraction, target, delta, n_splits, seed):
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    n_cal = max(1, int(round(cal_fraction * ids.size)))
    out = {m: {"cluster_held": [], "row_held": [], "threshold": [], "saving": []}
           for m in ("row-level", "cluster-aware")}
    for r in range(n_splits):
        cal_ids = set(rng.choice(ids, n_cal, replace=False))
        in_cal = np.fromiter((c in cal_ids for c in clusters), bool, clusters.size)
        for method in out:
            gate = RiskControlledGate(target, delta, seed=r)
            gate.fit(scores[in_cal], correct[in_cal],
                     groups=clusters[in_cal] if method == "cluster-aware" else None)
            res = gate.evaluate(scores[~in_cal], correct[~in_cal],
                                groups=clusters[~in_cal])
            rec = out[method]
            rec["cluster_held"].append(res["realized_cluster_recall"] >= target)
            rec["row_held"].append(res["realized_recall"] >= target)
            rec["threshold"].append(gate.threshold_)
            rec["saving"].append(res["compute_saving"])
    return {m: {"cluster_recall_held": float(np.mean(v["cluster_held"])),
                "row_recall_held": float(np.mean(v["row_held"])),
                "mean_threshold": float(np.mean(v["threshold"])),
                "mean_compute_saved": float(np.mean(v["saving"]))}
            for m, v in out.items()}


def to_markdown(summary):
    s = summary
    lines = [
        "# Triage gate coverage on held-out EC data", "",
        f"Target recall {s['target']:.2f}, delta {s['delta']}; "
        f"{s['n_rows']} held-out rows in {s['n_clusters']} clusters; "
        f"{s['n_splits']} random cluster splits per setting. "
        f"A valid gate holds in >= {1 - s['delta']:.0%} of splits for the "
        "recall it bounds.", "",
        "| Calibrate on | Gate | Cluster-averaged recall held | Row-weighted recall held "
        "| Mean threshold | Mean compute saved |",
        "|---|---|---|---|---|---|"]
    for frac, res in s["settings"].items():
        for method, m in res.items():
            lines.append(
                f"| {float(frac):.0%} of clusters | {method} | "
                f"{m['cluster_recall_held']:.1%} | {m['row_recall_held']:.1%} | "
                f"{m['mean_threshold']:.3f} | {m['mean_compute_saved']:.1%} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default=str(ROOT / "outputs" / "gate_calib.csv"))
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args(argv)

    cal = pd.read_csv(args.calibration)
    if "cluster" not in cal.columns:
        print("Calibration file has no 'cluster' column. Run "
              "`python -m ml.training.cluster_sequences` then "
              "`python -m ml.evaluation --gate-calibration "
              f"{args.calibration}`.", file=sys.stderr)
        return 2
    scores = cal["ml_confidence"].to_numpy(float)
    correct = cal["correct"].to_numpy().astype(bool)
    clusters = cal["cluster"].astype(str).to_numpy()

    summary = {"target": args.target, "delta": args.delta, "n_rows": int(len(cal)),
               "n_clusters": int(np.unique(clusters).size), "n_splits": args.splits,
               "accuracy": float(correct.mean()), "settings": {}}
    for frac in (0.2, 0.5):
        summary["settings"][str(frac)] = coverage(
            scores, correct, clusters, frac, args.target, args.delta,
            args.splits, args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_coverage.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    (out_dir / "gate_coverage.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
