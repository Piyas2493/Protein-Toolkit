"""
Analyse benchmarks/results/pose_validation_poses.csv: which pose label tracks
whether a pose is right?

Every row is one docked pose of a crystal ligand redocked into its own protein,
with the true RMSD to the crystal pose (correct = within 2 A). Scorers compared,
all judged on the SAME poses:

  vina score        -score
  label (legacy)    the STRONG/MODERATE/WEAK label before the aromatic-carbon fix
  label (fixed)     the label now
  fitted model      logistic regression on [score, score gap to the complex's best
                    pose, clashes, H-bond residues, nonpolar residues, contacts,
                    heavy atoms, pocket occupancy], scored by LEAVE-ONE-COMPLEX-OUT
                    cross-validation so a complex never helps predict itself

Metrics:
  pose AUC     pooled over all poses, correct vs incorrect
  top-1        of the pose Vina ranks first, how often is it right, and does the
               scorer's value for it separate the right ones from the wrong ones
  per-label    share of poses that are correct within each label

Decision rule, fixed before the results were seen: adopt the fitted model only
if its leave-one-complex-out pose AUC beats the rule-based label's by > 0.05;
otherwise keep the simple rule.

Usage:
    python -m benchmarks.pose_validation_analysis            # the comparison above
    python -m benchmarks.pose_validation_analysis fit        # write database/pose_plausibility.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
POSES = Path(__file__).parent / "results" / "pose_validation_poses.csv"
RESULTS = Path(__file__).parent / "results"
ORDINAL = {"WEAK": 0, "MODERATE": 1, "STRONG": 2}
FEATURES = ["score", "score_gap", "n_clashes", "n_hbond_residues", "n_nonpolar_residues",
            "n_contacts", "heavy_atoms", "pocket_occupancy"]


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def load() -> pd.DataFrame:
    df = pd.read_csv(POSES)
    df["score_gap"] = df["score"] - df.groupby("pdb")["score"].transform("min")
    df["ord_fixed"] = df["label"].map(ORDINAL)
    df["ord_legacy"] = df["label_legacy"].map(ORDINAL)
    return df


def loco_predictions(df: pd.DataFrame, features=None) -> np.ndarray:
    """Leave-one-complex-out probabilities from the fitted model."""
    features = features or FEATURES
    pred = np.zeros(len(df))
    for pdb in df["pdb"].unique():
        te = (df["pdb"] == pdb).to_numpy()
        sc = StandardScaler().fit(df.loc[~te, features])
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(df.loc[~te, features]),
                                                           df.loc[~te, "correct"])
        pred[te] = lr.predict_proba(sc.transform(df.loc[te, features]))[:, 1]
    return pred


def bootstrap_ci(df, cols, n_boot=400, seed=0):
    """95% interval of pooled pose AUC for each column, resampling COMPLEXES
    (poses of one complex are not independent), plus the fitted-minus-label gap."""
    rng = np.random.default_rng(seed)
    pdbs = df["pdb"].unique()
    groups = {p: g for p, g in df.groupby("pdb")}
    res = {c: [] for c in cols}
    gap = []
    for _ in range(n_boot):
        sample = pd.concat([groups[p] for p in rng.choice(pdbs, len(pdbs))])
        vals = {c: auc(sample["correct"], sample[c]) for c in cols}
        if any(v != v for v in vals.values()):
            continue
        for c in cols:
            res[c].append(vals[c])
        gap.append(vals["fitted"] - vals["ord_fixed"])
    ci = {c: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for c, v in res.items()}
    return ci, [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))]


def analyse(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["fitted"] = loco_predictions(df)
    legacy_feats = [("n_nonpolar_residues_legacy" if f == "n_nonpolar_residues" else f) for f in FEATURES]
    df["fitted_legacy_nonpolar"] = loco_predictions(df, legacy_feats)
    df["neg_score"] = -df["score"]
    top = df[df["pose_id"] == 1]
    out = {
        "n_complexes": int(df["pdb"].nunique()), "n_poses": int(len(df)),
        "pose_correct_rate": float(df["correct"].mean()),
        "vina_top1_success": float(top["correct"].mean()),
        "n_complexes_with_a_correct_pose": int(df.groupby("pdb")["correct"].max().sum()),
        "pose_auc": {
            "vina score": auc(df["correct"], -df["score"]),
            "label (legacy)": auc(df["correct"], df["ord_legacy"]),
            "label (fixed)": auc(df["correct"], df["ord_fixed"]),
            "fitted model (leave-one-complex-out)": auc(df["correct"], df["fitted"]),
            "fitted model, legacy nonpolar count": auc(df["correct"], df["fitted_legacy_nonpolar"]),
        },
        "top1_auc": {
            "vina score": auc(top["correct"], -top["score"]),
            "label (legacy)": auc(top["correct"], top["ord_legacy"]),
            "label (fixed)": auc(top["correct"], top["ord_fixed"]),
            "fitted model (leave-one-complex-out)": auc(top["correct"], top["fitted"]),
        },
        "per_label": {},
    }
    for col, name in (("label_legacy", "legacy"), ("label", "fixed")):
        out["per_label"][name] = {
            lab: {"n": int((df[col] == lab).sum()),
                  "share_correct": float(df.loc[df[col] == lab, "correct"].mean())
                  if (df[col] == lab).any() else None}
            for lab in ("STRONG", "MODERATE", "WEAK")}
    # the fitted model's probability bands, for reference
    bands = pd.cut(df["fitted"], [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
    out["fitted_bands"] = {str(k): {"n": int(len(g)), "share_correct": float(g["correct"].mean())}
                           for k, g in df.groupby(bands, observed=True)}
    ci, gap_ci = bootstrap_ci(df, ["neg_score", "ord_legacy", "ord_fixed", "fitted"])
    out["pose_auc_ci95"] = {"vina score": ci["neg_score"], "label (legacy)": ci["ord_legacy"],
                            "label (fixed)": ci["ord_fixed"],
                            "fitted model (leave-one-complex-out)": ci["fitted"]}
    out["fitted_minus_fixed_label_ci95"] = gap_ci
    tb = pd.cut(top["fitted"], [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
    out["top1_fitted_bands"] = {str(k): {"n": int(len(g)), "share_correct": float(g["correct"].mean())}
                                for k, g in top.groupby(tb, observed=True)}
    gain = out["pose_auc"]["fitted model (leave-one-complex-out)"] - out["pose_auc"]["label (fixed)"]
    out["decision"] = {"fitted_minus_fixed_label_auc": gain,
                       "adopt_fitted_model": bool(gain > 0.05)}
    return out


def to_markdown(o: dict) -> str:
    fmt = lambda v: "n/a" if v is None or v != v else f"{v:.2f}"
    lines = ["# Pose-label validation against crystal poses", "",
             f"{o['n_complexes']} complexes selected by rule from RCSB, {o['n_poses']} docked poses; "
             f"{o['pose_correct_rate']:.0%} of poses are within 2 A of the crystal pose; "
             f"Vina's pose 1 is right for {o['vina_top1_success']:.0%} of complexes; "
             f"{o['n_complexes_with_a_correct_pose']} of {o['n_complexes']} complexes have at least one correct pose.", "",
             "| Scorer | AUC over all poses | AUC on each complex's pose 1 |", "|---|---|---|"]
    for k in o["pose_auc"]:
        ci = o["pose_auc_ci95"].get(k)
        ci_txt = f" ({ci[0]:.2f}-{ci[1]:.2f})" if ci else ""
        lines.append(f"| {k} | {fmt(o['pose_auc'][k])}{ci_txt} | {fmt(o['top1_auc'].get(k))} |")
    lines += ["", "AUC 0.5 = no information, 1.0 = perfect separation of correct from incorrect poses.", "",
              "| Label | Version | Poses | Share correct |", "|---|---|---|---|"]
    for ver, labs in o["per_label"].items():
        for lab, v in labs.items():
            sc = "n/a" if v["share_correct"] is None else f"{v['share_correct']:.0%}"
            lines.append(f"| {lab} | {ver} | {v['n']} | {sc} |")
    lines += ["", "Fitted-model probability bands (leave-one-complex-out):", "",
              "| Band | Poses | Share correct |", "|---|---|---|"]
    for band, v in o["fitted_bands"].items():
        lines.append(f"| {band} | {v['n']} | {v['share_correct']:.0%} |")
    lines += ["", "Calibration on the pose people actually see (each complex's pose 1, leave-one-complex-out):", "",
              "| Fitted band | Complexes | Share correct |", "|---|---|---|"]
    for band, v in o["top1_fitted_bands"].items():
        lines.append(f"| {band} | {v['n']} | {v['share_correct']:.0%} |")
    d = o["decision"]
    lines += ["", f"Decision rule (fixed in advance): adopt the fitted model only if it beats the rule-based "
              f"label by more than 0.05 AUC. Difference: {d['fitted_minus_fixed_label_auc']:+.3f} "
              f"(bootstrap 95% interval {o['fitted_minus_fixed_label_ci95'][0]:+.2f} to {o['fitted_minus_fixed_label_ci95'][1]:+.2f}) -> "
              f"{'adopt the fitted model' if d['adopt_fitted_model'] else 'keep the rule-based label'}.", ""]
    return "\n".join(lines)


def fit_model(df: pd.DataFrame, out: Path) -> dict:
    """Fit the plausibility model on ALL poses and write it where the analyzer
    loads it. The honest accuracy figures are the leave-one-complex-out ones in
    pose_validation.md, not this model's fit to its own training poses."""
    sc = StandardScaler().fit(df[FEATURES])
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(df[FEATURES]), df["correct"])
    spec = {
        "description": "P(docked pose is within 2 A of the crystal pose); see docking/pose_analysis.py",
        "features": FEATURES, "mean": [float(x) for x in sc.mean_], "scale": [float(x) for x in sc.scale_],
        "coef": [float(x) for x in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
        "fitted_on": {"n_complexes": int(df["pdb"].nunique()), "n_poses": int(len(df)),
                      "share_correct": float(df["correct"].mean())},
        "bands": {"STRONG": ">= 0.5", "MODERATE": ">= 0.25", "WEAK": "< 0.25"},
    }
    out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", nargs="?", default="analyze", choices=["analyze", "fit"])
    args = parser.parse_args(argv)
    df = load()
    if args.cmd == "fit":
        spec = fit_model(df, ROOT / "database" / "pose_plausibility.json")
        print("fitted on", spec["fitted_on"])
        print(dict(zip(spec["features"], [round(c, 3) for c in spec["coef"]])), "intercept", round(spec["intercept"], 3))
        return 0
    out = analyse(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pose_validation.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "pose_validation.md").write_text(to_markdown(out), encoding="utf-8")
    print(to_markdown(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
