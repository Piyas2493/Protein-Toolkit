"""
Screening enrichment benchmark: can the screening pipeline rank measured
actives above measured weak binders of the same target?

Two targets: carbonic anhydrase II (a zinc metalloenzyme, 1CA2) and
estrogen receptor alpha (no metal, 3ERT). The docking site is whatever the
pipeline's pocket detector builds from the structure's HETATM records (the
Zn ion in 1CA2, the bound tamoxifen metabolite in 3ERT), i.e. known-site
docking, not blind site finding.

Ligands come from ChEMBL (binding assays only, exact-valued, with a
pChEMBL value, median per molecule): actives are pChEMBL >= 7, negatives
are pChEMBL <= 5. Negatives are chosen to match each active's heavy-atom
count, because docking scores grow with size and actives are larger on
average. A size-only baseline AUC is reported as the control: if docking
doesn't beat it, the pipeline is measuring size, not binding.

Caveats the numbers inherit: negatives are experimentally weak binders of
the SAME target (a hard test, but "weak" is not "non-binder"); one
receptor structure per target; ~50 vs ~50 ligands, so the AUC interval is
wide.

Usage:
    python -m benchmarks.screening --target ca2 --build   # write data/ca2_screen_set.tsv
    python -m benchmarks.screening --target ca2           # dock the set, write results
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGETS = {
    "ca2": {"chembl": "CHEMBL205", "pdb": "1CA2", "chain": "A"},
    "esr1": {"chembl": "CHEMBL206", "pdb": "3ERT", "chain": "A"},
}
HERE = Path(__file__).parent
ACTIVE_PCHEMBL, WEAK_PCHEMBL = 7.0, 5.0
HEAVY_ATOMS = (8, 40)      # range Vina handles well; applied to both classes
MAX_ROT_BONDS = 12
ZBG = "[NX3;H2]S(=O)(=O)[#6]"   # primary sulfonamide, CA II's zinc-binding group


def _chembl_activities(target_chembl_id):
    """{molecule_chembl_id: (median pChEMBL, smiles)}, binding assays only."""
    url = ("https://www.ebi.ac.uk/chembl/api/data/activity.json?"
           f"target_chembl_id={target_chembl_id}&assay_type=B"
           "&standard_relation=%3D&standard_units=nM"
           "&pchembl_value__isnull=false&limit=1000")
    values, smiles = {}, {}
    while url:
        req = urllib.request.Request(url, headers={
            "User-Agent": "ProteinToolkit-benchmark/0.1",
            "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            page = json.load(r)
        for a in page["activities"]:
            if a["canonical_smiles"]:
                values.setdefault(a["molecule_chembl_id"], []).append(
                    float(a["pchembl_value"]))
                smiles[a["molecule_chembl_id"]] = a["canonical_smiles"]
        nxt = page["page_meta"]["next"]
        url = "https://www.ebi.ac.uk" + nxt if nxt else None
    return {m: (float(np.median(v)), smiles[m]) for m, v in values.items()}


def _eligible(smiles):
    """Heavy-atom count if the molecule passes the shared size filters."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or "." in smiles:
        return None
    n = mol.GetNumHeavyAtoms()
    if not HEAVY_ATOMS[0] <= n <= HEAVY_ATOMS[1]:
        return None
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > MAX_ROT_BONDS:
        return None
    return n


def size_matched_negatives(active_sizes, pool, rng):
    """For each active size, the unused pool member (id -> size) closest in
    heavy-atom count, ties broken at random. Returns the chosen ids."""
    pool = dict(pool)
    chosen = []
    for size in active_sizes:
        gap = min(abs(s - size) for s in pool.values())
        pick = rng.choice([m for m, s in pool.items() if abs(s - size) == gap])
        chosen.append(pick)
        del pool[pick]
    return chosen


def build_set(path, target, n_per_class=50, seed=42):
    data = _chembl_activities(TARGETS[target]["chembl"])
    mols = {m: (p, s, _eligible(s)) for m, (p, s) in data.items()}
    mols = {m: v for m, v in mols.items() if v[2] is not None}
    rng = random.Random(seed)
    actives = sorted(m for m, v in mols.items() if v[0] >= ACTIVE_PCHEMBL)
    weak = {m: v[2] for m, v in mols.items() if v[0] <= WEAK_PCHEMBL}
    print(f"{target}: {len(actives)} eligible actives, {len(weak)} eligible weak",
          file=sys.stderr)
    picked = rng.sample(actives, n_per_class)
    negatives = size_matched_negatives([mols[m][2] for m in picked], weak, rng)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chembl_id", "label", "pchembl", "heavy_atoms", "smiles"])
        for label, ids in (("active", picked), ("weak", negatives)):
            for m in ids:
                w.writerow([m, label, mols[m][0], mols[m][2], mols[m][1]])
    print(f"Wrote {path}", file=sys.stderr)


def auc_with_ci(y, score, n_boot=2000, seed=0):
    """ROC AUC and a 95% percentile bootstrap interval (resampling each
    class separately so every resample has both)."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(roc_auc_score(y, score)), float(lo), float(hi)


def run(set_path, out_dir, target, seed=42, n_poses=5):
    from screen import run_screen

    pdb, chain = TARGETS[target]["pdb"], TARGETS[target]["chain"]
    with open(set_path, newline="", encoding="utf-8") as f:
        meta = {r["chembl_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    with tempfile.NamedTemporaryFile("w", suffix=".smi", delete=False,
                                     encoding="utf-8") as lib:
        for m, r in meta.items():
            lib.write(f"{r['smiles']} {m}\n")
    rows = run_screen(pdb, lib.name, chain=chain, docking_engine="vina",
                      n_poses=n_poses, seed=seed)

    ok = [r for r in rows if r["best_score"] != ""]
    y = [int(meta[r["name"]]["label"] == "active") for r in ok]
    score = [-float(r["best_score"]) for r in ok]          # higher = better
    size = [int(meta[r["name"]]["heavy_atoms"]) for r in ok]
    auc, lo, hi = auc_with_ci(y, score)
    size_auc = auc_with_ci(y, size)[0]
    act = [(float(meta[r["name"]]["pchembl"]), s)
           for r, s, yy in zip(ok, score, y) if yy]
    rho = spearmanr(*zip(*act))[0]

    top = sorted(range(len(ok)), key=lambda i: -score[i])[:max(1, len(ok) // 10)]
    ef10 = (sum(y[i] for i in top) / len(top)) / (sum(y) / len(y))
    labels = Counter((meta[r["name"]]["label"], r["overall"]) for r in ok)

    summary = {
        "target": target, "pdb": pdb, "chain": chain, "seed": seed,
        "n_docked": len(ok), "n_failed": len(rows) - len(ok),
        "n_active": sum(y), "n_weak": len(y) - sum(y),
        "roc_auc": auc, "roc_auc_95ci": [lo, hi],
        "size_only_auc": size_auc, "ef_top10pct": ef10,
        "spearman_score_vs_pchembl_actives": float(rho),
        "pose_labels": {f"{k[0]}/{k[1]}": v for k, v in sorted(labels.items())},
        "pockets": dict(Counter(r["pocket_id"] for r in ok)),
        "failures": [(r["name"], r["error"]) for r in rows if r["best_score"] == ""],
    }

    has = np.zeros(len(ok), dtype=int)
    if target == "ca2":
        # Context controls, not tuning targets: CA II binders are dominated by
        # zinc-binding primary sulfonamides, which Vina's scoring has no
        # special term for.
        zbg = Chem.MolFromSmarts(ZBG)
        has = np.array([Chem.MolFromSmiles(meta[r["name"]]["smiles"]).HasSubstructMatch(zbg)
                        for r in ok], dtype=int)
        y_arr, sc_arr, sub = np.array(y), np.array(score), has == 1
        within = (roc_auc_score(y_arr[sub], sc_arr[sub])
                  if sub.sum() >= 6 and 0 < y_arr[sub].sum() < sub.sum() else None)
        summary.update({
            "primary_sulfonamide_fraction": {
                "active": float(has[y_arr == 1].mean()),
                "weak": float(has[y_arr == 0].mean())},
            "sulfonamide_substructure_auc": float(roc_auc_score(y_arr, has)),
            "vina_auc_within_sulfonamides": None if within is None else float(within),
            "n_sulfonamides": int(sub.sum()),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"screening_{target}_ligands.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chembl_id", "label", "pchembl", "heavy_atoms", "best_score",
                    "top_pose_label", "primary_sulfonamide"])
        for r, h in zip(ok, has):
            m = meta[r["name"]]
            w.writerow([r["name"], m["label"], m["pchembl"], m["heavy_atoms"],
                        r["best_score"], r["overall"], int(h)])
    (out_dir / f"screening_{target}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"screening_{target}.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return summary


def to_markdown(s):
    controls = ""
    if "sulfonamide_substructure_auc" in s:
        f = s["primary_sulfonamide_fraction"]
        within = s["vina_auc_within_sulfonamides"]
        controls = (
            f"| Primary-sulfonamide fraction, actives / weak | {f['active']:.0%} / {f['weak']:.0%} |\n"
            f"| ROC AUC, 'has a primary sulfonamide' alone (control) | {s['sulfonamide_substructure_auc']:.2f} |\n"
            f"| ROC AUC, Vina score within the {s['n_sulfonamides']} sulfonamides | "
            f"{'n/a' if within is None else format(within, '.2f')} |\n")
    label_rows = "\n".join(f"| {k.split('/')[0]} | {k.split('/')[1]} | {v} |"
                           for k, v in s["pose_labels"].items())
    return f"""# Screening enrichment benchmark ({s['target'].upper()}, {s['pdb']} chain {s['chain']})

{s['n_active']} actives (ChEMBL binding assays, pChEMBL >= {ACTIVE_PCHEMBL:g}) vs {s['n_weak']} size-matched
weak binders (pChEMBL <= {WEAK_PCHEMBL:g}); {s['n_failed']} ligand(s) failed to dock and are excluded.

| Metric | Value |
|---|---|
| ROC AUC, best Vina score | **{s['roc_auc']:.2f}** (95% CI {s['roc_auc_95ci'][0]:.2f}-{s['roc_auc_95ci'][1]:.2f}) |
| ROC AUC, heavy-atom count only (control) | {s['size_only_auc']:.2f} |
| Enrichment factor, top 10% | {s['ef_top10pct']:.1f} (random = 1.0) |
| Spearman, score vs pChEMBL (actives only) | {s['spearman_score_vs_pchembl_actives']:.2f} |
{controls}
Pocket chosen per ligand: {s['pockets']}

| Truth | Top-pose label | Count |
|---|---|---|
{label_rows}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--target", choices=sorted(TARGETS), default="ca2")
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args(argv)
    path = HERE / "data" / f"{args.target}_screen_set.tsv"
    if args.build:
        build_set(path, args.target, args.n_per_class, args.seed)
    else:
        run(path, Path(args.out), args.target, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
