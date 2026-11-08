"""
Pose-label validation: does a pose's label predict whether the pose is RIGHT?

The STRONG / MODERATE / WEAK label claims to say how well supported a docked
pose looks. That is only useful if it tracks reality, and reality is checkable
for redocking: dock a crystal ligand back into its own protein and measure the
RMSD of every pose to the crystal pose (correct = within 2 A).

Four hand-picked complexes (benchmarks/redocking.py) are too few to judge a
label, so this builds a larger set by RULE, not by choice:

  select  sample structures from RCSB: X-ray, <= 2.0 A, one protein chain, one
          bound small molecule that is not a common additive or ion, 120-350 Da,
          8-28 heavy atoms, with a ring; fixed random seed.
  dock    dock each ligand into its own site (box centred on the crystal ligand,
          Vina, 9 poses) and write one row per pose: Vina score, the analyzer's
          evidence counts, its label, and the true RMSD.
  analyze see benchmarks/pose_validation_analysis.py

Usage:
    python -m benchmarks.pose_validation select
    python -m benchmarks.pose_validation dock            # resumable
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPLEXES = Path(__file__).parent / "data" / "pose_validation_complexes.csv"
POSES = Path(__file__).parent / "results" / "pose_validation_poses.csv"
SUCCESS_RMSD = 2.0

# solvents, buffers, cryoprotectants, crystallisation additives and ions
EXCLUDE = {
    "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "P6G", "PG6", "PE4", "PE3", "2PE", "15P", "SO4", "PO4",
    "ACT", "ACE", "ACY", "DMS", "FMT", "TRS", "MPD", "MRD", "CL", "NA", "ZN", "CA", "MG", "K", "BME",
    "IMD", "EPE", "MES", "NO3", "SCN", "EOH", "IOD", "BR", "FLC", "CIT", "TLA", "MLI", "NH4", "CO3",
    "HEZ", "PIP", "MOH", "BCT", "CAC", "TAM", "BTB", "BTN_", "NAG", "NDG", "BMA", "MAN", "FUC", "GAL",
    "GLC", "SIA", "PGO", "PGR", "OLC", "LDA", "DMF", "TFA", "ETA", "TAR", "SPD", "SPM", "PEP_", "DTT",
    "DTD", "MPO", "HEPES", "CXS", "AZI", "OXL", "MLA", "SUC", "TRE", "BU1", "BU2", "1BO", "2OP",
}
METALS = {"Zn", "Mg", "Ca", "Fe", "Cu", "Mn", "Co", "Ni", "Na", "K", "Cd", "Hg", "Pt", "Au", "Ag", "Pb"}


def _get(url, data=None):
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": "ProteinToolkit-benchmark/0.1", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def _term(attr, op, value):
    return {"type": "terminal", "service": "text",
            "parameters": {"attribute": attr, "operator": op, "value": value}}


def candidate_ids():
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            _term("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            _term("rcsb_entry_info.resolution_combined", "less_or_equal", 2.0),
            _term("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
            _term("rcsb_entry_info.deposited_polymer_entity_instance_count", "equals", 1),
            _term("rcsb_entry_info.nonpolymer_entity_count", "equals", 1),
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
    }
    d = _get("https://search.rcsb.org/rcsbsearch/v2/query", json.dumps(query).encode())
    return sorted(x["identifier"] for x in d["result_set"])


def select(n_candidates: int, n_keep: int, seed: int) -> None:
    from rdkit import Chem

    ids = candidate_ids()
    pool = random.Random(seed).sample(ids, min(n_candidates, len(ids)))
    print(f"{len(ids)} matching entries; examining {len(pool)} sampled at seed {seed}", file=sys.stderr)
    rows = []
    for pdb in pool:
        if len(rows) >= n_keep:
            break
        try:
            entry = _get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}")
            eids = entry["rcsb_entry_container_identifiers"].get("non_polymer_entity_ids") or []
            if len(eids) != 1:
                continue
            ent = _get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb}/{eids[0]}")
            comp = ent["pdbx_entity_nonpoly"]["comp_id"]
            if comp in EXCLUDE:
                continue
            cc = _get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp}")
            smiles = cc["rcsb_chem_comp_descriptor"]["SMILES"]
            mw = float(cc["chem_comp"]["formula_weight"])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or "." in smiles or not 120 <= mw <= 350:
                continue
            ha = mol.GetNumHeavyAtoms()
            if not 8 <= ha <= 28 or mol.GetRingInfo().NumRings() == 0:
                continue
            if any(a.GetSymbol() in METALS for a in mol.GetAtoms()):
                continue
            rows.append({"pdb": pdb, "comp": comp, "smiles": smiles, "mw": round(mw, 1), "heavy_atoms": ha})
            print(f"  kept {pdb} {comp} ({ha} heavy atoms)", file=sys.stderr)
        except Exception as exc:               # a missing field or a network blip: skip the entry
            print(f"  skipped {pdb}: {type(exc).__name__}", file=sys.stderr)
    COMPLEXES.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPLEXES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pdb", "comp", "smiles", "mw", "heavy_atoms"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} complexes -> {COMPLEXES}", file=sys.stderr)


# "label" / "label_legacy" are the RULE-based labels (what is being validated); "_legacy"
# columns come from the analyzer before the aromatic-carbon change
FIELDS = ["pdb", "comp", "heavy_atoms", "pose_id", "rmsd", "correct", "score", "n_contacts", "n_hbonds",
          "n_hbond_residues", "n_hydrophobic", "n_nonpolar_residues", "n_aromatic", "n_ionic", "n_metal",
          "n_clashes", "pocket_occupancy", "label", "n_nonpolar_residues_legacy", "label_legacy"]


def _chains(pdb_path, comp):
    prot, lig = set(), set()
    for line in Path(pdb_path).read_text(errors="replace").splitlines():
        if line.startswith("ATOM"):
            prot.add(line[21])
        elif line.startswith("HETATM") and line[17:20].strip() == comp:
            lig.add(line[21])
    return prot, lig


def dock_one(row):
    from rdkit import Chem

    from benchmarks.redocking import centroid, crystal_ligand, pose_rmsds
    from docking import (BindingBox, DockingManager, DockingRequest, PocketDetector,
                         PoseAnalyzer, ReceptorPreparation)
    from ligand import LigandPreparator, load_ligand
    from pipeline import _box_for_docking
    from structure.fetch import fetch_pdb

    path = str(fetch_pdb(row["pdb"]))
    prot, lig = _chains(path, row["comp"])
    if len(prot) != 1 or not lig or next(iter(lig)) not in prot:
        raise ValueError("ligand not on the single protein chain")
    chain = next(iter(prot))
    ref = crystal_ligand(path, row["comp"], chain, row["smiles"])
    smiles = Chem.MolToSmiles(ref)
    prep = ReceptorPreparation().prepare(path, chain=chain)
    pockets = PocketDetector(cutoff=5.0).detect(prep)
    c = centroid(ref)
    pocket = min(pockets, key=lambda p: math.dist(
        (p.box.center_x, p.box.center_y, p.box.center_z), c), default=None)
    if pocket is None or math.dist((pocket.box.center_x, pocket.box.center_y, pocket.box.center_z), c) > 4.0:
        raise ValueError("no pocket at the crystal ligand")

    ligand = LigandPreparator(seed=42).prepare(load_ligand(smiles=smiles)).ligand
    box = _box_for_docking(BindingBox(c[0], c[1], c[2], 8.0, 8.0, 8.0), ligand)
    result = DockingManager(engine="vina").dock(DockingRequest(
        receptor_path=prep.pdbqt(), ligand=ligand, box=box, n_poses=9, exhaustiveness=8, seed=42))
    rmsds = pose_rmsds(result.poses[0].provenance["pdbqt"], ref)
    report = PoseAnalyzer().analyze(result, prep.receptor_atoms, pocket)
    legacy = PoseAnalyzer(aromatic_hydrophobic=False).analyze(result, prep.receptor_atoms, pocket)
    out = []
    for pa, old, rmsd in zip(report.analyses, legacy.analyses, rmsds):
        out.append({"pdb": row["pdb"], "comp": row["comp"], "heavy_atoms": ref.GetNumAtoms(),
                    "pose_id": pa.pose_id, "rmsd": round(rmsd, 3), "correct": int(rmsd <= SUCCESS_RMSD),
                    "score": pa.score, "n_contacts": pa.n_contacts, "n_hbonds": pa.n_hbonds,
                    "n_hbond_residues": pa.n_hbond_residues, "n_hydrophobic": pa.n_hydrophobic,
                    "n_nonpolar_residues": pa.n_nonpolar_residues, "n_aromatic": pa.n_aromatic,
                    "n_ionic": pa.n_ionic, "n_metal": pa.n_metal, "n_clashes": pa.n_clashes,
                    "pocket_occupancy": round(pa.pocket_occupancy, 4), "label": pa.rule_label,
                    "n_nonpolar_residues_legacy": old.n_nonpolar_residues, "label_legacy": old.rule_label,
                    "plausibility": None if pa.plausibility is None else round(pa.plausibility, 4)})
    return out


def dock(limit):
    rows = list(csv.DictReader(open(COMPLEXES, encoding="utf-8")))
    done = set()
    if POSES.exists():
        done = {r["pdb"] for r in csv.DictReader(open(POSES, encoding="utf-8"))}
    new_file = not POSES.exists()
    POSES.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(POSES, "a", newline="", encoding="utf-8") as f:
        # dock_one also returns the deployed model's "plausibility"; it is not stored, because
        # for the fitted-on complexes it would be in-sample (the analysis computes its own
        # leave-one-complex-out probabilities).
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for row in rows:
            if row["pdb"] in done:
                continue
            if limit and n >= limit:
                break
            n += 1
            try:
                poses = dock_one(row)
                w.writerows(poses)
                f.flush()
                top = poses[0]
                print(f"  {row['pdb']} {row['comp']}: {len(poses)} poses, top-1 RMSD {top['rmsd']:.2f} A, "
                      f"best {min(p['rmsd'] for p in poses):.2f} A", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"  {row['pdb']} {row['comp']}: FAILED ({type(exc).__name__}: {str(exc)[:80]})",
                      file=sys.stderr, flush=True)
                traceback.print_exc(limit=1, file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--candidates", type=int, default=260)
    s.add_argument("--keep", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("dock")
    d.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "select":
        select(args.candidates, args.keep, args.seed)
    else:
        dock(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
