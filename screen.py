"""
Virtual screening: dock a library of candidate ligands against one
receptor's best-matching pocket and rank them by docking score, so you
can answer "which of these ligands is best suited to this protein?"
instead of docking one molecule you already picked.

Receptor preparation and pocket detection run ONCE; only the per-ligand
compatibility + docking step repeats (see
pipeline.dock_ligand_against_receptor).

Ligand library format: a text file, one candidate per line,
    SMILES [optional name]
Blank lines and lines starting with # are ignored.

Usage:
    python screen.py --pdb 1LYZ --library candidates.smi --engine vina --top 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ligand import load_ligand, LigandLoadError, LigandPreparator, LigandValidator
from docking import ReceptorPreparation, PocketDetector
from pipeline import dock_ligand_against_receptor
from structure.fetch import fetch_pdb, looks_like_pdb_id


class Candidate(NamedTuple):
    name: str
    smiles: str


def load_library(path: str) -> List[Candidate]:
    candidates = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        smiles = parts[0]
        name = parts[1].strip() if len(parts) > 1 else smiles
        candidates.append(Candidate(name=name, smiles=smiles))
    return candidates


def screen_one(candidate: Candidate, prep, pockets, *, docking_engine,
               n_poses, seed) -> dict:
    row = {"name": candidate.name, "smiles": candidate.smiles,
           "valid": False, "pocket_id": "", "best_score": "",
           "overall": "", "error": ""}

    try:
        lig = load_ligand(smiles=candidate.smiles)
    except LigandLoadError as exc:
        row["error"] = f"load failed: {exc}"
        return row

    validation = LigandValidator().validate(lig)
    if not validation.valid:
        row["error"] = "failed chemical validation"
        return row
    row["valid"] = True

    lig_prep = LigandPreparator(seed=seed).prepare(lig)

    try:
        compat, result, pose_report = dock_ligand_against_receptor(
            lig_prep.ligand, prep, pockets,
            docking_engine=docking_engine, n_poses=n_poses, seed=seed,
        )
    except Exception as exc:
        row["error"] = f"docking failed: {type(exc).__name__}: {exc}"
        return row

    row["pocket_id"] = compat.best_pocket_id or ""
    if pose_report and pose_report.ranked:
        best = pose_report.ranked[0]
        row["best_score"] = best.score
        row["overall"] = best.overall
    elif not compat.best_pocket_id:
        row["error"] = "no compatible pocket"

    return row


def run_screen(pdb_path: str, library_path: str, *, chain="A",
               docking_engine="mock", n_poses=5, seed=42):
    if not Path(pdb_path).exists() and looks_like_pdb_id(pdb_path):
        print(f"Fetching {pdb_path.upper()} from RCSB...", file=sys.stderr)
        pdb_path = str(fetch_pdb(pdb_path))

    prep = ReceptorPreparation().prepare(pdb_path, chain=chain)
    pockets = PocketDetector(cutoff=5.0).detect(prep)
    print(f"Receptor ready: {len(prep.receptor_atoms)} atoms, "
          f"{len(pockets)} pocket(s)", file=sys.stderr)

    candidates = load_library(library_path)
    rows = [
        screen_one(c, prep, pockets, docking_engine=docking_engine,
                   n_poses=n_poses, seed=seed)
        for c in candidates
    ]

    # Lower (more negative) docking score = better predicted binding.
    # Invalid/failed candidates sort last.
    def sort_key(row):
        return (row["best_score"] == "", row["best_score"] or 0.0)

    rows.sort(key=sort_key)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True,
                        help="Receptor PDB path or a PDB ID to fetch from RCSB")
    parser.add_argument("--library", required=True,
                        help="Ligand library file (one SMILES [name] per line)")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--engine", default="mock",
                        help="Docking engine (mock, vina, gnina)")
    parser.add_argument("--n-poses", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default="screen.csv")
    args = parser.parse_args(argv)

    rows = run_screen(
        args.pdb, args.library, chain=args.chain,
        docking_engine=args.engine, n_poses=args.n_poses, seed=args.seed,
    )

    fieldnames = ["name", "smiles", "valid", "pocket_id", "best_score",
                  "overall", "error"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nScreened {len(rows)} candidate(s) -> {args.out}\n")
    print(f"{'Rank':<5}{'Name':<20}{'Score':<10}{'Assessment':<12}")
    for i, row in enumerate(rows[:args.top], start=1):
        print(f"{i:<5}{row['name'][:19]:<20}"
              f"{str(row['best_score']):<10}{row['overall']:<12}")

    print("\nIMPORTANT: docking scores are a computational prediction, "
          "not measured binding affinity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
