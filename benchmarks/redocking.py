"""
Redocking benchmark: dock a crystal ligand back into its own receptor
and measure heavy-atom RMSD to the experimental pose. A pose within
2.0 Angstrom RMSD is the conventional success criterion.

Two modes per complex:
  native    box centered on the crystal ligand — isolates the docking
            engine + ligand/receptor preparation from pocket finding.
  pipeline  the toolkit's own PocketDetector/CompatibilityEngine choose
            the pocket. PocketDetector builds pockets from the structure's
            HETATM records, including the very ligand being redocked, so
            this checks the bookkeeping, NOT whether the site can be found.
  blind     same, but with every HETATM removed first, so the detector
            must find the site from protein geometry alone (cavity
            detection). The honest test of site finding.

The docking-input SMILES is derived from the crystal ligand itself
(bond orders from a connectivity template, stereochemistry from the 3D
coordinates), and the starting conformer is freshly embedded — the
crystal pose is never given to the docking engine.

Usage:
    python -m benchmarks.redocking --seeds 42 1 2 --out benchmarks/results
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from docking import (
    BindingBox, DockingManager, DockingRequest, PocketDetector,
    ReceptorPreparation, write_receptor_pdbqt,
)
from ligand import LigandPreparator, load_ligand
from pipeline import _box_for_docking, dock_ligand_against_receptor
from structure.fetch import fetch_pdb

SUCCESS_RMSD = 2.0  # Angstrom

# (PDB id, ligand residue name, chain, connectivity-only template SMILES).
# Each complex was verified against its PDB header (HETNAM/TITLE) before
# inclusion: a metalloenzyme, a charged S1 pocket, a deep polar pocket,
# and a purely apolar buried cavity.
COMPLEXES = [
    ("3HS4", "AZM", "A", "CC(=O)Nc1nnc(s1)S(N)(=O)=O"),    # CA II + acetazolamide
    ("3PTB", "BEN", "A", "NC(=N)c1ccccc1"),                # trypsin + benzamidine
    ("1STP", "BTN", "A", "OC(=O)CCCCC1SCC2NC(=O)NC12"),    # streptavidin + biotin
    ("181L", "BNZ", "A", "c1ccccc1"),                      # T4 lysozyme L99A + benzene
]


def crystal_ligand(pdb_path: str, resname: str, chain: str, template_smiles: str):
    """Crystal ligand as an RDKit mol (heavy atoms, correct bond orders
    and stereo, crystal coordinates)."""
    lines = []
    first_resnum = None
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("HETATM"):
            continue
        if line[17:20].strip() != resname or line[21] != chain:
            continue
        resnum = line[22:27]
        if first_resnum is None:
            first_resnum = resnum
        if resnum != first_resnum:
            continue
        if line[16] not in (" ", "A"):   # keep primary altloc only
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"No HETATM {resname} in chain {chain} of {pdb_path}")

    raw = Chem.MolFromPDBBlock("\n".join(lines), removeHs=True)
    template = Chem.MolFromSmiles(template_smiles)
    mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def centroid(mol):
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pts = [conf.GetAtomPosition(i) for i in range(n)]
    return (sum(p.x for p in pts) / n, sum(p.y for p in pts) / n,
            sum(p.z for p in pts) / n)


def pose_rmsds(out_pdbqt: str, reference):
    """Symmetry-aware heavy-atom RMSD (no re-alignment) of every pose in
    Vina's output PDBQT against the crystal ligand, in Vina rank order."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    pm = PDBQTMolecule(Path(out_pdbqt).read_text(encoding="utf-8"),
                       skip_typing=True)
    docked = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pm)[0])
    return [
        rdMolAlign.CalcRMS(docked, reference, prbId=i, refId=-1)
        for i in range(docked.GetNumConformers())
    ]


def run_native(prep, ref_mol, smiles, seed, n_poses=9, exhaustiveness=8):
    c = centroid(ref_mol)
    box = BindingBox(c[0], c[1], c[2], 8.0, 8.0, 8.0)
    lig = LigandPreparator(seed=seed).prepare(load_ligand(smiles=smiles)).ligand
    docking_box = _box_for_docking(box, lig)
    receptor = tempfile.mktemp(suffix="_receptor.pdbqt")
    write_receptor_pdbqt(prep.receptor_atoms, receptor)
    result = DockingManager(engine="vina").dock(DockingRequest(
        receptor_path=receptor, ligand=lig, box=docking_box,
        n_poses=n_poses, exhaustiveness=exhaustiveness, seed=seed))
    rmsds = pose_rmsds(result.poses[0].provenance["pdbqt"], ref_mol)
    return {
        "top1_rmsd": rmsds[0],
        "best_rmsd": min(rmsds),
        "top1_score": result.poses[0].score,
        "n_poses": len(rmsds),
        "box": [round(docking_box.size_x, 1)] * 3,
    }


def run_pipeline(prep, pockets, ref_mol, smiles, seed, n_poses=9):
    lig = LigandPreparator(seed=seed).prepare(load_ligand(smiles=smiles)).ligand
    compat, result, report = dock_ligand_against_receptor(
        lig, prep, pockets, docking_engine="vina", n_poses=n_poses, seed=seed)
    ref_c = centroid(ref_mol)
    if not compat.best_pocket_id or result is None:
        return {"pocket_found": False, "pocket_dist": None,
                "top1_rmsd": None, "best_rmsd": None, "label": None}
    pocket = next(p for p in pockets if p.pocket_id == compat.best_pocket_id)
    dist = math.dist((pocket.box.center_x, pocket.box.center_y,
                      pocket.box.center_z), ref_c)
    rmsds = pose_rmsds(result.poses[0].provenance["pdbqt"], ref_mol)
    return {
        "pocket_found": True,
        "pocket_id": pocket.pocket_id,
        "pocket_dist": dist,
        "top1_rmsd": rmsds[0],
        "best_rmsd": min(rmsds),
        "label": report.ranked[0].overall if report.ranked else None,
    }


def benchmark(seeds, n_poses, exhaustiveness):
    rows = []
    for pdb_id, resname, chain, template in COMPLEXES:
        print(f"== {pdb_id} / {resname}", file=sys.stderr)
        pdb_path = str(fetch_pdb(pdb_id))
        ref = crystal_ligand(pdb_path, resname, chain, template)
        smiles = Chem.MolToSmiles(ref)
        prep = ReceptorPreparation().prepare(pdb_path, chain=chain)
        pockets = PocketDetector(cutoff=5.0).detect(prep)

        native = [run_native(prep, ref, smiles, s, n_poses, exhaustiveness)
                  for s in seeds]
        pipe = run_pipeline(prep, pockets, ref, smiles, seeds[0], n_poses)

        blind_prep = ReceptorPreparation().prepare(pdb_path, chain=chain)
        blind_prep.structure.ligands = []      # no HETATM anchors -> cavities
        blind = run_pipeline(
            blind_prep, PocketDetector(cutoff=5.0).detect(blind_prep),
            ref, smiles, seeds[0], n_poses)
        rows.append({
            "pdb": pdb_id, "ligand": resname, "smiles": smiles,
            "heavy_atoms": ref.GetNumAtoms(),
            "rotatable_bonds": Chem.rdMolDescriptors.CalcNumRotatableBonds(ref),
            "native": native, "pipeline": pipe, "blind": blind,
        })
    return rows


def summarize(rows):
    def top1_success(r):
        return sum(1 for n in r["native"] if n["top1_rmsd"] <= SUCCESS_RMSD)
    n_runs = sum(len(r["native"]) for r in rows)
    ok = sum(top1_success(r) for r in rows)
    best_ok = sum(1 for r in rows for n in r["native"]
                  if n["best_rmsd"] <= SUCCESS_RMSD)
    return {"native_runs": n_runs, "native_top1_success": ok,
            "native_best_of_n_success": best_ok,
            "native_top1_rate": ok / n_runs if n_runs else None}


def to_markdown(rows, summary, seeds):
    out = ["# Redocking benchmark", "",
           f"Success = heavy-atom RMSD <= {SUCCESS_RMSD} A to the crystal "
           f"pose. Seeds: {', '.join(map(str, seeds))}. Vina exhaustiveness "
           "and pose count as in the run's JSON.", "",
           "## Native-box mode (box centered on the crystal ligand)", "",
           "| PDB | Ligand | Heavy atoms | Rot. bonds | Top-1 RMSD (A) "
           "per seed | Best-of-N RMSD (A) per seed | Top-1 Vina score |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        t1 = ", ".join(f"{n['top1_rmsd']:.2f}" for n in r["native"])
        bn = ", ".join(f"{n['best_rmsd']:.2f}" for n in r["native"])
        sc = statistics.mean(n["top1_score"] for n in r["native"])
        out.append(f"| {r['pdb']} | {r['ligand']} | {r['heavy_atoms']} | "
                   f"{r['rotatable_bonds']} | {t1} | {bn} | {sc:.2f} |")
    out += ["", f"**Top-1 success: {summary['native_top1_success']}/"
            f"{summary['native_runs']} runs** "
            f"({summary['native_top1_rate']:.0%}); best-of-N success: "
            f"{summary['native_best_of_n_success']}/{summary['native_runs']}.",
            ]
    for key, title, note in (
        ("pipeline", "Pipeline mode (pocket from HETATM records)",
         "The pocket is built from the structure's HETATM records, which "
         "include the crystal ligand itself: a small center-to-ligand "
         "distance here is by construction, not site finding."),
        ("blind", "Blind mode (all HETATM removed; cavity detection)",
         "The detector sees protein atoms only, so this is the honest test "
         "of whether the site is found.")):
        out += ["", f"## {title}", "", note, "",
                "| PDB | Pocket found | Pocket center to ligand (A) | "
                "Top-1 RMSD (A) | Best-of-N RMSD (A) | Pose label |",
                "|---|---|---|---|---|---|"]
        for r in rows:
            p = r[key]
            if not p["pocket_found"]:
                out.append(f"| {r['pdb']} | no | - | - | - | - |")
                continue
            out.append(f"| {r['pdb']} | {p['pocket_id']} | {p['pocket_dist']:.1f} | "
                       f"{p['top1_rmsd']:.2f} | {p['best_rmsd']:.2f} | "
                       f"{p['label']} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--n-poses", type=int, default=9)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = parser.parse_args(argv)

    rows = benchmark(args.seeds, args.n_poses, args.exhaustiveness)
    summary = summarize(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "redocking.json").write_text(json.dumps(
        {"seeds": args.seeds, "n_poses": args.n_poses,
         "exhaustiveness": args.exhaustiveness,
         "summary": summary, "complexes": rows}, indent=2), encoding="utf-8")
    md = to_markdown(rows, summary, args.seeds)
    (out_dir / "redocking.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
