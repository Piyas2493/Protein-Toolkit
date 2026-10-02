"""
End-to-end analysis pipeline (Sprint 15 conceptual).

This module exposes a single function that runs the full pipeline on a
sequence + structure + ligand triple and produces a structured report
matching Section 28.

    from pipeline import run_analysis
    report = run_analysis(
        fasta_path="...",
        pdb_path="...",
        ligand_smiles="...",
        chain="A",
    )

The function is intended to be the canonical entry point for the
CLI / API layer (Sprint 15).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ligand import (
    load_ligand,
    LigandLoadError,
    LigandPreparator,
    LigandValidator,
)
from docking import (
    ReceptorPreparation,
    PocketDetector,
    BindingBox,
    CompatibilityEngine,
    DockingManager,
    DockingRequest,
    PoseAnalyzer,
    DockingReportWriter,
    write_receptor_pdbqt,
)
from evidence_fusion import EvidenceFusionEngine, EvidenceItem, FusionReport
from structure.fetch import fetch_pdb, looks_like_pdb_id


# ----------------------------------------------------------------------
# Section 28 final report
# ----------------------------------------------------------------------

@dataclass
class FinalReport:
    target: Dict[str, Any] = field(default_factory=dict)
    ligand: Dict[str, Any] = field(default_factory=dict)
    best_pocket: Optional[Dict[str, Any]] = None
    docking: Dict[str, Any] = field(default_factory=dict)
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    active_site_overlap: Dict[str, Any] = field(default_factory=dict)
    evidence: Optional[Dict[str, Any]] = None
    overall_assessment: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "ligand": self.ligand,
            "best_pocket": self.best_pocket,
            "docking": self.docking,
            "interactions": self.interactions,
            "active_site_overlap": self.active_site_overlap,
            "evidence": self.evidence,
            "overall_assessment": self.overall_assessment,
            "provenance": self.provenance,
        }


# ----------------------------------------------------------------------
# Top-level pipeline
# ----------------------------------------------------------------------

def _box_for_docking(pocket_box, ligand, padding: float = 8.0) -> BindingBox:
    """Vina needs room to translate/rotate the ligand, not just enough
    to contain it in its current embedded orientation. Pocket boxes are
    sized for compatibility *scoring* ("does this fit snugly") and can
    be as small as 8 Angstrom on a side — far too tight for a 14+
    Angstrom drug-sized molecule, which then can only dock in clashing,
    unfavorable (near-zero or positive-scoring) poses. Expand to fit
    the ligand's longest dimension plus padding in every axis, keeping
    the pocket's own center.
    """
    bbox = ligand.bounding_box()
    extent = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    needed = extent + padding
    return BindingBox(
        center_x=pocket_box.center_x,
        center_y=pocket_box.center_y,
        center_z=pocket_box.center_z,
        size_x=max(pocket_box.size_x, needed),
        size_y=max(pocket_box.size_y, needed),
        size_z=max(pocket_box.size_z, needed),
    )


def dock_ligand_against_receptor(
    ligand,
    prep,
    pockets,
    *,
    docking_engine: str = "mock",
    n_poses: int = 5,
    seed: int = 42,
    active_residues: Optional[List[Any]] = None,
):
    """Compatibility-assess + dock one already-prepared ligand against an
    already-prepared receptor (receptor prep / pocket detection are the
    expensive, ligand-independent steps — callers that screen many
    ligands against the same receptor should do those once and call this
    per candidate, not re-run the full `run_analysis` pipeline each time).

    Returns (compat, docking_result_or_None, pose_report_or_None).
    """
    compat = CompatibilityEngine().assess(
        ligand, pockets, predicted_active_residues=active_residues,
    )

    if not compat.best_pocket_id:
        return compat, None, None

    best_pocket = next(p for p in pockets
                       if p.pocket_id == compat.best_pocket_id)

    # Real engines (Vina/GNINA) require receptor input in PDBQT, not the
    # raw PDB — write one from the already-parsed, chain-selected
    # receptor atoms. Harmless no-op cost for the mock engine.
    import tempfile
    receptor_pdbqt = tempfile.mktemp(suffix="_receptor.pdbqt")
    write_receptor_pdbqt(prep.receptor_atoms, receptor_pdbqt)

    docking_box = _box_for_docking(best_pocket.box, ligand)
    result = DockingManager(engine=docking_engine).dock(
        DockingRequest(
            receptor_path=receptor_pdbqt,
            ligand=ligand,
            box=docking_box,
            n_poses=n_poses,
            seed=seed,
        )
    )
    pose_report = PoseAnalyzer().analyze(
        result, prep.receptor_atoms, best_pocket, active_residues,
    )
    return compat, result, pose_report


def run_analysis(
    *,
    fasta_path: Optional[str] = None,
    pdb_path: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    ligand_path: Optional[str] = None,
    chain: Optional[str] = "A",
    docking_engine: str = "mock",
    n_poses: int = 5,
    seed: int = 42,
    active_residues: Optional[List[Any]] = None,
) -> FinalReport:
    """Run the full pipeline and return a Section 28-shaped report."""
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "docking_engine": docking_engine,
        "n_poses": n_poses,
        "seed": seed,
        "chain": chain,
    }

    report = FinalReport()
    report.target = {"name": Path(pdb_path).stem if pdb_path else "unknown",
                     "chain": chain,
                     "fasta": Path(fasta_path).name if fasta_path else None}
    report.provenance = provenance

    # ------------------------------------------------------------------
    # Ligand preparation
    # ------------------------------------------------------------------
    if ligand_smiles is None and ligand_path is None:
        raise ValueError("Provide ligand_smiles or ligand_path.")

    try:
        if ligand_smiles is not None:
            lig = load_ligand(smiles=ligand_smiles)
        else:
            lig = load_ligand(source=ligand_path)
    except LigandLoadError as exc:
        report.ligand = {"error": str(exc)}
        report.overall_assessment = (
            "LIGAND FAILED CHEMICAL VALIDATION; "
            "PIPELINE HALTED BEFORE DOCKING."
        )
        return report

    lig_validation = LigandValidator().validate(lig)
    lig_prep = LigandPreparator(seed=seed).prepare(lig)
    report.ligand = {
        "name": lig.name,
        "source_format": lig.source_format,
        "validation": lig_validation.to_dict(),
        "preparation_steps": [s.name for s in lig_prep.steps],
        "provenance_records": len(lig.provenance),
        "heavy_atoms": lig.heavy_atom_count(),
    }
    # Validation gate — do NOT proceed if invalid.
    if not lig_validation.valid:
        report.overall_assessment = (
            "LIGAND FAILED CHEMICAL VALIDATION; "
            "PIPELINE HALTED BEFORE DOCKING."
        )
        return report

    if not pdb_path:
        report.overall_assessment = "STRUCTURE NOT PROVIDED"
        return report

    # ------------------------------------------------------------------
    # Receptor preparation + pockets
    # ------------------------------------------------------------------
    prep = ReceptorPreparation().prepare(pdb_path, chain=chain)
    pockets = PocketDetector(cutoff=5.0).detect(prep)
    report.target["n_residues"] = len(prep.receptor_atoms)
    report.target["n_pockets"] = len(pockets)

    compat, result, pose_report = dock_ligand_against_receptor(
        lig_prep.ligand, prep, pockets,
        docking_engine=docking_engine, n_poses=n_poses, seed=seed,
        active_residues=active_residues,
    )
    report.best_pocket = (
        compat.pockets[0].pocket.to_dict()
        if compat.pockets else None
    )

    if pose_report is not None:
        report.docking = {
            "engine": result.engine,
            "n_poses": len(result.poses),
            "warnings": result.warnings,
        }
        if pose_report.ranked:
            best = pose_report.ranked[0]
            report.docking["best_pose"] = best.pose_id
            report.docking["best_score"] = best.score
            report.docking["overall_pose"] = best.overall
            # Top 10 interactions of the best pose
            report.interactions = [
                {
                    "protein_atom": c.protein_atom,
                    "ligand_atom": c.ligand_atom,
                    "residue": c.residue,
                    "residue_number": c.residue_number,
                    "chain": c.chain,
                    "distance": c.distance,
                    "interaction": c.interaction,
                }
                for c in best.contacts[:10]
            ]
            if active_residues:
                report.active_site_overlap = {
                    "best_pose_overlap": best.active_overlap,
                    "ratio": best.active_overlap_ratio,
                }

    # ------------------------------------------------------------------
    # Cross-evidence fusion
    # ------------------------------------------------------------------
    fusion = EvidenceFusionEngine().fuse(
        sequence_items=EvidenceFusionEngine.from_sequence(),
        structure_items=EvidenceFusionEngine.from_structure(
            pocket_scores=[p.to_dict() for p in pockets[:3]],
        ),
        docking_items=EvidenceFusionEngine.from_docking(
            compatibility_report=compat,
            pose_analysis_report=pose_report,
        ),
    )
    report.evidence = fusion.to_dict()

    # ------------------------------------------------------------------
    # Overall assessment
    # ------------------------------------------------------------------
    report.overall_assessment = _overall_verdict(compat, pose_report, fusion)
    provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def _overall_verdict(compat, pose_report, fusion: FusionReport) -> str:
    if compat.overall_assessment == "INCOMPATIBLE":
        return "INCOMPATIBLE — no pocket shows structural compatibility."
    if pose_report is None or not pose_report.ranked:
        return compat.overall_assessment

    best = pose_report.ranked[0]
    if best.overall == "STRONG" and compat.overall_assessment == \
            "HIGH-PRIORITY DOCKING CANDIDATE":
        return (
            "HIGH-PRIORITY DOCKING CANDIDATE — docking supports a plausible "
            "binding mode consistent with predicted active site."
        )
    if best.overall == "STRONG":
        return (
            "POSSIBLE BINDING MODE — docking supports a plausible pose, "
            "but compatibility evidence is mixed."
        )
    if best.overall == "MODERATE":
        return (
            "WEAK / MODERATE EVIDENCE — docking produced a plausible pose, "
            "but with limited supporting evidence."
        )
    return (
        "WEAK EVIDENCE — docking did not produce a clearly plausible pose."
    )


# ----------------------------------------------------------------------
# CLI driver (Sprint 15)
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Minimal CLI driver.

    Usage:
        python pipeline.py --pdb protein.pdb --smiles "CCO" [--chain A]
    """
    import argparse
    parser = argparse.ArgumentParser(description="ProteinToolkit pipeline")
    parser.add_argument("--pdb", help="Receptor PDB path or a PDB ID "
                        "(e.g. 1LYZ) to fetch from RCSB")
    parser.add_argument("--fasta", help="(optional) FASTA path")
    parser.add_argument("--smiles", help="Candidate ligand SMILES")
    parser.add_argument("--ligand", help="Candidate ligand file (SDF/MOL/MOL2/PDB)")
    parser.add_argument("--chain", default="A", help="Receptor chain")
    parser.add_argument("--engine", default="mock",
                        help="Docking engine (mock, vina, gnina)")
    parser.add_argument("--n_poses", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    if not (args.smiles or args.ligand):
        parser.error("Provide --smiles or --ligand")
    if not args.pdb:
        parser.error("Provide --pdb")

    if not Path(args.pdb).exists() and looks_like_pdb_id(args.pdb):
        print(f"Fetching {args.pdb.upper()} from RCSB...")
        try:
            args.pdb = str(fetch_pdb(args.pdb))
        except ValueError as error:
            parser.error(str(error))

    report = run_analysis(
        fasta_path=args.fasta,
        pdb_path=args.pdb,
        ligand_smiles=args.smiles,
        ligand_path=args.ligand,
        chain=args.chain,
        docking_engine=args.engine,
        n_poses=args.n_poses,
        seed=args.seed,
    )

    print()
    print("=" * 70)
    print("FINAL LIGAND–TARGET ASSESSMENT")
    print("=" * 70)
    print(f"Target                 : {report.target.get('name', 'unknown')}")
    print(f"Chain                  : {report.target.get('chain', '(all)')}")
    print(f"Ligand                 : {report.ligand.get('name', '(none)')}")
    if report.ligand:
        v = report.ligand.get("validation", {})
        print(f"Ligand heavy atoms     : {report.ligand.get('heavy_atoms', '-')}")
        print(f"Ligand valid           : {v.get('valid', '-')}")
    print(f"Best pocket            : "
          f"{report.best_pocket.get('pocket_id') if report.best_pocket else '-'}")
    print(f"Docking engine         : "
          f"{report.docking.get('engine', '-')}")
    print(f"Docking poses          : "
          f"{report.docking.get('n_poses', '-')}")
    print(f"Best pose              : "
          f"{report.docking.get('best_pose', '-')}")
    print(f"Best pose score        : "
          f"{report.docking.get('best_score', '-')}")
    print(f"Best pose assessment   : "
          f"{report.docking.get('overall_pose', '-')}")
    print()
    print("Overall assessment:", report.overall_assessment)
    print()
    print("IMPORTANT: This is a computational prediction and does not")
    print("establish experimental binding or biological activity.")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())