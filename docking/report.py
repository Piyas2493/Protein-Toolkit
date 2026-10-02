"""
Human-readable docking reports (Sprint 13, Section 28).

Produces output that closely matches the master-prompt Section 28 template.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .compatibility import CompatibilityReport, PocketCompatibility
from .pose_analysis import PoseAnalysisReport, PoseAnalysis
from .pocket import Pocket


class DockingReportWriter:
    """Print a structured docking / compatibility report."""

    @staticmethod
    def _line(label: str, value: Any, width: int = 22) -> None:
        print(f"{label:<{width}} : {value}")

    @staticmethod
    def _title(text: str) -> None:
        print()
        print("=" * 70)
        print(text)
        print("=" * 70)

    # ------------------------------------------------------------------

    @staticmethod
    def compatibility_report(report: CompatibilityReport) -> None:
        DockingReportWriter._title("LIGAND–POCKET COMPATIBILITY")

        DockingReportWriter._line("Ligand", report.ligand_name)
        DockingReportWriter._line("Heavy atoms", report.ligand_profile.heavy_atoms)
        DockingReportWriter._line("HBD", report.ligand_profile.hbd)
        DockingReportWriter._line("HBA", report.ligand_profile.hba)
        DockingReportWriter._line("Net charge", report.ligand_profile.formal_charge)
        DockingReportWriter._line("Aromatic atoms", report.ligand_profile.aromatic_atoms)

        print()
        print(f"Overall assessment : {report.overall_assessment}")
        print(f"Best pocket        : {report.best_pocket_id}")

        for compat in report.pockets:
            DockingReportWriter._pocket_block(compat)

        print()
        print(f"Disclaimer: {report.disclaimer}")

    @staticmethod
    def _pocket_block(compat: PocketCompatibility) -> None:
        DockingReportWriter._title(f"Pocket {compat.pocket_id}")
        DockingReportWriter._line("Geometry score", round(compat.geometry_score, 2))
        DockingReportWriter._line("Chemistry score", round(compat.chemistry_score, 2))
        DockingReportWriter._line("Steric clash risk", compat.steric_clash_risk)
        DockingReportWriter._line("Pocket compatibility", compat.pocket_compatibility)
        DockingReportWriter._line("Chemical compatibility", compat.chemical_compatibility)
        DockingReportWriter._line("Active-site overlap", compat.active_site_overlap)
        DockingReportWriter._line("Overall", compat.overall)
        if compat.warnings:
            print()
            print("Warnings:")
            for w in compat.warnings:
                print(f"  - {w}")

    @staticmethod
    def pose_report(report: PoseAnalysisReport) -> None:
        DockingReportWriter._title("DOCKED POSE ANALYSIS")
        DockingReportWriter._line("Ligand", report.ligand_name)
        DockingReportWriter._line("Pocket", report.pocket_id)
        DockingReportWriter._line("Best pose", report.best_pose_id)
        print()

        for pa in report.ranked:
            DockingReportWriter._pose_block(pa)

        print()
        print(f"Disclaimer: {report.disclaimer}")

    @staticmethod
    def _pose_block(pa: PoseAnalysis) -> None:
        DockingReportWriter._title(f"Pose {pa.pose_id}  ({pa.overall})")
        DockingReportWriter._line("Docking score", round(pa.score, 3))
        DockingReportWriter._line("RMSD to input", round(pa.rmsd_to_input, 3))
        DockingReportWriter._line("Contacts", pa.n_contacts)
        DockingReportWriter._line("H-bonds", pa.n_hbonds)
        DockingReportWriter._line("Hydrophobic", pa.n_hydrophobic)
        DockingReportWriter._line("Ionic", pa.n_ionic)
        DockingReportWriter._line("Aromatic", pa.n_aromatic)
        DockingReportWriter._line("Metal", pa.n_metal)
        DockingReportWriter._line("Clashes", pa.n_clashes)
        DockingReportWriter._line("Active-site overlap", pa.active_overlap)
        DockingReportWriter._line("Pocket occupancy", round(pa.pocket_occupancy, 3))

    @staticmethod
    def final_assessment(
        target_name: str,
        chain: Optional[str],
        compatibility: CompatibilityReport,
        best_pose: Optional[PoseAnalysis],
        best_pocket: Optional[Pocket],
        active_residues: List[Any],
    ) -> None:
        DockingReportWriter._title("FINAL LIGAND–TARGET ASSESSMENT")
        DockingReportWriter._line("Target", target_name)
        DockingReportWriter._line("Chain", chain or "(all)")

        DockingReportWriter._line("Ligand", compatibility.ligand_name)
        DockingReportWriter._line("Heavy atoms", compatibility.ligand_profile.heavy_atoms)
        DockingReportWriter._line("MW / HBD / HBA",
            f"- / {compatibility.ligand_profile.hbd} / {compatibility.ligand_profile.hba}")
        DockingReportWriter._line("Net charge", compatibility.ligand_profile.formal_charge)

        if best_pocket:
            DockingReportWriter._line("Best pocket", best_pocket.pocket_id)
            DockingReportWriter._line("Compatibility", compatibility.overall_assessment)

        if best_pose:
            DockingReportWriter._line("Best pose", best_pose.pose_id)
            DockingReportWriter._line("Docking score", round(best_pose.score, 3))
            DockingReportWriter._line("Pose assessment", best_pose.overall)

        if active_residues:
            print()
            print("Active-site residues considered:")
            for r in active_residues[:20]:
                if isinstance(r, tuple) and len(r) == 3:
                    print(f"  - {r[0]}:{r[1]}{r[2]}")

        print()
        print("IMPORTANT: This is a computational prediction and does not")
        print("establish experimental binding or biological activity.")
