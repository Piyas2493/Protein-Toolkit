"""
Docked pose analysis (Sprint 8).

For each pose, compute:
    - docking score (from the engine)
    - pose rank
    - RMSD to input ligand
    - protein-ligand contacts (within 4.0 Å)
    - hydrogen bond candidates (donor-acceptor distance)
    - hydrophobic contacts
    - salt bridges / ionic interactions
    - aromatic / pi-pi interactions
    - metal coordination
    - steric clashes (< 2.0 Å heavy-heavy)
    - active-site residue overlap
    - pocket occupancy

Then rank with multi-evidence:
    docking_score + hbond_count + active_overlap + clash_penalty

NOT a probability. Multi-evidence ordering, not a magic threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ligand import Ligand

from .manager import DockedPose, DockingResult
from .pocket import Pocket


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class ContactRecord:
    protein_atom: str           # "A:ASP184.OD2"
    ligand_atom: str            # "O3"
    residue: str                # "ASP"
    residue_number: int
    chain: str
    distance: float
    interaction: str            # human-readable label


@dataclass
class PoseAnalysis:
    pose_id: int
    score: float
    rmsd_to_input: float
    n_contacts: int
    n_hbonds: int
    n_hydrophobic: int
    n_ionic: int
    n_aromatic: int
    n_metal: int
    n_clashes: int
    active_overlap: int
    active_overlap_ratio: float
    pocket_occupancy: float
    overall: str                # STRONG / MODERATE / WEAK
    contacts: List[ContactRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Distinct-residue counts (n_hbonds / n_hydrophobic above count atom
    # PAIRS, which a small molecule near a few residues inflates).
    n_hbond_residues: int = 0
    n_nonpolar_residues: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "score": round(self.score, 3),
            "rmsd_to_input": round(self.rmsd_to_input, 3),
            "n_contacts": self.n_contacts,
            "n_hbonds": self.n_hbonds,
            "n_hydrophobic": self.n_hydrophobic,
            "n_ionic": self.n_ionic,
            "n_aromatic": self.n_aromatic,
            "n_metal": self.n_metal,
            "n_clashes": self.n_clashes,
            "n_hbond_residues": self.n_hbond_residues,
            "n_nonpolar_residues": self.n_nonpolar_residues,
            "active_overlap": self.active_overlap,
            "active_overlap_ratio": round(self.active_overlap_ratio, 3),
            "pocket_occupancy": round(self.pocket_occupancy, 3),
            "overall": self.overall,
            "contacts": [
                {
                    "protein_atom": c.protein_atom,
                    "ligand_atom": c.ligand_atom,
                    "residue": c.residue,
                    "residue_number": c.residue_number,
                    "chain": c.chain,
                    "distance": round(c.distance, 2),
                    "interaction": c.interaction,
                }
                for c in self.contacts
            ],
        }


@dataclass
class PoseAnalysisReport:
    ligand_name: str
    pocket_id: str
    analyses: List[PoseAnalysis] = field(default_factory=list)
    ranked: List[PoseAnalysis] = field(default_factory=list)
    best_pose_id: Optional[int] = None
    disclaimer: str = (
        "Pose ranking combines docking score with geometric and chemical "
        "evidence. It does not establish experimental binding affinity."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_name": self.ligand_name,
            "pocket_id": self.pocket_id,
            "analyses": [a.to_dict() for a in self.analyses],
            "ranked": [a.to_dict() for a in self.ranked],
            "best_pose_id": self.best_pose_id,
            "disclaimer": self.disclaimer,
        }


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------

# Mirror of structure.atom_classifier sets
_OXYGEN_ATOMS = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH"}
_NITROGEN_ATOMS = {"N", "NZ", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2"}
_SULFUR_ATOMS = {"SG", "SD"}
# Matched against the RESIDUE name of metal-ion HETATM records only.
# Never against atom names: "CA" is every residue's alpha carbon, so
# matching it there counts each backbone C-alpha as a calcium ion.
_METALS = {"ZN", "MG", "MN", "FE", "CU", "CA", "CO", "NI", "NA", "K"}
_METAL_COORDINATION_DISTANCE = 3.0  # Angstrom, to a ligand N/O/S donor

# Charged sidechains for ionic interactions
_POSITIVE_RES = {"ARG", "LYS", "HIS"}
_NEGATIVE_RES = {"ASP", "GLU"}


class PoseAnalyzer:
    """Analyze a DockingResult against a reference receptor+pocket."""

    def __init__(
        self,
        contact_cutoff: float = 4.0,
        clash_cutoff: float = 2.0,
        hbond_distance: float = 3.5,
    ):
        self.contact_cutoff = contact_cutoff
        self.clash_cutoff = clash_cutoff
        self.hbond_distance = hbond_distance

    def analyze(
        self,
        result: DockingResult,
        receptor_atoms,
        pocket: Pocket,
        active_residues: Optional[List[Any]] = None,
    ) -> PoseAnalysisReport:
        report = PoseAnalysisReport(
            ligand_name=result.request.ligand.name,
            pocket_id=pocket.pocket_id,
        )
        if not result.poses:
            report.warnings = ["No poses returned by docking engine."]
            return report

        active_set = set()
        if active_residues:
            for r in active_residues:
                if isinstance(r, str):
                    active_set.add(r)
                elif isinstance(r, tuple) and len(r) == 3:
                    active_set.add(f"{r[0]}:{r[1]}{r[2]}")

        for pose in result.poses:
            pa = self._analyze_pose(
                pose, receptor_atoms, pocket, active_set,
            )
            report.analyses.append(pa)

        # Rank by combined evidence
        ranked = sorted(
            report.analyses,
            key=self._rank_key,
            reverse=True,
        )
        report.ranked = ranked
        if ranked:
            report.best_pose_id = ranked[0].pose_id
        return report

    # ------------------------------------------------------------------
    # Per-pose
    # ------------------------------------------------------------------

    def _analyze_pose(
        self,
        pose: DockedPose,
        receptor_atoms,
        pocket: Pocket,
        active_set: set,
    ) -> PoseAnalysis:
        contacts: List[ContactRecord] = []
        n_hbonds = n_hydrophobic = n_ionic = n_aromatic = n_metal = n_clashes = 0
        seen_residues: set = set()
        hbond_residues: set = set()
        nonpolar_residues: set = set()

        for la in pose.ligand.atoms:
            for ra in receptor_atoms:
                d = math.dist(
                    (la.x, la.y, la.z),
                    (ra.x, ra.y, ra.z),
                )
                if d > self.contact_cutoff:
                    continue

                interaction = self._classify_interaction(la, ra, d)
                rec = ContactRecord(
                    protein_atom=f"{ra.chain_id}:{ra.residue_name}{ra.residue_number}.{ra.atom_name}",
                    ligand_atom=f"{la.element}{la.serial}",
                    residue=ra.residue_name,
                    residue_number=ra.residue_number,
                    chain=ra.chain_id,
                    distance=round(d, 2),
                    interaction=interaction,
                )
                contacts.append(rec)

                seen_residues.add((ra.chain_id, ra.residue_name, ra.residue_number))

                if d < self.clash_cutoff and la.element != "H" and ra.element != "H":
                    n_clashes += 1
                res_key = (ra.chain_id, ra.residue_name, ra.residue_number)
                if interaction == "Hydrogen Bond":
                    n_hbonds += 1
                    hbond_residues.add(res_key)
                elif interaction == "Hydrophobic":
                    n_hydrophobic += 1
                    nonpolar_residues.add(res_key)
                elif interaction == "Ionic":
                    n_ionic += 1
                elif interaction == "Aromatic":
                    n_aromatic += 1
                    nonpolar_residues.add(res_key)
                elif interaction == "Metal Coordination":
                    n_metal += 1

        # Active-site overlap
        overlap = sum(
            1 for r in seen_residues
            if f"{r[0]}:{r[1]}{r[2]}" in active_set
        )
        overlap_ratio = (overlap / len(active_set)) if active_set else 0.0

        # Pocket occupancy = #pose-contacting residues / #pocket residues
        if pocket.residues:
            in_pocket = sum(1 for r in seen_residues if r in set(pocket.residues))
            occupancy = in_pocket / len(pocket.residues)
        else:
            occupancy = 0.0

        overall = self._classify_pose(
            score=pose.score,
            n_clashes=n_clashes,
            n_hbond_residues=len(hbond_residues),
            n_metal=n_metal,
            n_nonpolar_residues=len(nonpolar_residues),
            active_overlap=overlap,
        )

        warnings: List[str] = []
        if n_clashes > 5:
            warnings.append(f"Many steric clashes ({n_clashes}); pose is strained.")
        if overlap == 0 and active_set:
            warnings.append("Pose does not contact any predicted active-site residues.")

        return PoseAnalysis(
            pose_id=pose.pose_id,
            score=pose.score,
            rmsd_to_input=pose.rmsd_to_input,
            n_contacts=len(contacts),
            n_hbonds=n_hbonds,
            n_hydrophobic=n_hydrophobic,
            n_ionic=n_ionic,
            n_aromatic=n_aromatic,
            n_metal=n_metal,
            n_clashes=n_clashes,
            active_overlap=overlap,
            active_overlap_ratio=overlap_ratio,
            pocket_occupancy=occupancy,
            overall=overall,
            contacts=contacts,
            warnings=warnings,
            n_hbond_residues=len(hbond_residues),
            n_nonpolar_residues=len(nonpolar_residues),
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_interaction(self, lig_atom, rec_atom, distance: float) -> str:
        lig_el = lig_atom.element
        rec_name = rec_atom.atom_name
        rec_res = rec_atom.residue_name
        d = distance

        if d <= self.hbond_distance:
            if rec_name in _OXYGEN_ATOMS and lig_el in {"O", "N"}:
                return "Hydrogen Bond"
            if rec_name in _NITROGEN_ATOMS and lig_el in {"O", "N"}:
                return "Hydrogen Bond"
            if rec_name in _OXYGEN_ATOMS and lig_el in {"F"}:
                return "Hydrogen Bond"

        if (rec_res in _METALS and lig_el in {"N", "O", "S"}
                and d <= _METAL_COORDINATION_DISTANCE):
            return "Metal Coordination"

        if rec_res in _POSITIVE_RES and lig_el in {"O", "N", "S"} and d <= 4.0:
            return "Ionic"
        if rec_res in _NEGATIVE_RES and lig_el in {"N", "O"} and d <= 4.0:
            return "Ionic"

        if lig_atom.aromatic and rec_res in {"PHE", "TYR", "TRP", "HIS"} and d <= 5.0:
            return "Aromatic"

        if lig_el == "C" and not lig_atom.aromatic and rec_res in {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"} and d <= 4.5:
            return "Hydrophobic"

        if rec_name in _SULFUR_ATOMS and lig_el in {"O", "N"}:
            return "Sulfur Interaction"

        return "Contact"

    def _classify_pose(
        self,
        score: float,
        n_clashes: int,
        n_hbond_residues: int,
        n_metal: int,
        n_nonpolar_residues: int,
        active_overlap: int,
    ) -> str:
        """Count independent lines of support for the pose.

          1. favorable docking score (<= -6.0 kcal/mol — a conventional
             rule of thumb for Vina, not a calibrated cutoff)
          2. polar anchoring: coordination to a metal ion, or H-bonds
             to >= 2 DISTINCT residues
          3. nonpolar enclosure: >= 3 DISTINCT residues in hydrophobic
             or aromatic contact
          4. contact with predicted active-site residues (only when the
             caller supplies them)

        STRONG = 3+ lines, MODERATE = 2, WEAK otherwise; a strained pose
        (many clashes) is always WEAK. This labels how well-SUPPORTED the
        pose geometry looks — it is not a validated binding predictor.
        """
        if n_clashes > 5:
            return "WEAK"

        lines = 0
        if score <= -6.0:
            lines += 1
        if n_metal >= 1 or n_hbond_residues >= 2:
            lines += 1
        if n_nonpolar_residues >= 3:
            lines += 1
        if active_overlap >= 1:
            lines += 1

        if lines >= 3:
            return "STRONG"
        if lines == 2:
            return "MODERATE"
        return "WEAK"

    def _rank_key(self, pa: PoseAnalysis) -> float:
        # The engine's own score is the primary ordering. The previous
        # key added +3 per atom-PAIR H-bond contact, which let a pose
        # with a few extra contacts outrank one that was many kcal/mol
        # better — and ranked an inferior pose "best". Only clashes (a
        # strained pose) and user-supplied active-site evidence adjust it.
        return (
            -pa.score * 2
            + pa.active_overlap * 5
            - pa.n_clashes * 4
        )


# Aliases for clarity
PoseRanker = PoseAnalyzer
