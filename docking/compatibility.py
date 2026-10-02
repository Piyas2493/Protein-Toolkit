"""
Ligand-pocket compatibility engine (Sprint 6 / Sections 14-16).

Pre-docking assessment. Answers:

    "Does this ligand appear structurally compatible with any
     candidate binding pocket of the target enzyme?"

The output is qualitative (HIGH / MEDIUM / LOW), not a probability.
It MUST NOT be presented as experimental validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ligand import Ligand

from .pocket import Pocket, BindingBox
from .ligand import (
    ligand_bounding_volume,
    ligand_radius_of_gyration,
    profile_ligand,
    ChemistryProfile,
    hba_positions,
    hbd_positions,
    aromatic_centroid,
    hydrophobic_positions,
)


# ----------------------------------------------------------------------
# Output containers
# ----------------------------------------------------------------------

@dataclass
class PocketCompatibility:
    pocket_id: str
    pocket: Pocket
    geometry_score: float           # 0-100
    chemistry_score: float          # 0-100
    steric_clash_risk: str          # LOW / MEDIUM / HIGH
    overall: str                    # HIGH / MEDIUM / LOW / INCOMPATIBLE
    pocket_compatibility: str       # HIGH / MEDIUM / LOW
    chemical_compatibility: str
    active_site_overlap: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pocket_id": self.pocket_id,
            "geometry_score": round(self.geometry_score, 2),
            "chemistry_score": round(self.chemistry_score, 2),
            "steric_clash_risk": self.steric_clash_risk,
            "overall": self.overall,
            "pocket_compatibility": self.pocket_compatibility,
            "chemical_compatibility": self.chemical_compatibility,
            "active_site_overlap": self.active_site_overlap,
            "details": self.details,
            "warnings": self.warnings,
        }


@dataclass
class CompatibilityReport:
    ligand_name: str
    ligand_profile: ChemistryProfile
    pockets: List[PocketCompatibility] = field(default_factory=list)
    best_pocket_id: Optional[str] = None
    overall_assessment: str = "INCOMPATIBLE"
    warnings: List[str] = field(default_factory=list)
    disclaimer: str = (
        "This is a pre-docking structural compatibility estimate. "
        "It does NOT establish experimental binding or activity."
    )

    # Lowercase alias used by tests
    @property
    def computational_disclaimer(self) -> str:
        return "This is a computational prediction that does not establish experimental binding."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_name": self.ligand_name,
            "ligand_profile": {
                "hbd": self.ligand_profile.hbd,
                "hba": self.ligand_profile.hba,
                "aromatic_atoms": self.ligand_profile.aromatic_atoms,
                "formal_charge": self.ligand_profile.formal_charge,
                "rotatable_bonds": self.ligand_profile.rotatable_bonds,
                "heavy_atoms": self.ligand_profile.heavy_atoms,
            },
            "best_pocket_id": self.best_pocket_id,
            "overall_assessment": self.overall_assessment,
            "pockets": [p.to_dict() for p in self.pockets],
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
            "computational_disclaimer": self.computational_disclaimer,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _classify(score: float) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def _classify_clash(risk: float) -> str:
    if risk >= 0.7:
        return "HIGH"
    if risk >= 0.4:
        return "MEDIUM"
    return "LOW"


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class CompatibilityEngine:
    """Assess ligand-pocket compatibility.

    Configurable knobs:
        clash_distance: minimum heavy-atom distance to allow
        max_oversize:   max ratio of ligand volume to pocket volume
        hb_distance_cut: H-bond distance threshold (Å)
        active_overlap_min: min # overlapping residues for HIGH
    """

    def __init__(
        self,
        clash_distance: float = 2.0,
        max_oversize: float = 1.5,
        hb_distance_cut: float = 4.0,
        active_overlap_min: int = 3,
    ):
        self.clash_distance = clash_distance
        self.max_oversize = max_oversize
        self.hb_distance_cut = hb_distance_cut
        self.active_overlap_min = active_overlap_min

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def assess(
        self,
        ligand: Ligand,
        pockets: List[Pocket],
        predicted_active_residues: Optional[List[Any]] = None,
    ) -> CompatibilityReport:
        profile = profile_ligand(ligand)
        report = CompatibilityReport(
            ligand_name=ligand.name,
            ligand_profile=profile,
        )

        if not pockets:
            report.warnings.append("No pockets available for compatibility assessment.")
            report.overall_assessment = "INCOMPATIBLE"
            return report

        # Ligand geometry
        lig_vol = ligand_bounding_volume(ligand)
        lig_rg = ligand_radius_of_gyration(ligand)
        lig_centroid = ligand.center()

        active_residues = set()
        if predicted_active_residues:
            for r in predicted_active_residues:
                # accept dicts, tuples, or strings
                if isinstance(r, str):
                    active_residues.add(r)
                elif isinstance(r, tuple) and len(r) == 3:
                    active_residues.add(f"{r[0]}:{r[1]}{r[2]}")

        # Per-pocket assessment
        for p in pockets:
            compat = self._assess_one(
                ligand, profile, p, lig_vol, lig_rg,
                active_residues,
            )
            report.pockets.append(compat)

        # Rank
        ranked = sorted(
            report.pockets,
            key=lambda c: c.geometry_score * 0.45
                          + c.chemistry_score * 0.45
                          + (100 if c.active_site_overlap == "HIGH" else 50
                             if c.active_site_overlap == "MEDIUM" else 0) * 0.10,
            reverse=True,
        )

        # Reorder pockets list by rank for downstream consumption
        report.pockets = ranked
        if ranked:
            report.best_pocket_id = ranked[0].pocket_id
            best = ranked[0]
            if best.overall == "HIGH":
                report.overall_assessment = "HIGH-PRIORITY DOCKING CANDIDATE"
            elif best.overall == "MEDIUM":
                report.overall_assessment = "MODERATE-PRIORITY CANDIDATE"
            elif best.overall == "LOW":
                report.overall_assessment = "LOW-PRIORITY CANDIDATE"
            else:
                report.overall_assessment = "INCOMPATIBLE"

        return report

    # ------------------------------------------------------------------
    # Per-pocket
    # ------------------------------------------------------------------

    def _assess_one(
        self,
        ligand: Ligand,
        profile: ChemistryProfile,
        pocket: Pocket,
        lig_vol: float,
        lig_rg: float,
        active_residues: set,
    ) -> PocketCompatibility:
        details: Dict[str, Any] = {}
        warnings: List[str] = []

        # ---- Geometry -----------------------------------------------
        pocket_vol = pocket.estimated_volume or 1.0
        ratio = lig_vol / pocket_vol if pocket_vol > 0 else 0.0
        details["ligand_volume"] = round(lig_vol, 1)
        details["pocket_volume"] = round(pocket_vol, 1)
        details["volume_ratio"] = round(ratio, 3)

        geom_score = 100.0
        # Penalise over-sized ligands
        if ratio > self.max_oversize:
            geom_score -= min(80, (ratio - self.max_oversize) * 100)
            warnings.append(
                f"Ligand bounding volume ({lig_vol:.0f} Å³) "
                f"exceeds pocket ({pocket_vol:.0f} Å³)."
            )
        # Penalise extremely small ligands in a large pocket
        if ratio < 0.05:
            geom_score -= 30
            warnings.append("Ligand is much smaller than pocket — weak complementarity.")
        # Penalise if ligand is too large in smallest dimension
        if pocket.box:
            cx, cy, cz = pocket.box.center_x, pocket.box.center_y, pocket.box.center_z
            sx, sy, sz = pocket.box.size_x, pocket.box.size_y, pocket.box.size_z
            details["distance_to_pocket_center"] = round(
                math.dist(ligand.center(), (cx, cy, cz)), 2
            )

        # Steric clash risk: rough — count heavy-atom pairs inside box
        # that fall below clash_distance from pocket residues (we do
        # not have residue coords here, so we approximate via ligand
        # bbox vs box)
        clash_risk = self._approx_clash_risk(ligand, pocket)
        details["clash_risk_value"] = round(clash_risk, 3)
        clash_class = _classify_clash(clash_risk)
        if clash_class == "HIGH":
            geom_score -= 25

        geom_score = max(0.0, min(100.0, geom_score))

        # ---- Chemistry ----------------------------------------------
        chem_score = 100.0
        hb_score = self._hydrogen_bond_score(ligand, pocket)
        chem_score = 0.6 * chem_score + 0.4 * hb_score

        # Hydrophobic complementarity
        hp_match = self._hydrophobic_match(ligand, pocket)
        chem_score = 0.7 * chem_score + 0.3 * hp_match

        # Aromatic compatibility
        if profile.aromatic_atoms > 0 and pocket.aromatic_residues > 0:
            chem_score += 5
        if profile.aromatic_atoms > 0 and pocket.aromatic_residues == 0:
            chem_score -= 10

        # Charge compatibility (very rough — uses pocket charge state)
        details["pocket_charged_residues"] = pocket.charged_residues
        if abs(profile.formal_charge) > 0 and pocket.charged_residues == 0:
            chem_score -= 15
            warnings.append(
                "Ligand carries a net charge but pocket has no charged residues."
            )

        # Metal coordination
        if any(a.element in {"O", "N", "S"} for a in ligand.atoms) and pocket.metal_residues:
            chem_score += 5

        chem_score = max(0.0, min(100.0, chem_score))

        # ---- Active-site overlap ------------------------------------
        overlap_count = sum(
            1 for r in pocket.residues
            if f"{r[0]}:{r[1]}{r[2]}" in active_residues
        ) if active_residues else 0
        if active_residues:
            details["active_overlap_count"] = overlap_count
            details["active_overlap_total"] = len(active_residues)
            if overlap_count >= self.active_overlap_min:
                active_overlap = "HIGH"
            elif overlap_count >= 1:
                active_overlap = "MEDIUM"
            else:
                active_overlap = "LOW"
        else:
            active_overlap = "UNKNOWN"

        # ---- Overall classification ----------------------------------
        pocket_compat = _classify(geom_score * 0.55 + chem_score * 0.45)
        if pocket_compat == "LOW" and clash_class == "HIGH":
            overall = "INCOMPATIBLE"
        else:
            overall = pocket_compat

        # Adjust overall by active-site overlap
        if active_overlap == "HIGH":
            if overall == "MEDIUM":
                overall = "HIGH"
        elif active_overlap == "LOW" and overall == "HIGH":
            overall = "MEDIUM"
            warnings.append(
                "Pocket chemistry matches but active-site overlap is low."
            )

        return PocketCompatibility(
            pocket_id=pocket.pocket_id,
            pocket=pocket,
            geometry_score=geom_score,
            chemistry_score=chem_score,
            steric_clash_risk=clash_class,
            overall=overall,
            pocket_compatibility=pocket_compat,
            chemical_compatibility=_classify(chem_score),
            active_site_overlap=active_overlap,
            details=details,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    def _hydrogen_bond_score(self, ligand: Ligand, pocket: Pocket) -> float:
        # Score proportional to presence of donor/acceptor atoms on both
        # sides, capped at 100.
        n_donor = len(hbd_positions(ligand))
        n_acceptor = len(hba_positions(ligand))
        n_polar_pocket = pocket.polar_residues
        if n_polar_pocket == 0:
            return 50.0  # neutral
        total = n_donor + n_acceptor
        if total == 0:
            return 60.0
        # Reward 10 points per polar contact up to 10
        return min(100.0, 30 + total * 8 + n_polar_pocket * 2)

    def _hydrophobic_match(self, ligand: Ligand, pocket: Pocket) -> float:
        n_hp = len(hydrophobic_positions(ligand))
        if n_hp == 0:
            return 60.0
        if pocket.hydrophobic_residues == 0:
            return 40.0
        # Reward up to 100 when both sides have many hydrophobic atoms
        return min(100.0, 40 + min(n_hp, pocket.hydrophobic_residues) * 6)

    def _approx_clash_risk(self, ligand: Ligand, pocket: Pocket) -> float:
        """Cheap proxy for steric clash risk using bounding-box overlap.

        Returns a value in [0, 1] where 1 means severe expected clash.
        """
        if not pocket.box:
            return 0.0
        minx, miny, minz, maxx, maxy, maxz = ligand.bounding_box()
        cx, cy, cz = pocket.box.center_x, pocket.box.center_y, pocket.box.center_z
        sx, sy, sz = pocket.box.size_x, pocket.box.size_y, pocket.box.size_z

        pocket_minx = cx - sx / 2
        pocket_miny = cy - sy / 2
        pocket_minz = cz - sz / 2
        pocket_maxx = cx + sx / 2
        pocket_maxy = cy + sy / 2
        pocket_maxz = cz + sz / 2

        # How much of the ligand lies outside the pocket box?
        outside = max(0.0, pocket_minx - minx) + max(0.0, maxx - pocket_maxx)
        outside += max(0.0, pocket_miny - miny) + max(0.0, maxy - pocket_maxy)
        outside += max(0.0, pocket_minz - minz) + max(0.0, maxz - pocket_maxz)
        # Normalise by box size
        box_diag = sx + sy + sz
        if box_diag <= 0:
            return 0.0
        return min(1.0, outside / (box_diag * 0.5))
