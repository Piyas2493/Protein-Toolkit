"""
Pocket detection and ranking (Sprint 6, Section 9).

Two sources of pockets:
  1. "Crystallographic" pockets — the ligands already bound in the
     structure define a binding pocket. Each HETATM ligand's
     contacting residues (cutoff) form one pocket.
  2. "Cavity" pockets — fallback: alpha-sphere-ish grid sampling
     around the receptor when no HETATM is present. We implement a
     simplified alpha-sphere proxy: scan a coarse grid of points and
     keep the points where many atoms are at a moderate distance
     (4-6 Å from a few residues) but none within 2.5 Å (i.e. surface
     cavities, not buried atoms).

Each Pocket becomes a candidate for compatibility assessment and the
source of a docking box (Section 18).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from structure.contacts import ContactAnalyzer
from structure.models import Atom, ProteinStructure

from .preparation import PreparedReceptor


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class BindingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
            "center_z": round(self.center_z, 3),
            "size_x": round(self.size_x, 3),
            "size_y": round(self.size_y, 3),
            "size_z": round(self.size_z, 3),
        }


@dataclass
class Pocket:
    pocket_id: str
    source: str                         # "crystallographic" | "cavity"
    ligand_resname: Optional[str]       # for crystallographic pockets
    residues: List[Tuple[str, str, int]] = field(default_factory=list)  # (chain, resname, resnum)
    contact_count: int = 0
    avg_distance: float = 0.0
    hydrophobic_residues: int = 0
    aromatic_residues: int = 0
    charged_residues: int = 0
    polar_residues: int = 0
    metal_residues: int = 0
    estimated_volume: float = 0.0
    estimated_depth: float = 0.0
    box: Optional[BindingBox] = None
    score: float = 0.0

    def residue_set(self) -> set:
        return set(self.residues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pocket_id": self.pocket_id,
            "source": self.source,
            "ligand_resname": self.ligand_resname,
            "n_residues": len(self.residues),
            "residues": [
                f"{c}:{r}{n}" for (c, r, n) in self.residues
            ],
            "contact_count": self.contact_count,
            "avg_distance": round(self.avg_distance, 2),
            "hydrophobic_residues": self.hydrophobic_residues,
            "aromatic_residues": self.aromatic_residues,
            "charged_residues": self.charged_residues,
            "polar_residues": self.polar_residues,
            "metal_residues": self.metal_residues,
            "estimated_volume": round(self.estimated_volume, 1),
            "estimated_depth": round(self.estimated_depth, 2),
            "box": self.box.to_dict() if self.box else None,
            "score": round(self.score, 2),
        }


# ----------------------------------------------------------------------
# Residue categorization
# ----------------------------------------------------------------------

_HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"}
_AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
_CHARGED_POS = {"ARG", "LYS"}
_CHARGED_NEG = {"ASP", "GLU"}
_POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS"}
_METAL = {"ZN", "MG", "MN", "FE", "CU", "CA", "CO", "NI"}


# ----------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------

class PocketDetector:
    """Detect and rank binding pockets of a prepared receptor."""

    def __init__(self, cutoff: float = 5.0, grid_step: float = 1.5):
        self.cutoff = cutoff
        self.grid_step = grid_step
        self.contact_analyzer = ContactAnalyzer(cutoff=cutoff)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, receptor: PreparedReceptor) -> List[Pocket]:
        pockets: List[Pocket] = []

        # 1) Crystallographic pockets from bound ligands (non-water)
        for idx, lig in enumerate(receptor.structure.ligands):
            if lig.residue_name in {"HOH", "WAT", "DOD"}:
                continue
            if receptor.selected_chain is not None and lig.chain_id != receptor.selected_chain:
                continue
            pocket = self._build_crystallographic_pocket(receptor, lig, idx)
            if pocket and pocket.residues:
                pockets.append(pocket)

        # 2) Cavity pockets (fallback when no HETATM)
        if not pockets:
            cavities = self._detect_cavities(receptor)
            pockets.extend(cavities)

        # Score and rank
        for p in pockets:
            self._score(p)
        pockets.sort(key=lambda p: p.score, reverse=True)
        # Assign rank IDs
        for i, p in enumerate(pockets, start=1):
            p.pocket_id = f"P{i}"
        return pockets

    # ------------------------------------------------------------------
    # Crystallographic
    # ------------------------------------------------------------------

    def _build_crystallographic_pocket(
        self,
        receptor: PreparedReceptor,
        lig: Any,
        idx: int,
    ) -> Optional[Pocket]:
        # Find residues of the receptor near the ligand
        residues_set: set = set()
        contacts: List[Dict[str, Any]] = []

        receptor_atom_by_serial: Dict[int, Atom] = {
            a.serial: a for a in receptor.receptor_atoms
        }
        for la in lig.atoms:
            for ra in receptor.receptor_atoms:
                d = math.dist(
                    (la.x, la.y, la.z),
                    (ra.x, ra.y, ra.z),
                )
                if d <= self.cutoff:
                    residues_set.add((ra.chain_id, ra.residue_name, ra.residue_number))
                    contacts.append({
                        "ligand": lig.residue_name,
                        "residue": ra.residue_name,
                        "residue_number": ra.residue_number,
                        "chain": ra.chain_id,
                        "distance": d,
                    })

        if not residues_set:
            return None

        # Build box from the ligand atoms
        xs = [a.x for a in lig.atoms]
        ys = [a.y for a in lig.atoms]
        zs = [a.z for a in lig.atoms]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        cz = (min(zs) + max(zs)) / 2
        size_x = max(8.0, (max(xs) - min(xs)) + 6.0)
        size_y = max(8.0, (max(ys) - min(ys)) + 6.0)
        size_z = max(8.0, (max(zs) - min(zs)) + 6.0)

        # Categorise residues
        pocket = Pocket(
            pocket_id=f"X{idx}",
            source="crystallographic",
            ligand_resname=lig.residue_name,
            residues=sorted(residues_set),
        )
        pocket.contact_count = len(contacts)
        if contacts:
            pocket.avg_distance = sum(c["distance"] for c in contacts) / len(contacts)
        pocket.box = BindingBox(cx, cy, cz, size_x, size_y, size_z)

        for (_, rname, _) in pocket.residues:
            if rname in _HYDROPHOBIC:
                pocket.hydrophobic_residues += 1
            if rname in _AROMATIC:
                pocket.aromatic_residues += 1
            if rname in _CHARGED_POS or rname in _CHARGED_NEG:
                pocket.charged_residues += 1
            if rname in _POLAR:
                pocket.polar_residues += 1
            if rname in _METAL:
                pocket.metal_residues += 1

        pocket.estimated_volume = size_x * size_y * size_z
        pocket.estimated_depth = min(size_x, size_y, size_z) / 2.0
        return pocket

    # ------------------------------------------------------------------
    # Cavity (fallback)
    # ------------------------------------------------------------------

    def _detect_cavities(self, receptor: PreparedReceptor) -> List[Pocket]:
        if not receptor.receptor_atoms:
            return []

        xs = [a.x for a in receptor.receptor_atoms]
        ys = [a.y for a in receptor.receptor_atoms]
        zs = [a.z for a in receptor.receptor_atoms]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        minz, maxz = min(zs), max(zs)

        # Coarse grid; sample at grid_step Å
        step = self.grid_step
        atom_coords = [(a.x, a.y, a.z) for a in receptor.receptor_atoms]

        cavity_centers: List[Tuple[float, float, float]] = []
        for x in _frange(minx, maxx, step):
            for y in _frange(miny, maxy, step):
                for z in _frange(minz, maxz, step):
                    near = 0
                    inside = False
                    for (ax, ay, az) in atom_coords:
                        d = math.dist((x, y, z), (ax, ay, az))
                        if d < 2.5:
                            inside = True
                            break
                        if 4.0 <= d <= 6.0:
                            near += 1
                    if not inside and near >= 8:
                        cavity_centers.append((x, y, z))

        # Cluster cavity centers (simple greedy)
        clusters: List[List[Tuple[float, float, float]]] = []
        for c in cavity_centers:
            placed = False
            for cluster in clusters:
                if any(math.dist(c, o) < 4.0 for o in cluster):
                    cluster.append(c)
                    placed = True
                    break
            if not placed:
                clusters.append([c])

        pockets: List[Pocket] = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 4:
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            cz = sum(p[2] for p in cluster) / len(cluster)
            size = 12.0
            box = BindingBox(cx, cy, cz, size, size, size)
            pocket = Pocket(
                pocket_id=f"C{i}",
                source="cavity",
                ligand_resname=None,
                residues=[],
                box=box,
                estimated_volume=size ** 3,
                estimated_depth=size / 2,
            )
            pockets.append(pocket)
        return pockets

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, pocket: Pocket) -> None:
        score = 0.0
        score += min(len(pocket.residues) * 5, 40)
        score += min(pocket.contact_count, 30)
        score += min(pocket.hydrophobic_residues * 3, 20)
        score += min(pocket.aromatic_residues * 4, 20)
        score += min(pocket.charged_residues * 3, 15)
        if pocket.metal_residues:
            score += 10
        if pocket.source == "crystallographic":
            score += 15
        pocket.score = min(score, 100.0)


def _frange(start: float, stop: float, step: float):
    x = start
    while x <= stop:
        yield x
        x += step
