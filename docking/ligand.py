"""
Docking-side ligand helpers.

The actual Ligand object lives in `ligand/`. This module provides
geometry helpers specific to docking: volume estimation, steric-clash
counts against a pocket, hydrogen-bond donor/acceptor positions in
3D, and aromatic centroid computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from ligand import Ligand


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def _bbox_volume(ligand: Ligand, padding: float = 0.0) -> float:
    minx, miny, minz, maxx, maxy, maxz = ligand.bounding_box()
    return (
        (maxx - minx + 2 * padding)
        * (maxy - miny + 2 * padding)
        * (maxz - minz + 2 * padding)
    )


def ligand_bounding_volume(ligand: Ligand, padding: float = 0.5) -> float:
    """Bounding-box volume in Å³ with a small padding for hydrogen extent."""
    return _bbox_volume(ligand, padding=padding)


def ligand_radius_of_gyration(ligand: Ligand) -> float:
    if not ligand.atoms:
        return 0.0
    cx, cy, cz = ligand.center()
    total = 0.0
    for a in ligand.atoms:
        d2 = (a.x - cx) ** 2 + (a.y - cy) ** 2 + (a.z - cz) ** 2
        total += d2
    return math.sqrt(total / len(ligand.atoms))


def centroid_of_element(ligand: Ligand, element: str) -> Tuple[float, float, float]:
    """Return the centroid of all atoms of a given element (or 0,0,0)."""
    matched = [a for a in ligand.atoms if a.element == element]
    if not matched:
        return (0.0, 0.0, 0.0)
    n = len(matched)
    return (
        sum(a.x for a in matched) / n,
        sum(a.y for a in matched) / n,
        sum(a.z for a in matched) / n,
    )


def aromatic_centroid(ligand: Ligand) -> Tuple[float, float, float]:
    atoms = [a for a in ligand.atoms if a.aromatic]
    if not atoms:
        return (0.0, 0.0, 0.0)
    n = len(atoms)
    return (
        sum(a.x for a in atoms) / n,
        sum(a.y for a in atoms) / n,
        sum(a.z for a in atoms) / n,
    )


# ----------------------------------------------------------------------
# Chemistry helpers
# ----------------------------------------------------------------------

@dataclass
class ChemistryProfile:
    hbd: int
    hba: int
    aromatic_atoms: int
    formal_charge: int
    rotatable_bonds: int
    heavy_atoms: int


def profile_ligand(ligand: Ligand) -> ChemistryProfile:
    return ChemistryProfile(
        hbd=sum(1 for a in ligand.atoms if a.element in {"N", "O"} and a.h_count > 0),
        hba=sum(1 for a in ligand.atoms if a.element in {"N", "O", "F"}),
        aromatic_atoms=sum(1 for a in ligand.atoms if a.aromatic),
        formal_charge=sum(a.formal_charge for a in ligand.atoms),
        rotatable_bonds=ligand.bond_count(),  # upper bound; refined by validator
        heavy_atoms=ligand.heavy_atom_count(),
    )


# ----------------------------------------------------------------------
# 3D H-bond / donor positions
# ----------------------------------------------------------------------

def hbd_positions(ligand: Ligand) -> List[Tuple[float, float, float]]:
    """Approximate H-bond donor positions (N/O with implicit H)."""
    return [
        (a.x, a.y, a.z)
        for a in ligand.atoms
        if a.element in {"N", "O"} and a.h_count > 0
    ]


def hba_positions(ligand: Ligand) -> List[Tuple[float, float, float]]:
    """Approximate H-bond acceptor positions (N/O/F lone pairs)."""
    return [
        (a.x, a.y, a.z)
        for a in ligand.atoms
        if a.element in {"N", "O", "F"}
    ]


def hydrophobic_positions(ligand: Ligand) -> List[Tuple[float, float, float]]:
    return [
        (a.x, a.y, a.z)
        for a in ligand.atoms
        if a.element in {"C"} and not a.aromatic
    ]
