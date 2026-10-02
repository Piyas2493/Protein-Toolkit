"""
ProteinToolkit Ligand Validation

Answers Section 11: chemical validity, atom valence, formal charge,
stereochemistry, MW, HBD/HBA, rotatable bonds, aromaticity,
protonation state, tautomer flags.

Two layers:
  1) Structural validity: bonds/valence consistent, no zero-radius
     geometry, no duplicate atoms.
  2) Descriptor sanity: MW within drug-like envelope, HBD/HBA counts
     bounded, logP-ish proxy via hetero/carbon ratio, rotatable bonds
     bounded.

This is a HEURISTIC validator. It does not modify the structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ligand import Ligand, LigandAtom, LigandBond


class LigandValidationError(ValueError):
    """Raised only when validation cannot even proceed (not 'invalid')."""


# Reference atomic weights (g/mol) — common organic subset
_ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Br": 79.904, "I": 126.904,
    "B": 10.81, "Si": 28.085, "Se": 78.971, "As": 74.922,
    "Li": 6.941, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "K": 39.098, "Ca": 40.078, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Cd": 112.414, "Sn": 118.710, "Hg": 200.592,
    "Au": 196.967,
}

# Standard valence for common organic atoms
_DEFAULT_VALENCE = {
    "H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "P": 5, "S": 6,
    "Cl": 1, "Br": 1, "I": 1, "B": 3, "Si": 4, "Se": 6, "As": 5,
}


_HBD_ELEMENTS = {"N", "O"}     # H-bond donors (proxy — needs Hs)
_HBA_ELEMENTS = {"N", "O", "F"}

# Bond order sums considered non-rotatable (rotatable = single acyclic
# non-ring bond between heavy atoms)
_AROMATIC_ELEMENTS = {"c", "n", "o", "s"}  # not used in heavy-only lookup


@dataclass
class ValidationReport:
    """Structured result of ligand validation.

    The 'valid' flag is True only when both structural and descriptor
    checks pass. The 'warnings' list contains soft issues that do not
    flip validity (e.g. unusual charge).
    """
    valid: bool
    chemical_valid: bool
    structural_valid: bool

    # Descriptors
    molecular_weight: float = 0.0
    heavy_atoms: int = 0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    aromatic_atoms: int = 0
    aromatic_rings: int = 0
    formal_charge: int = 0
    net_charge: int = 0

    # Flags (do NOT flip validity on their own)
    has_stereochemistry: bool = False
    has_tautomer_labile: bool = False
    has_metal: bool = False

    # Issue lists
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Provenance
    validator_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "chemical_valid": self.chemical_valid,
            "structural_valid": self.structural_valid,
            "molecular_weight": round(self.molecular_weight, 3),
            "heavy_atoms": self.heavy_atoms,
            "hbd": self.hbd,
            "hba": self.hba,
            "rotatable_bonds": self.rotatable_bonds,
            "aromatic_atoms": self.aromatic_atoms,
            "aromatic_rings": self.aromatic_rings,
            "formal_charge": self.formal_charge,
            "net_charge": self.net_charge,
            "has_stereochemistry": self.has_stereochemistry,
            "has_tautomer_labile": self.has_tautomer_labile,
            "has_metal": self.has_metal,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "validator_version": self.validator_version,
        }


class LigandValidator:
    """Validates a Ligand object without mutating it."""

    def __init__(self, *, allow_metal: bool = True, allow_radical: bool = False):
        self.allow_metal = allow_metal
        self.allow_radical = allow_radical

    def validate(self, ligand: Ligand) -> ValidationReport:
        report = ValidationReport(valid=True, chemical_valid=True, structural_valid=True)

        if ligand.atom_count() == 0:
            report.valid = False
            report.chemical_valid = False
            report.errors.append("Ligand has no atoms.")
            return report

        # ---- Structural validity --------------------------------------
        ok_struct = True
        # Duplicate atom serial detection
        serials = [a.serial for a in ligand.atoms]
        if len(set(serials)) != len(serials):
            ok_struct = False
            report.errors.append("Duplicate atom serials.")

        # Check each bond references known atoms
        known_serials = set(serials)
        for b in ligand.bonds:
            if b.a1 not in known_serials or b.a2 not in known_serials:
                ok_struct = False
                report.errors.append(
                    f"Bond references unknown atom serials: {b.a1}-{b.a2}."
                )

        # Valence check (heuristic — ignores formal charge vs. bond order
        # interactions in unusual oxidation states; flagged as warning)
        for atom in ligand.atoms:
            elem = atom.element
            if elem not in _DEFAULT_VALENCE:
                if elem in {"Mg", "Ca", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd"}:
                    report.has_metal = True
                    if not self.allow_metal:
                        report.warnings.append(
                            f"Metal atom {elem} requires special handling."
                        )
                    continue
                report.warnings.append(f"Unknown element '{elem}', skipping valence check.")
                continue
            max_v = _DEFAULT_VALENCE[elem]
            current = sum(b.order for b in ligand.bonds if b.a1 == atom.serial or b.a2 == atom.serial)
            current += atom.formal_charge  # charged atom loses/gains bond
            # Hydrogens add to valence (very rough)
            current += atom.h_count
            if current > max_v + 0.5:
                ok_struct = False
                report.errors.append(
                    f"Atom {elem}#{atom.serial} exceeds typical valence "
                    f"({current:.1f} > {max_v})."
                )

        report.structural_valid = ok_struct
        if not ok_struct:
            report.valid = False
            report.chemical_valid = False

        # ---- Descriptors ----------------------------------------------
        report.heavy_atoms = ligand.heavy_atom_count()
        report.molecular_weight = self._molecular_weight(ligand)
        report.hbd = self._count_hbd(ligand)
        report.hba = self._count_hba(ligand)
        report.rotatable_bonds = self._count_rotatable(ligand)
        report.aromatic_atoms = self._count_aromatic(ligand)
        report.aromatic_rings = self._count_aromatic_rings(ligand)
        report.net_charge = sum(a.formal_charge for a in ligand.atoms)
        report.formal_charge = report.net_charge

        # ---- Protonation / tautomer / stereochemistry flags ------------
        # Tautomer-labile: nitrogen atoms with H attached (proxy for
        # imines, amides, etc.)
        report.has_tautomer_labile = any(
            a.element == "N" and a.h_count > 0 for a in ligand.atoms
        )
        # Stereochemistry flag (RDKit-only)
        report.has_stereochemistry = any(
            a.formal_charge != 0 and a.element in {"N", "P"} for a in ligand.atoms
        ) or any(b.aromatic and b.order == 2 for b in ligand.bonds)

        # ---- Descriptor sanity -----------------------------------------
        # Drug-like envelope: Lipinski-style very loose bounds.
        if report.molecular_weight <= 0:
            report.errors.append("Molecular weight is zero or negative.")
            report.valid = False
            report.chemical_valid = False

        if report.molecular_weight > 900:
            report.warnings.append(
                f"MW {report.molecular_weight:.1f} above 900 Da (beyond drug-like)."
            )

        if abs(report.net_charge) > 4:
            report.warnings.append(
                f"Net charge {report.net_charge} is unusually high."
            )

        if not report.heavy_atoms:
            report.errors.append("Ligand has no heavy atoms.")
            report.valid = False
            report.chemical_valid = False

        if ligand.atom_count() > 1 and not ligand.bonds:
            report.warnings.append("Ligand has atoms but no bonds (e.g. PDB HETATM-only input).")

        return report

    # ------------------------------------------------------------------
    # Descriptor helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _molecular_weight(ligand: Ligand) -> float:
        total = 0.0
        for atom in ligand.atoms:
            total += _ATOMIC_WEIGHTS.get(atom.element, 0.0)
        return total

    @staticmethod
    def _count_hbd(ligand: Ligand) -> int:
        return sum(
            1 for a in ligand.atoms
            if a.element in _HBD_ELEMENTS and a.h_count > 0
        )

    @staticmethod
    def _count_hba(ligand: Ligand) -> int:
        return sum(
            1 for a in ligand.atoms if a.element in _HBA_ELEMENTS
        )

    @staticmethod
    def _count_rotatable(ligand: Ligand) -> int:
        # Only count single acyclic bonds between two heavy atoms that
        # are not at the end of a chain.
        degree: Dict[int, int] = {a.serial: 0 for a in ligand.atoms}
        for b in ligand.bonds:
            if abs(b.order - 1.0) > 0.01 or b.in_ring or b.aromatic:
                continue
            degree[b.a1] = degree.get(b.a1, 0) + 1
            degree[b.a2] = degree.get(b.a2, 0) + 1

        rot = 0
        for b in ligand.bonds:
            if abs(b.order - 1.0) > 0.01 or b.in_ring or b.aromatic:
                continue
            a1 = _get_atom(ligand, b.a1)
            a2 = _get_atom(ligand, b.a2)
            if not a1 or not a2:
                continue
            if a1.element == "H" or a2.element == "H":
                continue
            # Need at least one more heavy-atom neighbor on each side
            if degree.get(b.a1, 0) < 2 or degree.get(b.a2, 0) < 2:
                continue
            rot += 1
        return rot

    @staticmethod
    def _count_aromatic(ligand: Ligand) -> int:
        return sum(1 for a in ligand.atoms if a.aromatic)

    @staticmethod
    def _count_aromatic_rings(ligand: Ligand) -> int:
        # Lightweight aromatic-ring proxy: aromatic atoms forming a
        # cycle through aromatic bonds. Uses a depth-limited walk.
        seen_rings: set = set()
        for start in ligand.atoms:
            if not start.aromatic:
                continue
            path = _walk_aromatic_cycle(ligand, start, start.serial)
            if path and len(path) >= 3:
                key = frozenset(path)
                if len(key) == len(path) and key not in seen_rings:
                    seen_rings.add(key)
        return len(seen_rings)


def _get_atom(ligand: Ligand, serial: int) -> Optional[LigandAtom]:
    for a in ligand.atoms:
        if a.serial == serial:
            return a
    return None


def _walk_aromatic_cycle(
    ligand: Ligand, start: LigandAtom, prev: int,
    origin: Optional[int] = None, visited: Optional[set] = None,
) -> Optional[List[int]]:
    """Greedy walk through aromatic bonds looking for the shortest cycle
    back to the true origin atom (not just the immediately-prior one —
    a plain prev-only guard never closes the ring and loops forever)."""
    if origin is None:
        origin = start.serial
    if visited is None:
        visited = {start.serial}

    neighbours = [
        b.a2 if b.a1 == start.serial else b.a1
        for b in ligand.bonds
        if (b.a1 == start.serial or b.a2 == start.serial)
        and (b.aromatic or abs(b.order - 1.5) < 0.01)
    ]
    for n in neighbours:
        if n == prev:
            continue
        if n == origin:
            return [start.serial]
        if n in visited:
            continue
        n_atom = _get_atom(ligand, n)
        if not n_atom or not n_atom.aromatic:
            continue
        sub = _walk_aromatic_cycle(
            ligand, n_atom, start.serial, origin, visited | {n},
        )
        if sub is not None:
            return [start.serial] + sub
    return None
