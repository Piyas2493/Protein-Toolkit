"""
Spatial catalytic-site detection (structure-aware).

core/catalytic.py flags a catalytic system whenever the required residue
TYPES occur often enough anywhere in the sequence — a composition check,
not a position check. When a real 3D structure is available, this module
checks what sequence alone cannot: whether specific residue INSTANCES of
the required types are actually close together in space, which is what
a real catalytic site requires. A true triad/dyad found this way is much
stronger evidence than the sequence-only composition match.

# ponytail: residue position = centroid of its own atoms, not the true
# side-chain functional-group tip. Good enough to separate "plausible
# triad" from "these residues are on opposite sides of the protein";
# swap in side-chain-atom distances if false positives show up on real
# validated triads.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List

from .contacts import ContactAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "database" / "catalytic_sites.json"

PROXIMITY_CUTOFF = 8.0  # Angstrom, residue-centroid to residue-centroid

# ponytail: safety cap on combinations tried per rule, global lock style —
# bail out (treat as "not found") rather than hang on a pathological
# multimer with thousands of copies of one residue type.
MAX_COMBINATIONS = 200_000


class _Residue:
    __slots__ = ("chain_id", "residue_number", "residue_name", "x", "y", "z")

    def __init__(self, chain_id, residue_number, residue_name, atoms):
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.residue_name = residue_name
        n = len(atoms)
        self.x = sum(a.x for a in atoms) / n
        self.y = sum(a.y for a in atoms) / n
        self.z = sum(a.z for a in atoms) / n

    def label(self) -> str:
        return f"{self.chain_id}:{self.residue_name}{self.residue_number}"


def _group_residues(structure) -> List[_Residue]:
    groups: Dict[tuple, list] = {}
    names: Dict[tuple, str] = {}
    for atom in structure.atoms:
        key = (atom.chain_id, atom.residue_number)
        groups.setdefault(key, []).append(atom)
        names[key] = atom.residue_name
    return [
        _Residue(chain_id, number, names[(chain_id, number)], atoms)
        for (chain_id, number), atoms in groups.items()
    ]


class SpatialCatalyticDetector:
    """Checks catalytic_sites.json rules against real 3D residue positions."""

    def __init__(self, cutoff: float = PROXIMITY_CUTOFF):
        with open(DATABASE, encoding="utf-8") as f:
            self.rules = json.load(f)
        self.cutoff = cutoff

    def predict(self, structure) -> List[dict]:
        residues = _group_residues(structure)
        by_type: Dict[str, List[_Residue]] = {}
        for r in residues:
            by_type.setdefault(r.residue_name, []).append(r)

        hits = []
        for rule in self.rules:
            required = Counter(rule["residues"])

            if any(len(by_type.get(t, [])) < n for t, n in required.items()):
                continue

            site = self._find_clustered_instances(required, by_type)
            if site is not None:
                hits.append({
                    **rule,
                    "residues_found": [r.label() for r in site],
                })

        return hits

    def _find_clustered_instances(self, required, by_type):
        """Try each distinct way to pick the required residue instances
        (distinct instances for a repeated type, e.g. two different ASPs)
        and return the first set where every pair is within cutoff."""
        per_type_choices = [
            list(combinations(by_type[residue_type], count))
            for residue_type, count in required.items()
        ]

        total = 1
        for choices in per_type_choices:
            total *= max(len(choices), 1)
        if total > MAX_COMBINATIONS:
            return None

        for combo in product(*per_type_choices):
            group = [r for part in combo for r in part]
            if self._all_within_cutoff(group):
                return group
        return None

    def _all_within_cutoff(self, residues) -> bool:
        return all(
            ContactAnalyzer.distance(a, b) <= self.cutoff
            for a, b in combinations(residues, 2)
        )
