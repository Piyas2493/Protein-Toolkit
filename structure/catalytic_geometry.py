"""
Spatial catalytic-site detection (structure-aware).

core/catalytic.py names a catalytic system from a conserved active-site motif
in the SEQUENCE. This module asks what sequence alone cannot: are the actual
catalytic residues of that system close together in 3D, the way a working site
requires? A true triad/dyad found this way is stronger evidence than a motif.

Two modes (`strict`, the default, and the older geometry-only `strict=False`):

strict    a system is reported only if its motif is present in the chain's
          sequence, and the residues considered are the ones the motif marks as
          catalytic (e.g. the Ser inside G-D-S-G-G, not any serine nearby). Their
          distance is measured between side-chain FUNCTIONAL atoms (Ser OG, Cys
          SG, His ND1/NE2, Asp OD1/OD2, Glu OE1/OE2). The residues need only form a
          CONNECTED cluster (each within FUNCTIONAL_CUTOFF of another), because a
          triad is a chain, Ser-His-Asp, not a clique: Ser and Asp can be 6-8 A apart.
legacy    any residues of the required TYPES whose centroids lie within
          PROXIMITY_CUTOFF. It named the wrong serine in every serine protease
          tested (Ser214 for Ser195 in trypsin) and reported systems that were
          not there; benchmarks/active_site_validation.py has the numbers.
          Kept for tests of the geometry itself.

Residues the motifs do not mark (e.g. the Asp of a serine triad) are chosen
from all residues of that type, so a result is a PLAUSIBLE catalytic cluster,
not a proven active site. Only the four systems in database/catalytic_sites.json
are covered.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "database" / "catalytic_sites.json"

PROXIMITY_CUTOFF = 8.0     # Angstrom, residue centroid to centroid (legacy mode)
FUNCTIONAL_CUTOFF = 5.0    # Angstrom, between side-chain functional atoms (strict mode)

# ponytail: safety cap on combinations tried per rule, global lock style -
# bail out (treat as "not found") rather than hang on a pathological
# multimer with thousands of copies of one residue type.
MAX_COMBINATIONS = 200_000

_FUNCTIONAL_ATOMS = {
    "SER": {"OG"}, "CYS": {"SG"}, "HIS": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
}
_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


class _Residue:
    __slots__ = ("chain_id", "residue_number", "residue_name", "x", "y", "z", "functional")

    def __init__(self, chain_id, residue_number, residue_name, atoms):
        self.chain_id = chain_id
        self.residue_number = residue_number
        self.residue_name = residue_name
        n = len(atoms)
        self.x = sum(a.x for a in atoms) / n
        self.y = sum(a.y for a in atoms) / n
        self.z = sum(a.z for a in atoms) / n
        names = _FUNCTIONAL_ATOMS.get(residue_name, ())
        # fall back to the centroid when the side-chain atoms are not there
        self.functional = [(a.x, a.y, a.z) for a in atoms if a.atom_name in names] \
            or [(self.x, self.y, self.z)]

    def label(self) -> str:
        return f"{self.chain_id}:{self.residue_name}{self.residue_number}"

    def key(self) -> Tuple[str, int]:
        return (self.chain_id, self.residue_number)


def _group_residues(structure) -> List[_Residue]:
    modres = getattr(structure, "modified_residues", None) or {}
    groups: Dict[tuple, list] = {}
    names: Dict[tuple, str] = {}
    for atom in structure.atoms:
        key = (atom.chain_id, atom.residue_number)
        groups.setdefault(key, []).append(atom)
        # a modified residue (e.g. an oxidised catalytic Cys stored as CSD) counts as its parent
        names[key] = modres.get(key, atom.residue_name)
    return [
        _Residue(chain_id, number, names[(chain_id, number)], atoms)
        for (chain_id, number), atoms in groups.items()
    ]


def _chain_sequences(structure) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """[(one-letter sequence, [(residue_number, residue_name), ...])] per
    polymer chain, from ATOM records only (no waters or ligands)."""
    modres = getattr(structure, "modified_residues", None) or {}
    # modified residues are HETATM records, so they sit outside chain.atoms
    hetero: Dict[str, Dict[int, str]] = {}
    for lig in structure.ligands:
        parent = modres.get((lig.chain_id, lig.residue_number))
        if parent:
            hetero.setdefault(lig.chain_id, {})[lig.residue_number] = parent

    out = []
    for chain in structure.chains:
        residues: Dict[int, str] = {}
        for atom in chain.atoms:
            residues.setdefault(atom.residue_number,
                                modres.get((chain.chain_id, atom.residue_number), atom.residue_name))
        residues.update(hetero.get(chain.chain_id, {}))
        order = sorted(residues.items())
        seq = "".join(_ONE_LETTER.get(name, "X") for _, name in order)
        out.append((seq, order))
    return out


def _centroid_distance(a: _Residue, b: _Residue) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def _functional_distance(a: _Residue, b: _Residue) -> float:
    return min(math.dist(p, q) for p in a.functional for q in b.functional)


class SpatialCatalyticDetector:
    """Checks catalytic_sites.json rules against real 3D residue positions."""

    def __init__(self, cutoff: Optional[float] = None, strict: bool = True):
        with open(DATABASE, encoding="utf-8") as f:
            self.rules = json.load(f)
        self.strict = strict
        self.cutoff = cutoff if cutoff is not None else (
            FUNCTIONAL_CUTOFF if strict else PROXIMITY_CUTOFF)
        self._distance = _functional_distance if strict else _centroid_distance

    # ------------------------------------------------------------------
    def _motif_evidence(self, structure) -> Dict[str, Tuple[List[str], Set[Tuple[str, int]]]]:
        """rule name -> (names of the motifs found, the residues those motifs
        mark as catalytic), over every chain's sequence."""
        found: Dict[str, Tuple[List[str], Set[Tuple[str, int]]]] = {}
        chains = _chain_sequences(structure)
        chain_ids = [c.chain_id for c in structure.chains]
        for rule in self.rules:
            motifs_hit: List[str] = []
            sites: Set[Tuple[str, int]] = set()
            for chain_id, (seq, order) in zip(chain_ids, chains):
                for motif in rule.get("motifs", []):
                    for match in re.finditer(motif["regex"], seq):
                        if motif["name"] not in motifs_hit:
                            motifs_hit.append(motif["name"])
                        for site in motif.get("sites", []):
                            idx = match.start() + site["offset"]
                            if idx < len(order) and order[idx][1] == site["residue"]:
                                sites.add((chain_id, order[idx][0]))
            found[rule["name"]] = (motifs_hit, sites)
        return found

    def predict(self, structure) -> List[dict]:
        residues = _group_residues(structure)
        by_type: Dict[str, List[_Residue]] = {}
        for r in residues:
            by_type.setdefault(r.residue_name, []).append(r)
        evidence = self._motif_evidence(structure) if self.strict else {}

        hits = []
        for rule in self.rules:
            required = Counter(rule["residues"])
            candidates = {t: list(by_type.get(t, [])) for t in required}

            motifs_hit: List[str] = []
            if self.strict and "motifs" in rule:
                motifs_hit, sites = evidence[rule["name"]]
                if not motifs_hit:
                    continue            # no conserved motif, no system
                for t in required:      # use the residues the motif marks, where it marks any
                    marked = [r for r in candidates[t] if r.key() in sites]
                    if marked:
                        candidates[t] = marked

            if any(len(candidates[t]) < n for t, n in required.items()):
                continue

            site = self._find_clustered_instances(required, candidates)
            if site is not None:
                hit = {**rule, "residues_found": [r.label() for r in site]}
                if self.strict:
                    hit["motifs_found"] = motifs_hit
                hits.append(hit)

        return hits

    def _find_clustered_instances(self, required, candidates):
        """Try each distinct way to pick the required residue instances
        (distinct instances for a repeated type, e.g. two different ASPs)
        and return the tightest set in which every pair is within the cutoff."""
        per_type_choices = [
            list(combinations(candidates[residue_type], count))
            for residue_type, count in required.items()
        ]

        total = 1
        for choices in per_type_choices:
            total *= max(len(choices), 1)
        if total > MAX_COMBINATIONS:
            return None

        best, best_spread = None, None
        for combo in product(*per_type_choices):
            group = [r for part in combo for r in part]
            spread = self._spread(group)
            if spread <= self.cutoff and (best is None or spread < best_spread):
                best, best_spread = group, spread
        return best

    def _spread(self, group) -> float:
        """How loose the cluster is. Legacy: the longest pairwise distance.
        Strict: the longest edge of the minimum spanning tree, i.e. how far apart
        the residues must be reached to link them all together."""
        if len(group) < 2:
            return 0.0
        d = {(i, j): self._distance(group[i], group[j])
             for i, j in combinations(range(len(group)), 2)}
        if not self.strict:
            return max(d.values())
        in_tree, spread = {0}, 0.0
        while len(in_tree) < len(group):
            dist, nxt = min((d[(min(i, j), max(i, j))], j)
                            for i in in_tree for j in range(len(group)) if j not in in_tree)
            spread = max(spread, dist)
            in_tree.add(nxt)
        return spread
