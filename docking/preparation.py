"""
Receptor preparation for docking (Sprint 13 / Section 13 of prompt).

Wraps the existing structure parser and produces a docking-ready
receptor representation:

    Target PDB
        ↓
    Structure validation
        ↓
    Chain selection
        ↓
    Water/cofactor handling
        ↓
    Metal detection
        ↓
    Known ligand detection
        ↓
    Pocket detection
        ↓
    Candidate pocket ranking

The prepared receptor preserves provenance so reproducibility metadata
(Section 33) is recorded for downstream tools.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from structure.models import Atom, Ligand, ProteinStructure
from structure.parser import parse_pdb


# ----------------------------------------------------------------------
# Standard residue table
# ----------------------------------------------------------------------

_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}

_WATERS = {"HOH", "WAT", "DOD", "H2O"}

_METALS = {
    "ZN", "MG", "MN", "FE", "CU", "CA", "CO", "NI", "NA", "K",
}


@dataclass
class PreparedReceptor:
    pdb_path: str
    structure: ProteinStructure
    selected_chain: Optional[str]
    waters_removed: int
    cofactors_kept: List[str] = field(default_factory=list)
    metals_kept: List[str] = field(default_factory=list)
    known_ligand_resnames: List[str] = field(default_factory=list)
    receptor_atoms: List[Atom] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, **details: Any) -> None:
        self.provenance.append({
            "step": step,
            "time": datetime.now(timezone.utc).isoformat(),
            **details,
        })


class ReceptorPreparation:
    """Prepare a target PDB for downstream pocket/docking work."""

    def __init__(
        self,
        *,
        remove_waters: bool = True,
        keep_metals: bool = True,
        keep_known_ligands: bool = True,
        keep_cofactors: bool = True,
    ):
        self.remove_waters = remove_waters
        self.keep_metals = keep_metals
        self.keep_known_ligands = keep_known_ligands
        self.keep_cofactors = keep_cofactors

    def prepare(
        self,
        pdb_path: str,
        *,
        chain: Optional[str] = None,
    ) -> PreparedReceptor:
        path = Path(pdb_path)
        if not path.exists():
            raise FileNotFoundError(f"Receptor PDB not found: {path}")

        structure = parse_pdb(str(path))

        prep = PreparedReceptor(
            pdb_path=str(path),
            structure=structure,
            selected_chain=chain,
            waters_removed=0,
        )

        prep.record("receptor_loaded", n_atoms=len(structure.atoms),
                    n_chains=len(structure.chains))

        # Protein only (ATOM records). structure.atoms also holds every
        # HETATM — the bound crystal ligand, crystallization additives,
        # metals — and building the receptor from it left the bound
        # ligand sitting in its own pocket (docking it back is then
        # impossible) and duplicated metals, which are re-added below.
        protein_atoms = [a for c in structure.chains for a in c.atoms]
        chain_atoms = protein_atoms
        if chain is not None:
            chain_atoms = [a for a in protein_atoms if a.chain_id == chain]
            prep.record("chain_selected", chain=chain,
                        atoms_after=len(chain_atoms))

        # Waters
        if self.remove_waters:
            before = len(chain_atoms)
            chain_atoms = [a for a in chain_atoms
                           if a.residue_name not in _WATERS]
            prep.waters_removed = before - len(chain_atoms)

        # Partition ligands by category
        metals: List[Ligand] = []
        cofactors: List[Ligand] = []
        known_ligands: List[Ligand] = []
        for lig in structure.ligands:
            resname = lig.residue_name
            if chain is not None and lig.chain_id != chain:
                continue
            if resname in _WATERS:
                continue
            if resname in _METALS:
                metals.append(lig)
            elif resname in {"HEM", "NAD", "NADP", "FAD", "ATP", "ADP",
                             "GTP", "GDP", "SAM", "SAH", "PLP", "FAD",
                             "FMN", "UDP", "CDP", "GDP"}:
                cofactors.append(lig)
            else:
                known_ligands.append(lig)

        prep.metals_kept = [m.residue_name for m in metals] if self.keep_metals else []
        prep.cofactors_kept = [c.residue_name for c in cofactors] if self.keep_cofactors else []
        prep.known_ligand_resnames = [l.residue_name for l in known_ligands] \
            if self.keep_known_ligands else []

        # Receptor atoms = protein atoms + (metals/cofactors we keep)
        keep_resnames = set(prep.metals_kept) | set(prep.cofactors_kept)
        receptor = list(chain_atoms)
        for lig in metals + cofactors:
            if lig.residue_name in keep_resnames:
                receptor.extend(lig.atoms)
        prep.receptor_atoms = receptor

        prep.record("preparation_done",
                    receptor_atoms=len(receptor),
                    waters_removed=prep.waters_removed,
                    metals=len(prep.metals_kept),
                    cofactors=len(prep.cofactors_kept),
                    known_ligands=len(prep.known_ligand_resnames))

        return prep


# ----------------------------------------------------------------------
# AutoDock receptor atom typing
# ----------------------------------------------------------------------
#
# Standard amino acids are a fixed, known vocabulary (20 residues, a
# handful of atom names each) — unlike arbitrary ligand SMILES, this
# doesn't need to be computed, just looked up. AutoDock4 types that
# matter for H-bonding/aromaticity (the rest fall through to the plain
# element symbol):
#   A  = aromatic ring carbon (vs "C" aliphatic)
#   OA = H-bond-accepting oxygen
#   NA = H-bond-accepting nitrogen
#   SA = H-bond-accepting sulfur
#
# No receptor hydrogens are present (X-ray structures don't resolve
# them), so donor character can't be expressed either way — only
# acceptor typing is meaningful here.

# Ring carbons only; ring NITROGENS are typed via _ACCEPTOR_NITROGENS
# below (AutoDock has no separate "aromatic nitrogen" type).
_AROMATIC_RING_CARBONS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "CD2", "CE1"},
}

# Imidazole nitrogens: whichever tautomer, one accepts — treat both as
# acceptor-capable rather than guess a specific protonation state.
_ACCEPTOR_NITROGENS = {
    "HIS": {"ND1", "NE2"},
}

# Residues whose sulfur is commonly typed as acceptor-capable by
# AutoDock preparers (Cys thiol / Met thioether); everything else
# falls through to plain "S".
_ACCEPTOR_SULFUR_RESIDUES = {"CYS", "MET"}


def _autodock_atom_type(atom: Atom) -> str:
    element = atom.element.capitalize()

    if element == "C":
        ring = _AROMATIC_RING_CARBONS.get(atom.residue_name, ())
        return "A" if atom.atom_name in ring else "C"

    if element == "O":
        # Backbone carbonyl, hydroxyl, carboxylate — essentially every
        # oxygen in a standard amino acid is an H-bond acceptor.
        return "OA"

    if element == "N":
        acceptors = _ACCEPTOR_NITROGENS.get(atom.residue_name, ())
        return "NA" if atom.atom_name in acceptors else "N"

    if element == "S":
        return "SA" if atom.residue_name in _ACCEPTOR_SULFUR_RESIDUES else "S"

    # Metals and anything else: plain element symbol (already correctly
    # cased above), e.g. "Zn", "Mg", "Fe".
    return element


def write_receptor_pdbqt(atoms: List[Atom], out_path: str) -> str:
    """Write a minimal rigid-receptor PDBQT (plain ATOM records, no
    ROOT/TORSDOF — that wrapper is for flexible ligands, not receptors).

    Placeholder partial charges (real ones need a proper preparer like
    MGLTools/ADFRsuite), but atom TYPING is real standard-amino-acid
    chemistry (see _autodock_atom_type) — not just the raw element.
    """
    lines = []
    for a in atoms:
        atom_type = _autodock_atom_type(a)
        lines.append(
            f"ATOM  {a.serial:>5d} {a.atom_name:<4s}{a.residue_name:>4s} "
            f"{a.chain_id:1s}{a.residue_number:>4d}    "
            f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
            f"{a.occupancy:>6.2f}{a.b_factor:>6.2f}    "
            f"{0.0:>6.3f} {atom_type:<2s}"
        )
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
