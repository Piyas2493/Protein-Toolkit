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

import contextlib
import io
import math
import os
import tempfile
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
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
    protein_atoms: List[Atom] = field(default_factory=list)   # ATOM records
    hetero_atoms: List[Atom] = field(default_factory=list)    # kept metals/cofactors
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    pdbqt_path: Optional[str] = None
    pdbqt_info: Dict[str, Any] = field(default_factory=dict)

    def pdbqt(self, ph: Optional[float] = None) -> str:
        """Docking-ready receptor PDBQT, built once and cached. Receptor
        protonation/charging is ligand-independent and costs seconds, so
        screening loops must not redo it per ligand."""
        if self.pdbqt_path is None:
            fd, path = tempfile.mkstemp(suffix="_receptor.pdbqt")
            os.close(fd)
            self.pdbqt_info = write_prepared_receptor_pdbqt(
                self, path, ph=DEFAULT_PH if ph is None else ph)
            self.pdbqt_path = path
            self.record("receptor_pdbqt", **self.pdbqt_info)
        return self.pdbqt_path

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
        hetero = [a for lig in metals + cofactors
                  if lig.residue_name in keep_resnames for a in lig.atoms]
        prep.protein_atoms = list(chain_atoms)
        prep.hetero_atoms = hetero
        receptor = prep.protein_atoms + hetero
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
# This table is the FALLBACK typing, used by write_receptor_pdbqt when the
# protonating preparer below is unavailable. X-ray structures have no
# hydrogens, so without them donor character can't be expressed — only
# acceptor typing is meaningful here. (Vina marks a polar N/O as an
# H-bond donor only if an HD hydrogen is bonded to it.)

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


def _pdbqt_line(serial: int, a: Atom, atom_type: str, charge: float) -> str:
    return (
        f"ATOM  {serial:>5d} {a.atom_name:<4s}{a.residue_name:>4s} "
        f"{a.chain_id:1s}{a.residue_number:>4d}    "
        f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
        f"{a.occupancy:>6.2f}{a.b_factor:>6.2f}    "
        f"{charge:>6.3f} {atom_type:<2s}"
    )


def write_receptor_pdbqt(atoms: List[Atom], out_path: str) -> str:
    """FALLBACK rigid-receptor PDBQT: no hydrogens (so no HD donor atoms),
    all partial charges 0.0, but real standard-amino-acid acceptor/aromatic
    typing (see _autodock_atom_type). Plain ATOM records, no ROOT/TORSDOF —
    that wrapper is for flexible ligands, not receptors. Prefer
    write_prepared_receptor_pdbqt, which protonates and charges.
    """
    lines = [_pdbqt_line(a.serial, a, _autodock_atom_type(a), 0.0)
             for a in atoms]
    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------
# Protonated, charged receptor: PDB2PQR/PROPKA -> meeko
# ----------------------------------------------------------------------
#
#   protein atoms -> pdb2pqr (rebuilds missing heavy atoms, PROPKA pKa at
#   the requested pH, adds hydrogens, optimizes the H-bond network) -> PQR
#   -> meeko (matches each residue to a chemical template, assigns
#   AutoDock4 types incl. HD donors, Gasteiger-based template charges).
#
# Metals/cofactors are appended afterwards with the table typing above:
# pdb2pqr has no parameters for them, and meeko has no templates for
# arbitrary cofactors. Cofactors therefore get no hydrogens and zero
# charge; metals get their formal charge.
#
# Vina's scoring function ignores partial charges; what the preparation
# changes for Vina is atom typing (HD donors, aromatic carbons), and the
# charges matter only for AutoDock4-style scoring.

DEFAULT_PH = 7.4

_METAL_CHARGES = {"ZN": 2.0, "MG": 2.0, "CA": 2.0, "MN": 2.0, "FE": 2.0,
                  "CU": 2.0, "CO": 2.0, "NI": 2.0, "NA": 1.0, "K": 1.0}

# A polar H this close to a retained metal means pdb2pqr protonated an
# atom that is actually coordinating it (it can't see the metal).
_H_METAL_CLASH = 2.5


def _pdb2pqr_text(atoms: List[Atom]) -> Tuple[str, int, List[str]]:
    """Protein atoms as PDB text for pdb2pqr, the residue count, and the
    residues left out. pdb2pqr rebuilds missing side-chain atoms but not a
    missing backbone, so residues lacking N/CA/C (e.g. a C-terminal residue
    resolved only to N) are left out and reported rather than crashing the
    whole preparation. Residues are renumbered 1..N in file order: the
    parser drops insertion codes, so (chain, number) is not a unique key."""
    lines, left_out, n_res, serial = [], [], 0, 0
    residues = groupby(atoms, key=lambda a: (a.chain_id, a.residue_number,
                                             a.residue_name))
    for (chain, number, name), group in residues:
        group = list(group)
        if not {"N", "CA", "C"} <= {a.atom_name for a in group}:
            left_out.append(f"{chain}:{number} {name}")
            continue
        n_res += 1
        for a in group:
            serial += 1
            element = (a.element or a.atom_name.lstrip("0123456789")[:1]).upper()
            lines.append(
                f"ATOM  {serial % 100000:>5d} {a.atom_name:<4s} "
                f"{a.residue_name:>3s} {chain or 'A':1s}{n_res % 10000:>4d}    "
                f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}"
                f"          {element:>2s}")
    return "\n".join(lines) + "\nEND\n", n_res, left_out


def _protonate(pdb_text: str, ph: float) -> str:
    """PQR text (hydrogens added at `ph`, PROPKA titration states)."""
    import propka.parameters as propka_parameters
    from pdb2pqr.main import VERSION, run_pdb2pqr

    # propka 3.5 reads `self.__annotations__` on an instance, which Python
    # 3.14 (PEP 649) no longer provides; expose the class's instead.
    try:
        propka_parameters.Parameters().__annotations__
    except AttributeError:
        def _annotations(self, name):
            if name == "__annotations__":
                return type(self).__annotations__
            raise AttributeError(name)
        propka_parameters.Parameters.__getattr__ = _annotations

    loggers = [logging.getLogger(n) for n in ("pdb2pqr", f"PDB2PQR{VERSION}", "propka")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:         # per-residue "Missing atom" chatter
        lg.setLevel(logging.ERROR)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdb_in, pqr_out = Path(tmp, "in.pdb"), Path(tmp, "out.pqr")
            pdb_in.write_text(pdb_text, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):  # PROPKA's tables
                run_pdb2pqr([pdb_in, pqr_out, "--ff", "AMBER", "--keep-chain",
                             "--whitespace", "--titration-state-method",
                             "propka", "--with-ph", ph])
            return pqr_out.read_text(encoding="utf-8")
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def _pqr_to_pdbqt(pqr_text: str, n_residues: int) -> str:
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer

    polymer = Polymer.from_pqr_string(pqr_text, mk_prep=MoleculePreparation())
    ignored = polymer.get_ignored_monomers()
    n_valid = len(polymer.get_valid_monomers())
    # A missing residue silently reshapes the pocket; refuse rather than dock.
    if ignored or n_valid != n_residues:
        raise ValueError(f"{n_residues} residues in, {n_valid} templated by "
                         f"meeko, ignored: {sorted(ignored)}")
    return PDBQTWriterLegacy.write_from_polymer(polymer)[0]


def write_prepared_receptor_pdbqt(
    prep: PreparedReceptor, out_path: str, *, ph: float = DEFAULT_PH,
) -> Dict[str, Any]:
    """Write `prep`'s receptor as a protonated, charged PDBQT and return a
    provenance dict ("route" is "pdb2pqr+meeko" or "legacy").

    Falls back to write_receptor_pdbqt — no hydrogens, zero charges — with a
    warning if pdb2pqr/propka/meeko are missing or any residue can't be
    prepared (so a pocket residue is never silently dropped).
    """
    try:
        pdb_text, n_res, left_out = _pdb2pqr_text(prep.protein_atoms)
        body = _pqr_to_pdbqt(_protonate(pdb_text, ph), n_res).rstrip("\n")
    except Exception as exc:       # ImportError, pdb2pqr/meeko failure, ...
        cause = exc.__cause__ or exc      # pdb2pqr re-raises with no message
        reason = f"{type(cause).__name__}: {cause}"
        warnings.warn(
            f"Receptor protonation/charging unavailable ({reason}); using "
            "the unprotonated zero-charge receptor. Vina will see no "
            "receptor H-bond donors, so scores are less reliable.",
            RuntimeWarning, stacklevel=2)
        write_receptor_pdbqt(prep.receptor_atoms, out_path)
        return {"route": "legacy", "ph": None, "reason": reason}

    lines = body.splitlines()
    n_hd = sum(1 for ln in lines if ln[77:79].strip() == "HD")
    charge = sum(float(ln[70:76]) for ln in lines)
    metals = [(a.x, a.y, a.z) for a in prep.hetero_atoms
              if a.residue_name in _METALS]
    hd_near_metal = sum(
        1 for ln in lines if ln[77:79].strip() == "HD" and any(
            math.dist((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])), m)
            < _H_METAL_CLASH for m in metals))
    if hd_near_metal:
        warnings.warn(f"{hd_near_metal} polar hydrogen(s) within "
                      f"{_H_METAL_CLASH} A of a metal: a coordinating atom "
                      "was probably protonated.", RuntimeWarning, stacklevel=2)
    for serial, a in enumerate(prep.hetero_atoms, start=len(lines) + 1):
        lines.append(_pdbqt_line(serial, a, _autodock_atom_type(a),
                                 _METAL_CHARGES.get(a.residue_name, 0.0)))
    Path(out_path).write_text("\n".join(lines + ["END"]) + "\n",
                              encoding="utf-8")
    return {"route": "pdb2pqr+meeko", "ph": ph, "n_atoms": len(lines),
            "n_hd": n_hd, "net_charge": round(charge, 2),
            "hd_near_metal": hd_near_metal, "residues_left_out": left_out}
