"""
ProteinToolkit Ligand I/O

A self-contained ligand representation that does NOT depend on RDKit.
When RDKit is installed, SMILES parsing is delegated to it; otherwise we
fall back to a tiny validator-only parser that captures atoms/bonds as
flat records so the rest of the pipeline (validation, preparation,
compatibility, docking adapters) can still operate.

The output is a normalized Ligand object with provenance so every
downstream module can record exactly what transformations were applied.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_FORMATS = ("smi", "smiles", "sdf", "mol", "mol2", "pdb", "txt")


class LigandLoadError(ValueError):
    """Raised when a ligand cannot be read or parsed."""


# ----------------------------------------------------------------------
# Element / bond utilities (stdlib only)
# ----------------------------------------------------------------------

# Standard organic subset + common metals in ligands
_VALID_ELEMENTS = {
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "B", "Si", "Se", "As",
    "Li", "Na", "Mg", "Al", "K", "Ca", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Cd", "Sn", "Hg", "Au",
}


_AROMATIC_LOWER = {"c", "n", "o", "s"}


@dataclass
class LigandAtom:
    serial: int
    element: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    formal_charge: int = 0
    aromatic: bool = False
    in_ring: bool = False
    h_count: int = 0  # implicit hydrogens (filled in by preparation)


@dataclass
class LigandBond:
    a1: int          # serial of first atom
    a2: int          # serial of second atom
    order: float = 1.0   # 1, 1.5, 2, 3
    aromatic: bool = False
    in_ring: bool = False


@dataclass
class Ligand:
    """A normalized, self-describing ligand container.

    provenance records every transformation the ligand went through.
    It is the contract for reproducibility (Section 33).
    """
    name: str
    atoms: List[LigandAtom] = field(default_factory=list)
    bonds: List[LigandBond] = field(default_factory=list)
    source_format: str = ""
    source_path: Optional[str] = None
    smiles: Optional[str] = None
    # True when RDKit parsed and sanitized this structure — its own
    # valence/aromaticity model is correct where our flattened bond-order
    # sum (1.5 per aromatic bond, regardless of lone-pair donation) is
    # not; see LigandValidator's valence check.
    chemistry_verified: bool = False
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ---- provenance -----------------------------------------------------
    def record(self, step: str, **details: Any) -> None:
        """Append an immutable provenance record."""
        entry = {
            "step": step,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        entry.update(details)
        self.provenance.append(entry)

    # ---- structural geometry -------------------------------------------
    def center(self) -> Tuple[float, float, float]:
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        n = len(self.atoms)
        return (
            sum(a.x for a in self.atoms) / n,
            sum(a.y for a in self.atoms) / n,
            sum(a.z for a in self.atoms) / n,
        )

    def bounding_box(self) -> Tuple[float, float, float, float, float, float]:
        """Return (min_x, min_y, min_z, max_x, max_y, max_z)."""
        if not self.atoms:
            return (0, 0, 0, 0, 0, 0)
        xs = [a.x for a in self.atoms]
        ys = [a.y for a in self.atoms]
        zs = [a.z for a in self.atoms]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def atom_count(self) -> int:
        return len(self.atoms)

    def heavy_atom_count(self) -> int:
        return sum(1 for a in self.atoms if a.element != "H")

    def bond_count(self) -> int:
        return len(self.bonds)

    # ---- serialization --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_format": self.source_format,
            "source_path": self.source_path,
            "smiles": self.smiles,
            "atoms": [
                {
                    "serial": a.serial,
                    "element": a.element,
                    "x": a.x, "y": a.y, "z": a.z,
                    "formal_charge": a.formal_charge,
                    "aromatic": a.aromatic,
                    "in_ring": a.in_ring,
                    "h_count": a.h_count,
                }
                for a in self.atoms
            ],
            "bonds": [
                {
                    "a1": b.a1, "a2": b.a2,
                    "order": b.order,
                    "aromatic": b.aromatic,
                    "in_ring": b.in_ring,
                }
                for b in self.bonds
            ],
            "provenance": list(self.provenance),
            "created_at": self.created_at,
        }


# ----------------------------------------------------------------------
# Format detection
# ----------------------------------------------------------------------

def _detect_format(path: str | None, text: str | None) -> str:
    if path:
        ext = Path(path).suffix.lower().lstrip(".")
        if ext in {"smi", "smiles", "txt"}:
            return "smiles"
        if ext in {"sdf", "mol"}:
            return "sdf"
        if ext == "mol2":
            return "mol2"
        if ext == "pdb":
            return "pdb"
        raise LigandLoadError(f"Unsupported ligand file extension: .{ext}")
    # Heuristic on the textual payload
    if text is None:
        raise LigandLoadError("No ligand input supplied.")
    head = text.lstrip()[:200]
    if head.startswith("SMILES"):
        return "smiles"
    if head.startswith("MOL2") or "@<TRIPOS>MOLECULE" in head[:200].upper():
        return "mol2"
    if "HEADER" in head[:50] or "ATOM  " in head[:50] or "HETATM" in head[:50]:
        return "pdb"
    if "V2000" in head or "V3000" in head or "M  END" in head:
        return "sdf"
    # Default: treat as bare SMILES
    return "smiles"


# ----------------------------------------------------------------------
# SMILES parser (minimal, RDKit optional)
# ----------------------------------------------------------------------

# Atoms allowed in SMILES (uppercase = aliphatic, lowercase aromatic)
_ELEMENT_RE = re.compile(r"\[?(?P<elem>[A-Z][a-z]?)(?P<low>[a-z])?")


def _parse_smiles_fallback(smiles: str) -> Tuple[List[LigandAtom], List[LigandBond]]:
    """A minimal tokenizer-style SMILES parser.

    It supports:
        - organic subset (B, C, N, O, P, S, F, Cl, Br, I)
        - aromatic lowercase c, n, o, s
        - ring closures with single digits
        - bonds: - (single), = (double), # (triple), : (aromatic)
        - branches with ()
        - bracket atoms like [NH3+], [O-], [nH]

    It does NOT produce 3D coordinates. The preparator handles 3D.
    Stereochemistry (@, /, \\) is captured only as a flag for the
    validator; coordinates are not produced.
    """
    atoms: List[LigandAtom] = []
    bonds: List[LigandBond] = []
    stack: List[Tuple[int, Optional[int]]] = []  # (parent, last_main)
    ring_open: Dict[int, int] = {}                # ring bond -> serial
    pending_bond: Optional[Tuple[float, bool]] = None
    main_last: Optional[int] = None              # last atom on main chain

    i = 0
    serial = 0

    def add_atom(elem: str, aromatic: bool = False,
                 charge: int = 0, h: int = 0) -> int:
        nonlocal serial
        serial += 1
        atoms.append(LigandAtom(
            serial=serial,
            element=elem,
            formal_charge=charge,
            aromatic=aromatic,
            h_count=h,
        ))
        return serial

    def add_bond(a: int, b: int, order: float, aromatic: bool) -> None:
        if a == b:
            return
        for ex in bonds:
            if {ex.a1, ex.a2} == {a, b}:
                return
        bonds.append(LigandBond(a, b, order, aromatic))

    organic = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

    while i < len(smiles):
        c = smiles[i]

        if c.isspace() or c == "\t":
            i += 1
            continue

        if c == ".":
            # Disconnector: separate fragments, no bond between them.
            pending_bond = None
            main_last = None
            i += 1
            continue

        if c == "(":
            if not atoms:
                raise LigandLoadError("Branch '(' before any atom.")
            # Remember parent + the main-chain last so we can restore
            # the main chain when the branch closes.
            stack.append((atoms[-1].serial, main_last))
            i += 1
            continue

        if c == ")":
            if not stack:
                raise LigandLoadError("Unmatched ')' in SMILES.")
            _, prev_main = stack.pop()
            main_last = prev_main
            i += 1
            continue

        if c in "-=#":
            order = {"-": 1.0, "=": 2.0, "#": 3.0}[c]
            pending_bond = (order, False)
            i += 1
            continue

        if c == ":":
            pending_bond = (1.5, True)
            i += 1
            continue

        if c == "%":
            # two-digit ring closure
            if i + 2 >= len(smiles) or not (smiles[i+1].isdigit() and smiles[i+2].isdigit()):
                raise LigandLoadError("Invalid ring-closure %nn.")
            ring_id = int(smiles[i+1:i+3])
            i += 3
        elif c.isdigit():
            ring_id = int(c)
            i += 1
        else:
            ring_id = None

        # ring closure handling
        if ring_id is not None:
            order, aromatic_flag = pending_bond if pending_bond else (1.0, False)
            pending_bond = None
            if not atoms:
                raise LigandLoadError("Ring closure before any atom.")
            last = atoms[-1].serial
            if ring_id in ring_open:
                other = ring_open.pop(ring_id)
                add_bond(other, last, order, aromatic_flag)
            else:
                ring_open[ring_id] = last
            continue

        # bracketed atom [ ... ]
        if c == "[":
            end = smiles.find("]", i)
            if end == -1:
                raise LigandLoadError("Unclosed '[' in SMILES.")
            body = smiles[i+1:end]
            i = end + 1

            m = re.match(
                r"(?P<elem>[A-Z][a-z]?)?(?P<iso>[0-9]*)?"
                r"(?P<his>H(?P<hn>\d*))?"
                r"(?P<chg>[+-]\d*)?",
                body,
            )
            if not m:
                raise LigandLoadError(f"Unparseable bracket atom: [{body}]")

            elem = m.group("elem") or "C"
            hn = m.group("hn")
            chg = m.group("chg")

            if elem not in _VALID_ELEMENTS:
                raise LigandLoadError(f"Unsupported element: {elem}")

            aromatic = elem in _AROMATIC_LOWER
            charge = 0
            if chg:
                if chg in {"+", "-"}:
                    charge = 1 if chg == "+" else -1
                else:
                    charge = int(chg[0] + "1" if len(chg) == 1 else chg)

            h_count = int(hn) if hn else 0

            new_serial = add_atom(elem, aromatic=aromatic,
                                  charge=charge, h=h_count)

        else:
            # simple atom
            two = c.isupper() and i + 1 < len(smiles) and smiles[i+1] in "lrg"
            if two and (c + smiles[i+1]) in {"Cl", "Br", "Si"}:
                elem = c + smiles[i+1]
                aromatic = False
                i += 2
            else:
                if c.isupper():
                    elem = c
                    aromatic = False
                elif c in _AROMATIC_LOWER:
                    elem = c.upper()
                    aromatic = True
                else:
                    raise LigandLoadError(f"Invalid atom symbol: '{c}'")
                i += 1

            if elem not in _VALID_ELEMENTS:
                raise LigandLoadError(f"Unsupported element: {elem}")

            new_serial = add_atom(elem, aromatic=aromatic)

        # Connect to the right parent. After a branch close, the parent
        # is the last atom on the *main* chain (main_last), not atoms[-2].
        if stack:
            parent = stack[-1][0]
        elif main_last is not None:
            parent = main_last
        elif len(atoms) >= 2:
            parent = atoms[-2].serial
        else:
            parent = None

        if parent is not None and new_serial != parent:
            order, aromatic_flag = pending_bond if pending_bond else (1.0, False)
            pending_bond = None
            add_bond(parent, new_serial, order, aromatic_flag)

        # The new atom becomes the active main-chain atom unless we
        # are inside a branch (where main_last is preserved).
        if not stack:
            main_last = new_serial

    if stack:
        raise LigandLoadError("Unclosed '(' in SMILES.")
    if ring_open:
        raise LigandLoadError("Unclosed ring closure in SMILES.")

    if not atoms:
        raise LigandLoadError("SMILES contained no atoms.")

    return atoms, bonds


# ----------------------------------------------------------------------
# SDF / MOL parser (V2000)
# ----------------------------------------------------------------------

def _parse_sdf(text: str) -> Tuple[List[LigandAtom], List[LigandBond], str]:
    lines = text.splitlines()
    if len(lines) < 4:
        raise LigandLoadError("SDF/MOL too short.")

    # Counts line is the 4th line (index 3) — V2000 only
    try:
        counts = lines[3]
        n_atoms = int(counts[0:3])
        n_bonds = int(counts[3:6])
    except (ValueError, IndexError) as exc:
        raise LigandLoadError("Malformed SDF counts line.") from exc

    atoms: List[LigandAtom] = []
    bonds: List[LigandBond] = []

    offset = 4
    for i in range(n_atoms):
        line = lines[offset + i]
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
            elem = line[31:34].strip()
            charge = 0
            chg_field = line[36:39].strip()
            if chg_field:
                # SDF charges: 1=+3, 2=+2, 3=+1, 4=doublet, 5=-1, 6=-2, 7=-3
                chg_map = {"1": 3, "2": 2, "3": 1, "5": -1, "6": -2, "7": -3}
                charge = chg_map.get(chg_field, 0)
        except (ValueError, IndexError) as exc:
            raise LigandLoadError(
                f"Malformed atom line {i+1} in SDF."
            ) from exc

        if elem not in _VALID_ELEMENTS:
            raise LigandLoadError(f"Unsupported element in SDF: {elem}")

        atoms.append(LigandAtom(
            serial=i + 1, element=elem,
            x=x, y=y, z=z, formal_charge=charge,
        ))

    offset += n_atoms
    for j in range(n_bonds):
        line = lines[offset + j]
        try:
            a1 = int(line[0:3])
            a2 = int(line[3:6])
            order = int(line[6:9])
        except (ValueError, IndexError) as exc:
            raise LigandLoadError(
                f"Malformed bond line {j+1} in SDF."
            ) from exc

        order_val = {1: 1.0, 2: 2.0, 3: 3.0}.get(order, 1.0)
        if order == 4:        # aromatic in V2000
            order_val = 1.5
        bonds.append(LigandBond(a1, a2, order_val, order == 4))

    # Title is line 1 (index 0), trimmed
    name = lines[0].strip() or "ligand"
    return atoms, bonds, name


# ----------------------------------------------------------------------
# PDB ligand parser — reuses our structure parser for HETATM rows
# ----------------------------------------------------------------------

def _parse_ligand_pdb(text: str, path: Optional[str]) -> Tuple[List[LigandAtom], List[LigandBond], str, List[str]]:
    """Read HETATM rows; very small ligands usually lack CONECT records,
    so bonds may be empty. The user can provide connectivity separately
    or via SMILES.

    Returns (atoms, bonds, name, warnings). Malformed rows are
    reported as warnings rather than silently dropped.
    """
    atoms: List[LigandAtom] = []
    name = "ligand"
    serial = 0
    warnings: List[str] = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        if len(line) < 6:
            continue
        record = line[:6].strip()
        if record != "HETATM":
            continue
        serial += 1
        try:
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resnum = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() or atom_name[:1]
        except (ValueError, IndexError) as exc:
            warnings.append(
                f"line {lineno}: malformed HETATM row ({exc}); skipped."
            )
            continue
        if element not in _VALID_ELEMENTS:
            element = atom_name[:1].upper()
        if element not in _VALID_ELEMENTS:
            warnings.append(
                f"line {lineno}: unsupported element '{element}'; skipped."
            )
            continue
        name = resname or name
        atoms.append(LigandAtom(
            serial=serial, element=element,
            x=x, y=y, z=z,
        ))

    if not atoms:
        raise LigandLoadError(
            "No HETATM records found in ligand PDB. "
            + (" ".join(warnings) if warnings else "")
        )

    return atoms, [], name, warnings


# ----------------------------------------------------------------------
# MOL2 parser (Tripos) — atoms + bonds only, sufficient for geometry
# ----------------------------------------------------------------------

def _parse_mol2(text: str) -> Tuple[List[LigandAtom], List[LigandBond], str]:
    atoms: List[LigandAtom] = []
    bonds: List[LigandBond] = []
    name = "ligand"
    section = None
    serial = 0
    atom_serial_map: Dict[int, int] = {}

    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("@<TRIPOS>MOLECULE"):
            section = "molecule"
            continue
        if up.startswith("@<TRIPOS>ATOM"):
            section = "atom"
            continue
        if up.startswith("@<TRIPOS>BOND"):
            section = "bond"
            continue
        if up.startswith("@<TRIPOS>"):
            section = None
            continue
        if section == "molecule" and name == "ligand" and s:
            name = s.split()[0]
        elif section == "atom":
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                orig_serial = int(parts[0])
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                elem = parts[1].split(".")[0]
            except ValueError:
                continue
            if elem not in _VALID_ELEMENTS:
                continue
            serial += 1
            atom_serial_map[orig_serial] = serial
            atoms.append(LigandAtom(
                serial=serial, element=elem, x=x, y=y, z=z,
            ))
        elif section == "bond":
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                a1 = atom_serial_map[int(parts[1])]
                a2 = atom_serial_map[int(parts[2])]
                order_str = parts[3]
            except (KeyError, ValueError):
                continue
            order_map = {"1": 1.0, "2": 2.0, "3": 3.0, "ar": 1.5, "am": 1.5}
            order = order_map.get(order_str.lower(), 1.0)
            bonds.append(LigandBond(a1, a2, order, order_str.lower() in {"ar", "am"}))

    if not atoms:
        raise LigandLoadError("No atoms found in MOL2.")
    return atoms, bonds, name


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load_ligand(
    source: str | os.PathLike | None = None,
    *,
    text: Optional[str] = None,
    name: Optional[str] = None,
    smiles: Optional[str] = None,
) -> Ligand:
    """Load a ligand from a path, an in-memory text, or a SMILES string.

    Resolution priority:
        1) `smiles` argument (canonical string chemistry input)
        2) `text` argument (raw file content)
        3) `source` (file path)
    """
    if smiles is not None:
        fmt = "smiles"
        body = smiles
        path_str: Optional[str] = None
    elif text is not None:
        fmt = _detect_format(None, text)
        body = text
        path_str = None
    elif source is not None:
        path = Path(source)
        if not path.exists():
            raise LigandLoadError(f"Ligand file not found: {path}")
        body = path.read_text(encoding="utf-8", errors="replace")
        path_str = str(path)
        fmt = _detect_format(str(path), body)
    else:
        raise LigandLoadError("No ligand input provided.")

    if fmt == "smiles":
        # Prefer RDKit if installed
        used_rdkit = False
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
            mol = Chem.MolFromSmiles(body)
            if mol is None or mol.GetNumAtoms() == 0:
                raise LigandLoadError(f"RDKit could not parse SMILES: {body!r}")
            atoms, bonds = _rdkit_to_records(mol)
            used_rdkit = True
        except ImportError:
            atoms, bonds = _parse_smiles_fallback(body)
        lig_name = name or "ligand"
        lig = Ligand(
            name=lig_name,
            atoms=atoms, bonds=bonds,
            source_format="smiles",
            source_path=path_str,
            smiles=body,
            chemistry_verified=used_rdkit,
        )
        lig.record("loaded", format="smiles", length=len(body))
        return lig

    if fmt == "sdf":
        atoms, bonds, lig_name = _parse_sdf(body)
        lig_name = name or lig_name
        lig = Ligand(
            name=lig_name, atoms=atoms, bonds=bonds,
            source_format="sdf", source_path=path_str,
        )
        lig.record("loaded", format="sdf")
        return lig

    if fmt == "mol2":
        atoms, bonds, lig_name = _parse_mol2(body)
        lig_name = name or lig_name
        lig = Ligand(
            name=lig_name, atoms=atoms, bonds=bonds,
            source_format="mol2", source_path=path_str,
        )
        lig.record("loaded", format="mol2")
        return lig

    if fmt == "pdb":
        atoms, bonds, lig_name, warnings = _parse_ligand_pdb(body, path_str)
        lig_name = name or lig_name
        lig = Ligand(
            name=lig_name, atoms=atoms, bonds=bonds,
            source_format="pdb", source_path=path_str,
        )
        lig.record("loaded", format="pdb", warnings=warnings)
        return lig

    raise LigandLoadError(f"Unsupported ligand format: {fmt}")


def _rdkit_to_records(mol) -> Tuple[List[LigandAtom], List[LigandBond]]:
    from rdkit import Chem  # type: ignore
    atoms: List[LigandAtom] = []
    bonds: List[LigandBond] = []
    conf = mol.GetConformer() if mol.GetNumConformers() else None

    for atom in mol.GetAtoms():
        idx = atom.GetIdx() + 1
        x = y = z = 0.0
        if conf is not None:
            pos = conf.GetAtomPosition(atom.GetIdx())
            x, y, z = pos.x, pos.y, pos.z
        atoms.append(LigandAtom(
            serial=idx,
            element=atom.GetSymbol(),
            x=x, y=y, z=z,
            formal_charge=atom.GetFormalCharge(),
            aromatic=atom.GetIsAromatic(),
            in_ring=atom.IsInRing(),
            h_count=atom.GetTotalNumHs(),
        ))

    for bond in mol.GetBonds():
        order = bond.GetBondType()
        order_val = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5,
        }.get(order, 1.0)
        bonds.append(LigandBond(
            a1=bond.GetBeginAtomIdx() + 1,
            a2=bond.GetEndAtomIdx() + 1,
            order=order_val,
            aromatic=order == Chem.BondType.AROMATIC,
            in_ring=bond.IsInRing(),
        ))
    return atoms, bonds
