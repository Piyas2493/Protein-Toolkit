"""
ProteinToolkit Structure Models
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Atom:
    serial: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str


@dataclass
class Ligand:
    residue_name: str
    chain_id: str
    residue_number: int
    atoms: List[Atom] = field(default_factory=list)


@dataclass
class Chain:
    chain_id: str
    atoms: List[Atom] = field(default_factory=list)


@dataclass
class ProteinStructure:
    atoms: List[Atom] = field(default_factory=list)
    ligands: List[Ligand] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    # Residues the file records as chemically modified (MODRES), e.g. an
    # oxidised catalytic cysteine stored as HETATM CSD: (chain, number) -> the
    # standard parent residue name, here "CYS". Needed to read such a chain.
    modified_residues: Dict[Tuple[str, int], str] = field(default_factory=dict)
    