"""
ProteinToolkit Ligand Engine

Handles ligand input, validation, preparation, and provenance.
Sprint 4 / Sprint 5.

Public entry points:
    load_ligand(...)        - read SMILES / SDF / MOL / MOL2 / PDB
    Ligand(...)             - in-memory ligand object
    LigandValidator(...)    - chemical validity + descriptors
    LigandPreparator(...)   - standardization + protonation + 3D conformer
"""
from .ligand import (
    Ligand,
    LigandAtom,
    LigandBond,
    load_ligand,
    SUPPORTED_FORMATS,
    LigandLoadError,
)
from .validation import (
    LigandValidator,
    LigandValidationError,
    ValidationReport,
)
from .preparation import (
    LigandPreparator,
    PreparationReport,
    PreparationStep,
)

__all__ = [
    "Ligand",
    "LigandAtom",
    "LigandBond",
    "load_ligand",
    "SUPPORTED_FORMATS",
    "LigandLoadError",
    "LigandValidator",
    "LigandValidationError",
    "ValidationReport",
    "LigandPreparator",
    "PreparationReport",
    "PreparationStep",
]
