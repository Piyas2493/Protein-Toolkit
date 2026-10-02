"""
ProteinToolkit Structure Package
"""

from .structural_active_site import StructuralActiveSitePredictor

from .models import (
    Atom,
    Ligand,
    Chain,
    ProteinStructure,
)

from .contacts import ContactAnalyzer
from .atom_classifier import AtomInteractionClassifier
from .pocket_scoring import PocketScorer
from .catalytic_geometry import SpatialCatalyticDetector

__all__ = [
    "Atom",
    "Ligand",
    "Chain",
    "ProteinStructure",
    "StructuralActiveSitePredictor",
    "ContactAnalyzer",
    "AtomInteractionClassifier",
    "PocketScorer",
    "SpatialCatalyticDetector",
]
