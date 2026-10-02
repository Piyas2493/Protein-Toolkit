"""
ProteinToolkit Machine Learning Feature Extraction Engine

Two feature tiers:
    - BASE features (32 columns): amino-acid percent, length, MW,
      pI, aromaticity, instability, gravy, helix/turn/sheet.
    - EXTENDED features (32 + 400 = 432 columns): adds dipeptide
      composition (20 x 20 AA pairs).

`include_dipeptides=False` (default) preserves the original schema
so models trained on the BASE feature set keep working.
`include_dipeptides=True` enables the wider schema for the next-gen
classifier.
"""
from __future__ import annotations

from typing import List

from Bio.SeqUtils.ProtParam import ProteinAnalysis


_BASE_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class FeatureExtractor:

    BASE_FEATURE_NAMES: List[str] = (
        [f"AA_{a}" for a in _BASE_AMINO_ACIDS]
        + [
            "AA_U", "Length", "MolecularWeight", "IsoelectricPoint",
            "Aromaticity", "InstabilityIndex", "Gravy",
            "Helix", "Turn", "Sheet",
        ]
    )

    DIPEPTIDE_FEATURE_NAMES: List[str] = [
        f"DIPEP_{a}{b}" for a in _BASE_AMINO_ACIDS for b in _BASE_AMINO_ACIDS
    ]

    @property
    def extended_feature_names(self) -> List[str]:
        return self.BASE_FEATURE_NAMES + self.DIPEPTIDE_FEATURE_NAMES

    def extract(self, sequence: str, *, include_dipeptides: bool = False) -> dict:
        sequence = sequence.upper().replace("\n", "")

        analysis_sequence = sequence.replace("U", "C")
        analysis = ProteinAnalysis(analysis_sequence)

        amino_acids = analysis.amino_acids_percent
        features: dict = {}

        for aa in _BASE_AMINO_ACIDS:
            features[f"AA_{aa}"] = amino_acids.get(aa, 0.0)

        features["AA_U"] = (
            sequence.count("U") / len(sequence) if sequence else 0.0
        )

        features["Length"] = len(sequence)
        features["MolecularWeight"] = analysis.molecular_weight()
        features["IsoelectricPoint"] = analysis.isoelectric_point()
        features["Aromaticity"] = analysis.aromaticity()
        features["InstabilityIndex"] = analysis.instability_index()
        features["Gravy"] = analysis.gravy()

        helix, turn, sheet = analysis.secondary_structure_fraction()
        features["Helix"] = helix
        features["Turn"] = turn
        features["Sheet"] = sheet

        if include_dipeptides:
            features.update(self._dipeptide_freqs(sequence))

        return features

    @staticmethod
    def _dipeptide_freqs(sequence: str) -> dict:
        """Normalized dipeptide frequencies over the 20 standard AAs.

        Sequences with non-standard residues are kept in the dipeptide
        pair only when BOTH residues are standard; non-standard residues
        are skipped to avoid polluting the 20x20 distribution.
        """
        standard = set(_BASE_AMINO_ACIDS)
        counts = {(a, b): 0 for a in _BASE_AMINO_ACIDS
                  for b in _BASE_AMINO_ACIDS}
        total = 0
        for i in range(len(sequence) - 1):
            a, b = sequence[i], sequence[i + 1]
            if a in standard and b in standard:
                counts[(a, b)] += 1
                total += 1
        out = {}
        for (a, b), c in counts.items():
            out[f"DIPEP_{a}{b}"] = (c / total) if total else 0.0
        return out
