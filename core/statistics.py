"""
Sequence statistics module.

Uses BioPython's ProteinAnalysis to calculate
physicochemical properties.
"""

from dataclasses import dataclass
from Bio.SeqUtils.ProtParam import ProteinAnalysis

_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class SequenceStatistics:

    sequence: str

    def __post_init__(self):
        # ProtParam's per-residue scale tables (GRAVY, instability,
        # secondary-structure fraction, ...) only cover the 20 standard
        # amino acids and KeyError on ambiguous codes (X/B/Z/J/U/O) —
        # which SequenceValidator treats as valid. Exclude them here
        # rather than crash or guess at what they might be.
        clean = "".join(c for c in self.sequence if c in _STANDARD_AA)
        self.analysis = ProteinAnalysis(clean)

    def molecular_weight(self):
        return round(
            self.analysis.molecular_weight(),
            2
        )

    def aromaticity(self):
        return round(
            self.analysis.aromaticity(),
            3
        )

    def instability_index(self):
        return round(
            self.analysis.instability_index(),
            2
        )

    def gravy(self):
        return round(
            self.analysis.gravy(),
            3
        )

    def isoelectric_point(self):
        return round(
            self.analysis.isoelectric_point(),
            2
        )

    def amino_acid_percent(self):

        aa = self.analysis.amino_acids_percent

        return {
            k: round(v, 2)
            for k, v in aa.items()
        }

    def extinction_coefficients(self):

        reduced, oxidized = \
            self.analysis.molar_extinction_coefficient()

        return {
            "Reduced": reduced,
            "Oxidized": oxidized
        }

    def secondary_structure(self):

        helix, turn, sheet = \
            self.analysis.secondary_structure_fraction()

        return {
            "Alpha Helix": round(
                helix * 100,
                2
            ),
            "Turn": round(
                turn * 100,
                2
            ),
            "Beta Sheet": round(
                sheet * 100,
                2
            )
        }