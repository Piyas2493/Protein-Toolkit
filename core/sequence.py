from collections import Counter
from dataclasses import dataclass


@dataclass
class ProteinSequence:

    name: str

    sequence: str

    def length(self):

        return len(self.sequence)

    def amino_acids(self):

        return sorted(
            set(self.sequence)
        )

    def residue_count(self):

        return dict(Counter(self.sequence))
