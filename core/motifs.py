"""
Motif Detection Engine

Loads motifs from JSON and searches
protein sequences using regex.
"""

import json
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "database" / "motifs.json"


class MotifDatabase:

    def __init__(self):

        with open(DATABASE,
                  encoding="utf-8") as f:

            self.motifs = json.load(f)

    def search(self,
               sequence):

        sequence = sequence.upper()

        hits = []

        for motif in self.motifs:

            pattern = re.compile(
                motif["regex"]
            )

            for match in pattern.finditer(sequence):

                hits.append({

                    "name":
                        motif["name"],

                    "family":
                        motif["family"],

                    "ligand":
                        motif["ligand"],

                    "confidence":
                        # Database stores 0-1; rescale to the 0-100 scale
                        # every other predictor module uses.
                        round(motif["confidence"] * 100),

                    "description":
                        motif["description"],

                    "sequence":
                        match.group(),

                    "start":
                        match.start()+1,

                    "end":
                        match.end()

                })

        return hits