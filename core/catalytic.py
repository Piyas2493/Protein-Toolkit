"""
Catalytic residue prediction engine
"""

import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE = BASE_DIR / "database" / "catalytic_sites.json"


AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL"
}


class CatalyticPredictor:

    def __init__(self):

        with open(DATABASE, encoding="utf-8") as f:

            self.rules = json.load(f)

    def predict(self, sequence):

        sequence = sequence.upper()

        residues = [AA3.get(x) for x in sequence if x in AA3]

        counts = Counter(residues)

        hits = []

        for rule in self.rules:

            required = Counter(rule["residues"])

            if all(counts[r] >= n for r, n in required.items()):

                # Copy (not the shared rule dict) with confidence rescaled
                # from the database's 0-1 scale to the 0-100 scale every
                # other predictor module uses.
                hits.append({
                    **rule,
                    "confidence": round(rule["confidence"] * 100),
                })

        return hits