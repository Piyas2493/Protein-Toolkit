"""
Catalytic residue prediction engine

A catalytic system (serine / cysteine / aspartic / metalloprotease) is reported
only when the sequence contains a CONSERVED ACTIVE-SITE MOTIF for it
(database/catalytic_sites.json, "motifs"). The earlier rule fired whenever the
protein merely contained enough His/Ser/Asp... residues somewhere, which nearly
every protein does, so it called almost everything a protease. Each hit now
carries the matching motif and position as evidence.

The motifs are high-precision and partial-recall: benchmarks/catalytic_validation.py
measures both against UniProt EC subclasses. A protease from a family none of
the motifs covers is NOT reported, so no hit is not evidence that a protein is
not a protease. (Rules without a "motifs" list keep the composition-only
behaviour.)
"""

import json
import re
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

            # Necessary, not sufficient: the residue types must be present.
            if not all(counts[r] >= n for r, n in required.items()):
                continue

            evidence = []
            if "motifs" in rule:
                for motif in rule["motifs"]:
                    for match in re.finditer(motif["regex"], sequence):
                        evidence.append({
                            "motif": motif["name"],
                            "start": match.start() + 1,
                            "end": match.end(),
                            "sequence": match.group(),
                        })
                if not evidence:
                    continue      # composition alone is not evidence

            # Copy (not the shared rule dict) with confidence rescaled
            # from the database's 0-1 scale to the 0-100 scale every
            # other predictor module uses.
            hits.append({
                **rule,
                "confidence": round(rule["confidence"] * 100),
                "evidence": evidence,
            })

        return hits