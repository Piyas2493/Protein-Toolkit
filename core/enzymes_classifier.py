"""
Simple enzyme classification engine.
"""

import json

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE = BASE_DIR / "database" / "enzymes.json"


class EnzymeClassifier:

    def __init__(self):

        with open(
            DATABASE,
            encoding="utf-8"
        ) as f:

            self.rules = json.load(f)

    def classify(
        self,
        name,
        motif_hits
    ):

        motif_names = {
            m["name"]
            for m in motif_hits
        }

        name = name.lower()

        predictions = []

        for rule in self.rules:

            score = 0

            for keyword in rule["keywords"]:

                if keyword.lower() in name:
                    score += 2

            for motif in rule["motifs"]:

                if motif in motif_names:
                    score += 3

            if score:

                # Every rule's own ceiling (all keywords + all motifs
                # matched), so consumers can weight by *how much* of this
                # rule's evidence was found, not just whether any was.
                max_score = len(rule["keywords"]) * 2 + len(rule["motifs"]) * 3

                predictions.append({

                    "class": rule["class"],

                    "score": score,

                    "max_score": max_score

                })

        predictions.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return predictions