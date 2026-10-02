"""
Protein Function Prediction Engine
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE = BASE_DIR / "database" / "function.json"


class FunctionPredictor:

    def __init__(self):

        with open(
            DATABASE,
            encoding="utf-8"
        ) as f:

            self.rules = json.load(f)

    def predict(

        self,

        motif_hits,

        enzyme_predictions,

        catalytic_hits=None

    ):

        motif_names = {

            hit["name"]

            for hit in motif_hits

        }

        enzyme_classes = {

            item["class"]

            for item in enzyme_predictions

        }

        # Catalytic-residue evidence (e.g. a confirmed protease triad)
        # names an enzyme_class too, even when no motif/keyword match
        # produced a rule-based enzyme prediction — don't discard it.
        enzyme_classes |= {
            hit["enzyme_class"]
            for hit in (catalytic_hits or [])
        }

        predictions = []

        for rule in self.rules:

            score = 0

            if rule["enzyme_class"] in enzyme_classes:

                score += 40

            for motif in rule["motifs"]:

                if motif in motif_names:

                    score += 60

            if score:

                predictions.append({

                    "function":rule["name"],

                    "description":rule["description"],

                    "score":min(score, 100),

                    "ligands":rule["ligands"]

                })

        predictions.sort(

            key=lambda x:x["score"],

            reverse=True

        )

        return predictions
