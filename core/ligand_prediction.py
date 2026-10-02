"""
Ligand Prediction Engine
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE = BASE_DIR / "database" / "ligands.json"


class LigandPredictor:

    def __init__(self):

        with open(DATABASE, encoding="utf-8") as f:

            self.database = json.load(f)

    def predict(

        self,

        motif_hits,

        enzyme_predictions,

        function_predictions,

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

        function_names = {

            item["function"]

            for item in function_predictions

        }

        predictions = []

        for ligand in self.database:

            hits = []

            if any(

                ec in enzyme_classes

                for ec in ligand["enzyme_classes"]

            ):

                hits.append(35)

            if any(

                motif in motif_names

                for motif in ligand["motifs"]

            ):

                hits.append(40)

            if any(

                fn in function_names

                for fn in ligand["functions"]

            ):

                hits.append(25)

            if not hits:
                continue

            # Full weight for the strongest signal; a smaller bonus for
            # each additional one. These three channels are often
            # correlated (the same single motif can drive an enzyme-class
            # match AND a function match), so treating them as fully
            # independent would let one weak motif hit reach max score.
            hits.sort(reverse=True)
            score = hits[0] + sum(h * 0.3 for h in hits[1:])

            predictions.append({

                "ligand": ligand["name"],

                "family": ligand["family"],

                "confidence": min(round(score), 100)

            })

        predictions.sort(

            key=lambda x:x["confidence"],

            reverse=True

        )

        return predictions
        