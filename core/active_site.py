"""
Protein Active Site Prediction Engine

Version 1
"""

import json

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE = BASE_DIR / "database" / "active_site_rules.json"


class ActiveSitePredictor:

    def __init__(self):

        with open(
            DATABASE,
            encoding="utf-8"
        ) as f:

            self.rules = json.load(f)

    def predict(

        self,

        motif_hits,

        sequence_length

    ):

        predictions = []

        for hit in motif_hits:

            for rule in self.rules:

                if hit["name"] != rule["motif"]:
                    continue

                start = max(
                    1,
                    hit["start"] - rule["radius"]
                )

                end = min(
                    sequence_length,
                    hit["end"] + rule["radius"]
                )

                predictions.append(

                    {

                        "site":

                        rule["site"],

                        "region":

                        f"{start}-{end}",

                        "motif":

                        hit["name"],

                        "matched":

                        hit["sequence"],

                        "confidence":

                        rule["confidence"]

                    }

                )

        predictions.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return predictions