"""
ProteinToolkit Structural Active Site Predictor
"""


class StructuralActiveSitePredictor:

    CATALYTIC_RESIDUES = {
        "ASP",
        "GLU",
        "HIS",
        "LYS",
        "SER",
        "CYS",
        "ARG",
        "TYR"
    }

    def predict(self, contacts):

        predictions = []
        seen = set()

        for contact in contacts:

            residue = contact["residue"]

            if residue not in self.CATALYTIC_RESIDUES:
                continue

            key = (
                contact["chain"],
                contact["residue"],
                contact["residue_number"]
            )

            if key in seen:
                continue

            seen.add(key)

            score = self.score(contact)

            predictions.append({

                "chain": contact["chain"],

                "residue": residue,

                "residue_number": contact["residue_number"],

                "interaction": contact["interaction"],

                "distance": contact["distance"],

                "score": score,

                "confidence": score

            })

        predictions.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return predictions

    def score(self, contact):

        score = 0

        distance = contact["distance"]

        if distance <= 2.5:
            score += 40
        elif distance <= 3.0:
            score += 30
        elif distance <= 3.5:
            score += 20

        interaction = contact["interaction"]

        if interaction == "Hydrogen Bond":
            score += 30

        elif interaction == "Metal Coordination":
            score += 35

        elif interaction.endswith("Ionic"):
            score += 25

        score += 20

        return min(score, 100)