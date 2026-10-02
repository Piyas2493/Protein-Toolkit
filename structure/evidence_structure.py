"""
ProteinToolkit Structural Evidence Fusion
"""


class EvidenceFusion:

    def fuse(
        self,
        structural_sites,
        pocket_scores
    ):

        results = []

        for residue in structural_sites:

            score = residue["score"]

            # Add pocket score if residue belongs to the top-ranked pocket
            if pocket_scores:
                score += pocket_scores[0]["score"] * 0.20

            confidence = self.confidence(score)

            results.append({

                "chain": residue["chain"],

                "residue": residue["residue"],

                "residue_number": residue["residue_number"],

                "interaction": residue["interaction"],

                "confidence": round(confidence, 2)

            })

        results.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return results

    @staticmethod
    def confidence(score):

        if score > 100:
            score = 100

        return score