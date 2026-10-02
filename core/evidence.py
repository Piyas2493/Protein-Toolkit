"""
Evidence Integration Engine
"""


CONFIDENCE_LABELS = (
    (90, "Very High"),
    (75, "High"),
    (60, "Moderate"),
    (40, "Low"),
    (0, "Very Low"),
)


def _enzyme_strength(enzyme_predictions):
    if not enzyme_predictions:
        return 0.0

    top = enzyme_predictions[0]

    if not top.get("max_score"):
        return 0.0

    return top["score"] / top["max_score"]


class EvidenceIntegrator:

    def integrate(
        self,
        motif_hits,
        catalytic_hits,
        enzyme_predictions,
        function_predictions,
        ligand_predictions,
        active_sites
    ):

        report = {

            "motifs": len(motif_hits),

            "catalytic": len(catalytic_hits),

            "enzymes": len(enzyme_predictions),

            "functions": len(function_predictions),

            "ligands": len(ligand_predictions),

            "active_sites": len(active_sites),

            "score": 0,

            "confidence": "",

            "summary": []

        }

        # (strength in [0,1], weight, summary message). Presence-only
        # evidence uses strength 1.0/0.0; enzyme evidence is scaled by
        # how much of that rule's own possible evidence was matched.
        components = [
            (1.0 if motif_hits else 0.0, 20,
             "Sequence motifs detected."),
            (1.0 if catalytic_hits else 0.0, 20,
             "Catalytic residues identified."),
            (_enzyme_strength(enzyme_predictions), 20,
             "Enzyme class predicted."),
            (1.0 if function_predictions else 0.0, 15,
             "Protein function inferred."),
            (1.0 if ligand_predictions else 0.0, 15,
             "Probable ligand assigned."),
            (1.0 if active_sites else 0.0, 10,
             "Active-site region predicted."),
        ]

        score = 0

        for strength, weight, message in components:

            if strength > 0:
                score += round(weight * strength)
                report["summary"].append(message)

        report["score"] = score

        report["confidence"] = next(
            label for threshold, label in CONFIDENCE_LABELS
            if score >= threshold
        )

        return report
