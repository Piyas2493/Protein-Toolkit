"""
ProteinToolkit Structural Summary Engine
"""


class StructureSummary:

    def generate(self, analysis):

        summary = {

            "structure": {

                "atoms": analysis["atoms"],

                "chains": len(analysis["chains"]),

                "ligands": len(analysis["ligands"])

            },

            "binding_sites": len(
                analysis["binding_sites"]
            ),

            "pockets": len(
                analysis["pocket_scores"]
            ),

            "predicted_active_sites": len(
                analysis["active_sites"]
            ),

            "top_prediction": None

        }

        if analysis["final_predictions"]:

            summary["top_prediction"] = (
                analysis["final_predictions"][0]
            )

        return summary