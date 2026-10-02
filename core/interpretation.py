"""
Biological Interpretation Engine
"""


class InterpretationEngine:

    def generate(
        self,
        enzyme_predictions,
        function_predictions,
        ligand_predictions,
        active_sites,
        motif_hits,
        catalytic_hits
    ):

        summary = {}

        summary["enzyme"] = (
            enzyme_predictions[0]["class"]
            if enzyme_predictions
            else "Unknown"
        )

        summary["function"] = (
            function_predictions[0]["function"]
            if function_predictions
            else "Unknown"
        )

        summary["ligand"] = (
            ligand_predictions[0]["ligand"]
            if ligand_predictions
            else "Unknown"
        )

        summary["active_site"] = (
            active_sites[0]["region"]
            if active_sites
            else "Not Predicted"
        )

        summary["motifs"] = len(motif_hits)
        summary["catalytic"] = len(catalytic_hits)

        interpretation = []

        if summary["enzyme"] != "Unknown":
            interpretation.append(
                f"This protein is most consistent with a {summary['enzyme']}."
            )

        if summary["function"] != "Unknown":
            interpretation.append(
                f"The predicted molecular function is '{summary['function']}'."
            )

        if summary["ligand"] != "Unknown":
            interpretation.append(
                f"The most likely interacting ligand is {summary['ligand']}."
            )

        if summary["active_site"] != "Not Predicted":
            interpretation.append(
                f"The primary active-site region is predicted around {summary['active_site']}."
            )

        if summary["motifs"] > 0:
            interpretation.append(
                f"{summary['motifs']} conserved motif(s) support this prediction."
            )

        if summary["catalytic"] > 0:
            interpretation.append(
                f"{summary['catalytic']} catalytic residue(s) were identified."
            )

        summary["interpretation"] = interpretation

        return summary
    