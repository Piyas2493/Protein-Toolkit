"""
ProteinToolkit Structural Interpretation Engine
"""


class StructuralInterpreter:

    def interpret(self, predictions):

        results = []

        for prediction in predictions:

            interpretation = self._generate_interpretation(prediction)

            results.append({

                "chain": prediction["chain"],

                "residue": prediction["residue"],

                "residue_number": prediction["residue_number"],

                "interaction": prediction["interaction"],

                "confidence": prediction["confidence"],

                "interpretation": interpretation

            })

        return results

    def _generate_interpretation(self, prediction):

        confidence = prediction["confidence"]

        interaction = prediction["interaction"]

        if confidence >= 90:

            level = "Highly likely"

        elif confidence >= 75:

            level = "Likely"

        elif confidence >= 60:

            level = "Possible"

        else:

            level = "Weak"

        return (
            f"{level} catalytic or ligand-binding residue "
            f"supported by {interaction.lower()}."
        )