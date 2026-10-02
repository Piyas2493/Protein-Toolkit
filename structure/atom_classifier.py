"""
ProteinToolkit Atom-Level Interaction Classifier
"""


class AtomInteractionClassifier:

    OXYGEN = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH"}

    NITROGEN = {"N", "NZ", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2"}

    SULFUR = {"SG", "SD"}

    METALS = {
        "ZN",
        "MG",
        "MN",
        "FE",
        "CU",
        "CA",
        "CO",
        "NI"
    }

    def classify(self, contact):

        protein_atom = contact["atom"]
        ligand_atom = contact["ligand_atom"]
        ligand = contact["ligand"]
        distance = contact["distance"]

        if distance <= 3.5:

            if protein_atom in self.OXYGEN and ligand_atom in self.OXYGEN:
                return "Hydrogen Bond"

            if protein_atom in self.NITROGEN:
                return "Hydrogen Bond"

        if ligand in self.METALS:
            return "Metal Coordination"

        if protein_atom in self.SULFUR:
            return "Sulfur Interaction"

        return "Contact"