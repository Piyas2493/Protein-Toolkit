"""
ProteinToolkit Interaction Classifier
"""


class InteractionClassifier:

    POLAR = {
        "SER", "THR", "ASN", "GLN",
        "HIS", "TYR", "CYS"
    }

    POSITIVE = {
        "LYS", "ARG", "HIS"
    }

    NEGATIVE = {
        "ASP", "GLU"
    }

    HYDROPHOBIC = {
        "ALA", "VAL", "LEU", "ILE",
        "MET", "PHE", "TRP", "PRO"
    }

    AROMATIC = {
        "PHE", "TYR", "TRP", "HIS"
    }

    def classify(self, contact):

        residue = contact["residue"]

        distance = contact["distance"]

        if residue in self.POLAR and distance <= 3.5:
            return "Hydrogen Bond"

        if residue in self.POSITIVE:
            return "Positive Ionic"

        if residue in self.NEGATIVE:
            return "Negative Ionic"

        if residue in self.AROMATIC:
            return "Aromatic"

        if residue in self.HYDROPHOBIC:
            return "Hydrophobic"

        return "Van der Waals"