"""
ProteinToolkit Contact Analysis
"""

from math import sqrt


class ContactAnalyzer:

    def __init__(self, cutoff=3.5):
        self.cutoff = cutoff

    @staticmethod
    def distance(atom1, atom2):
        return sqrt(
            (atom1.x - atom2.x) ** 2 +
            (atom1.y - atom2.y) ** 2 +
            (atom1.z - atom2.z) ** 2
        )

    def find_contacts(self, structure):

        contacts = []

        for ligand in structure.ligands:

            for ligand_atom in ligand.atoms:

                for chain in structure.chains:

                    for atom in chain.atoms:

                        d = self.distance(atom, ligand_atom)

                        if d <= self.cutoff:

                            contacts.append({

                                "ligand": ligand.residue_name,

                                "chain": atom.chain_id,

                                "residue": atom.residue_name,

                                "residue_number": atom.residue_number,

                                "atom": atom.atom_name,

                                "ligand_atom": ligand_atom.atom_name,

                                "distance": round(d, 2)

                            })

        return contacts