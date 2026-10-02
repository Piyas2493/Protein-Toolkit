"""
ProteinToolkit PDB Parser
"""

from collections import defaultdict

from .models import (
    Atom,
    Chain,
    Ligand,
    ProteinStructure,
)


def parse_pdb(pdb_file):

    atoms = []
    chains = {}
    ligands = {}

    with open(pdb_file, "r", encoding="utf-8") as file:

        for line in file:

            record = line[:6].strip()

            if record not in ("ATOM", "HETATM"):
                continue

            atom = Atom(

                serial=int(line[6:11]),

                atom_name=line[12:16].strip(),

                residue_name=line[17:20].strip(),

                chain_id=line[21].strip(),

                residue_number=int(line[22:26]),

                x=float(line[30:38]),

                y=float(line[38:46]),

                z=float(line[46:54]),

                occupancy=float(line[54:60]),

                b_factor=float(line[60:66]),

                element=line[76:78].strip()

            )

            atoms.append(atom)

            # Protein chains
            if record == "ATOM":

                if atom.chain_id not in chains:

                    chains[atom.chain_id] = Chain(
                        chain_id=atom.chain_id
                    )

                chains[atom.chain_id].atoms.append(atom)

            # Ligands
            elif record == "HETATM":

                if atom.residue_name == "HOH":
                    continue

                key = (
                    atom.chain_id,
                    atom.residue_name,
                    atom.residue_number
                )

                if key not in ligands:

                    ligands[key] = Ligand(

                        residue_name=atom.residue_name,

                        chain_id=atom.chain_id,

                        residue_number=atom.residue_number

                    )

                ligands[key].atoms.append(atom)

    return ProteinStructure(

        atoms=atoms,

        chains=list(chains.values()),

        ligands=list(ligands.values())

    )