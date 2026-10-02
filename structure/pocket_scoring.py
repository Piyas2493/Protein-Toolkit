"""
ProteinToolkit Pocket Scoring Engine
"""


class PocketScorer:

    def score(self, binding_sites, contacts):

        results = []

        for ligand, residues in binding_sites.items():

            ligand_contacts = [
                c for c in contacts
                if c["ligand"] == ligand
            ]

            residue_count = len(residues)

            interaction_count = len(ligand_contacts)

            avg_distance = (
                sum(c["distance"] for c in ligand_contacts)
                / interaction_count
                if interaction_count
                else 999
            )

            score = 0

            # Number of unique residues
            score += min(residue_count * 10, 40)

            # Number of interactions
            score += min(interaction_count * 2, 30)

            # Shorter average distance is better
            if avg_distance <= 2.5:
                score += 30
            elif avg_distance <= 3.0:
                score += 25
            elif avg_distance <= 3.5:
                score += 20
            else:
                score += 10

            results.append({

                "ligand": ligand,

                "residue_count": residue_count,

                "interaction_count": interaction_count,

                "average_distance": round(avg_distance, 2),

                "score": min(score, 100)

            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results