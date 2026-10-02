"""
Protein Report Engine

Creates formatted reports from analysis results.
"""


class Report:

    @staticmethod
    def title(text):

        print()

        print("=" * 70)

        print(text)

        print("=" * 70)

    @staticmethod
    def line(label, value):

        print(f"{label:25} : {value}")

    @staticmethod
    def blank():

        print()

    @staticmethod
    def _list_report(title, items, empty_message, fields):
        """Shared skeleton for "title, then one block of labeled fields
        per item" reports. `fields` is [(label, fn(item) -> value)];
        fn may return None to skip that line for a given item."""

        Report.title(title)

        if not items:

            print(empty_message)

            return

        for item in items:

            print()

            for label, fn in fields:

                value = fn(item)

                if value is not None:

                    Report.line(label, value)

    @staticmethod
    def _bullets(heading, items):

        print()

        print(heading)

        print()

        for item in items:

            print(f"- {item}")

    @staticmethod
    def motif_report(motif_hits):

        Report._list_report(
            "MOTIF ANALYSIS", motif_hits, "No motifs detected.",
            [
                ("Motif", lambda h: h["name"]),
                ("Family", lambda h: h["family"]),
                ("Ligand", lambda h: h["ligand"]),
                ("Confidence", lambda h: f"{h['confidence']}%"),
                ("Sequence", lambda h: h["sequence"]),
                ("Region", lambda h: f"{h['start']} - {h['end']}"),
            ],
        )

    @staticmethod
    def catalytic_report(catalytic_hits):

        Report._list_report(
            "CATALYTIC RESIDUE ANALYSIS", catalytic_hits,
            "No catalytic signatures detected.",
            [
                ("Catalytic System", lambda h: h["name"]),
                ("Function", lambda h: h["function"]),
                ("Class", lambda h: h["enzyme_class"]),
                ("Confidence", lambda h: f"{h['confidence']}%"),
            ],
        )

    @staticmethod
    def enzyme_report(predictions):

        Report._list_report(
            "ENZYME CLASSIFICATION", predictions,
            "No enzyme class predicted.",
            [
                ("Class", lambda i: i["class"]),
                ("Score", lambda i: i["score"]),
            ],
        )

    @staticmethod
    def confidence_report(result):

        Report.title("GLOBAL CONFIDENCE")

        Report.line(
            "Overall",
            f"{result['confidence']}%"
        )

        print()

        print("Evidence Breakdown")

        print()

        for key, value in result["breakdown"].items():

            Report.line(key, value)

    @staticmethod
    def function_report(predictions):

        Report._list_report(
            "PROTEIN FUNCTION PREDICTION", predictions,
            "No function predicted.",
            [
                ("Function", lambda i: i.get("function", "Unknown")),
                ("Score", lambda i: i.get("score", 0)),
                ("Description", lambda i: i.get("description", "N/A")),
                ("Ligands", lambda i: ", ".join(i.get("ligands", []))),
            ],
        )

    @staticmethod
    def active_site_report(predictions):

        Report._list_report(
            "ACTIVE SITE PREDICTION", predictions,
            "No active-site detected.",
            [
                ("Predicted Site", lambda s: s.get("site", "Unknown")),
                ("Sequence Region", lambda s: s.get("region", "Unknown")),
                ("Evidence", lambda s: s.get("motif", "N/A")),
                ("Confidence", lambda s: (
                    f"{s['confidence']}%" if "confidence" in s else None
                )),
                ("Residues", lambda s: (
                    ", ".join(s["residues"]) if "residues" in s else None
                )),
            ],
        )

    @staticmethod
    def ligand_report(predictions):

        Report._list_report(
            "LIGAND PREDICTION", predictions, "No ligand predicted.",
            [
                ("Ligand", lambda l: l["ligand"]),
                ("Family", lambda l: l["family"]),
                ("Confidence", lambda l: f"{l['confidence']}%"),
            ],
        )

    @staticmethod
    def evidence_report(result):

        Report.title("EVIDENCE INTEGRATION")

        Report.line(
            "Overall Score",
            f"{result['score']} / 100"
        )

        Report.line(
            "Confidence",
            result["confidence"]
        )

        print()

        print("Evidence Breakdown")

        print()

        for key in ("motifs", "catalytic", "enzymes",
                    "functions", "ligands", "active_sites"):

            Report.line(key.replace("_", " ").title(), result[key])

        Report._bullets("Summary", result["summary"])

    @staticmethod
    def interpretation_report(result):

        Report.title("BIOLOGICAL INTERPRETATION")

        Report.line(
            "Predicted Enzyme",
            result["enzyme"]
        )

        Report.line(
            "Predicted Function",
            result["function"]
        )

        Report.line(
            "Likely Ligand",
            result["ligand"]
        )

        Report.line(
            "Active Site",
            result["active_site"]
        )

        Report._bullets("Interpretation", result["interpretation"])

    @staticmethod
    def structure_report(result):

        Report.title("STRUCTURE ANALYSIS")

        Report.line(
            "Protein Atoms",
            result["atoms"]
        )

        Report.line(
            "Chains",
            len(result["chains"])
        )

        Report.line(
            "Ligands",
            len(result["ligands"])
        )

        Report.line(
            "Contacts",
            len(result["contacts"])
        )

    @staticmethod
    def structure_summary(result):

        Report.title("STRUCTURE SUMMARY")

        Report.line(
            "Protein Atoms",
            result["atoms"]
        )

        Report.line(
            "Chains",
            len(result["chains"])
        )

        Report.line(
            "Ligands",
            len(result["ligands"])
        )

    @staticmethod
    def contact_report(contacts):

        Report._list_report(
            "LIGAND CONTACTS", contacts, "No ligand contacts detected.",
            [
                ("Ligand", lambda c: c["ligand"]),
                ("Residue", lambda c: f"{c['residue']}{c['residue_number']}"),
                ("Chain", lambda c: c["chain"]),
                ("Interaction", lambda c: c.get("interaction", "Unknown")),
                ("Distance", lambda c: f"{c['distance']} Angstrom"),
            ],
        )

    @staticmethod
    def binding_site_report(binding_sites):

        Report.title("BINDING SITE ANALYSIS")

        if not binding_sites:

            print("No binding sites detected.")

            return

        for ligand, residues in binding_sites.items():

            print()

            Report.line(
                "Ligand",
                ligand
            )

            print("Binding Residues")

            for chain, residue, number in sorted(
                residues, key=lambda r: (r[0], r[2])
            ):

                print(
                    f"  - {chain}:{residue}{number}"
                )

    @staticmethod
    def interaction_report(contacts):

        Report._list_report(
            "INTERACTION ANALYSIS", contacts, "No interactions detected.",
            [
                ("Ligand", lambda c: c["ligand"]),
                ("Residue", lambda c: f"{c['residue']}{c['residue_number']}"),
                ("Interaction", lambda c: c["interaction"]),
                ("Distance", lambda c: f"{c['distance']} Angstrom"),
            ],
        )

    @staticmethod
    def atom_interaction_report(contacts):

        Report._list_report(
            "ATOM-LEVEL INTERACTION ANALYSIS", contacts,
            "No interactions detected.",
            [
                ("Ligand", lambda c: c["ligand"]),
                ("Protein Atom", lambda c: c["atom"]),
                ("Ligand Atom", lambda c: c["ligand_atom"]),
                ("Residue", lambda c: f"{c['residue']}{c['residue_number']}"),
                ("Interaction", lambda c: c["interaction"]),
                ("Distance", lambda c: f"{c['distance']} Angstrom"),
            ],
        )

    @staticmethod
    def pocket_score_report(results):

        Report.title(
            "POCKET RANKING"
        )

        if not results:

            print("No pockets detected.")
            return

        for rank, pocket in enumerate(results, start=1):

            print()

            Report.line(
                "Rank",
                rank
            )

            Report.line(
                "Ligand",
                pocket["ligand"]
            )

            Report.line(
                "Score",
                pocket["score"]
            )

            Report.line(
                "Residues",
                pocket["residue_count"]
            )

            Report.line(
                "Interactions",
                pocket["interaction_count"]
            )

            Report.line(
                "Average Distance",
                f"{pocket['average_distance']} Angstrom"
            )
    @staticmethod
    def structural_active_site_report(results):

        Report._list_report(
            "STRUCTURAL ACTIVE SITE PREDICTION", results,
            "No structural active sites predicted.",
            [
                ("Residue", lambda r:
                    f"{r['chain']}:{r['residue']}{r['residue_number']}"),
                ("Interaction", lambda r: r["interaction"]),
                ("Distance", lambda r: f"{r['distance']} Å"),
                ("Score", lambda r: r["score"]),
            ],
        )

    @staticmethod
    def spatial_catalytic_report(hits):

        Report._list_report(
            "SPATIAL CATALYTIC SITE DETECTION", hits,
            "No spatially-clustered catalytic site found.",
            [
                ("Catalytic System", lambda h: h["name"]),
                ("Residues Found", lambda h: ", ".join(h["residues_found"])),
                ("Function", lambda h: h["function"]),
                ("Class", lambda h: h["enzyme_class"]),
            ],
        )

    @staticmethod
    def structural_evidence_report(results):

        Report._list_report(
            "STRUCTURAL EVIDENCE FUSION", results,
            "No predictions available.",
            [
                ("Residue", lambda r:
                    f"{r['chain']}:{r['residue']}{r['residue_number']}"),
                ("Interaction", lambda r: r["interaction"]),
                ("Confidence", lambda r: f"{r['confidence']} %"),
            ],
        )

    @staticmethod
    def structural_interpretation_report(results):

        Report._list_report(
            "STRUCTURAL INTERPRETATION", results,
            "No interpretation available.",
            [
                ("Residue", lambda i:
                    f"{i['chain']}:{i['residue']}{i['residue_number']}"),
                ("Confidence", lambda i: f"{i['confidence']} %"),
                ("Interpretation", lambda i: i["interpretation"]),
            ],
        )

    @staticmethod
    def structural_summary_report(summary):

        Report.title(
            "STRUCTURAL SUMMARY"
        )

        Report.line(
            "Atoms",
            summary["structure"]["atoms"]
        )

        Report.line(
            "Chains",
            summary["structure"]["chains"]
        )

        Report.line(
            "Ligands",
            summary["structure"]["ligands"]
        )

        Report.line(
            "Binding Sites",
            summary["binding_sites"]
        )

        Report.line(
            "Binding Pockets",
            summary["pockets"]
        )

        Report.line(
            "Predicted Active Sites",
            summary["predicted_active_sites"]
        )

        if summary["top_prediction"]:

            Report.line(
                "Top Residue",
                f"{summary['top_prediction']['chain']}:"
                f"{summary['top_prediction']['residue']}"
                f"{summary['top_prediction']['residue_number']}"
            )

            Report.line(
                "Confidence",
                f"{summary['top_prediction']['confidence']} %"
            )        