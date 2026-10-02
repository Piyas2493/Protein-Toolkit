"""
ProteinToolkit Structure Analyzer
"""

from .parser import parse_pdb
from .contacts import ContactAnalyzer
from .atom_classifier import AtomInteractionClassifier
from .pocket_scoring import PocketScorer
from .structural_active_site import StructuralActiveSitePredictor
from .catalytic_geometry import SpatialCatalyticDetector
from .evidence_structure import EvidenceFusion
from .structural_interpretation import StructuralInterpreter
from .summary import StructureSummary


class StructureAnalyzer:

    def __init__(self, cutoff=3.5):

        self.contact_analyzer = ContactAnalyzer(cutoff)

        self.interaction_classifier = AtomInteractionClassifier()

        self.pocket_scorer = PocketScorer()

        self.structural_active_site = (
            StructuralActiveSitePredictor()
        )

        self.spatial_catalytic_detector = SpatialCatalyticDetector()

        self.evidence_fusion = EvidenceFusion()

        self.structural_interpreter = (
            StructuralInterpreter()
        )

        self.summary_engine = StructureSummary()

    def analyze(self, pdb_file):

        # --------------------------------------------------
        # Parse PDB Structure
        # --------------------------------------------------
        structure = parse_pdb(pdb_file)

        # --------------------------------------------------
        # Contact Analysis
        # --------------------------------------------------
        contacts = self.contact_analyzer.find_contacts(
            structure
        )

        # --------------------------------------------------
        # Interaction Classification
        # --------------------------------------------------
        for contact in contacts:

            contact["interaction"] = (
                self.interaction_classifier.classify(
                    contact
                )
            )

        # --------------------------------------------------
        # Binding Site Detection
        # --------------------------------------------------
        binding_sites = {}

        for contact in contacts:

            ligand = contact["ligand"]

            residue = (
                contact["chain"],
                contact["residue"],
                contact["residue_number"]
            )

            if ligand not in binding_sites:

                binding_sites[ligand] = set()

            binding_sites[ligand].add(
                residue
            )

        # --------------------------------------------------
        # Pocket Scoring
        # --------------------------------------------------
        pocket_scores = self.pocket_scorer.score(
            binding_sites,
            contacts
        )

        # --------------------------------------------------
        # Structural Active Site Prediction
        # --------------------------------------------------
        active_sites = (
            self.structural_active_site.predict(
                contacts
            )
        )

        # --------------------------------------------------
        # Spatial Catalytic-Site Detection
        # --------------------------------------------------
        spatial_catalytic_sites = (
            self.spatial_catalytic_detector.predict(
                structure
            )
        )

        # --------------------------------------------------
        # Evidence Fusion
        # --------------------------------------------------
        final_predictions = (
            self.evidence_fusion.fuse(
                active_sites,
                pocket_scores
            )
        )

        # --------------------------------------------------
        # Structural Interpretation
        # --------------------------------------------------
        interpretation = (
            self.structural_interpreter.interpret(
                final_predictions
            )
        )

        # --------------------------------------------------
        # Structural Summary
        # --------------------------------------------------
        summary = self.summary_engine.generate({

            "atoms": len(structure.atoms),

            "chains": structure.chains,

            "ligands": structure.ligands,

            "binding_sites": binding_sites,

            "pocket_scores": pocket_scores,

            "active_sites": active_sites,

            "final_predictions": final_predictions

        })

        # --------------------------------------------------
        # Final Results
        # --------------------------------------------------
        return {

            "atoms": len(structure.atoms),

            "chains": structure.chains,

            "ligands": structure.ligands,

            "contacts": contacts,

            "binding_sites": binding_sites,

            "pocket_scores": pocket_scores,

            "active_sites": active_sites,

            "spatial_catalytic_sites": spatial_catalytic_sites,

            "final_predictions": final_predictions,

            "interpretation": interpretation,

            "summary": summary

        }