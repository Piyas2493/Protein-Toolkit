"""
ProteinToolkit Main Pipeline
"""

from pathlib import Path
import sys

from core.sequence import ProteinSequence
from core.validator import SequenceValidator
from core.statistics import SequenceStatistics
from core.motifs import MotifDatabase
from core.catalytic import CatalyticPredictor
from core.enzymes_classifier import EnzymeClassifier
from core.scoring import ConfidenceScorer
from core.Report import Report
from core.function_prediction import FunctionPredictor
from core.active_site import ActiveSitePredictor
from core.ligand_prediction import LigandPredictor
from core.evidence import EvidenceIntegrator
from core.interpretation import InterpretationEngine
from core.exporter import ExportEngine

from structure.analyzer import StructureAnalyzer
from structure.fetch import fetch_pdb, looks_like_pdb_id

# Machine Learning
from ml.predictor import EnzymePredictor


# ============================================================
# INPUT
# ============================================================

seq = input("Protein Sequence : ")

seq = SequenceValidator.clean(seq)

validation = SequenceValidator.validate(seq)

print(validation)

if not validation["valid"]:

    print()
    print("Sequence contains unsupported amino acids:")

    print(
        "Unsupported:",
        " ".join(validation["unsupported"])
    )

    if validation["ambiguous"]:

        print(
            "Ambiguous:",
            " ".join(validation["ambiguous"])
        )

    raise SystemExit(1)


# ============================================================
# PROTEIN OBJECT
# ============================================================

protein = ProteinSequence(
    "Unknown Protein",
    seq
)

stats = SequenceStatistics(seq)


# ============================================================
# SEQUENCE SUMMARY
# ============================================================

print()
print("=" * 60)
print("SEQUENCE SUMMARY")
print("=" * 60)

print("Length :", protein.length())

print(
    "Molecular Weight :",
    stats.molecular_weight(),
    "Da"
)

print(
    "Isoelectric Point :",
    stats.isoelectric_point()
)

print(
    "GRAVY :",
    stats.gravy()
)

print(
    "Instability Index :",
    stats.instability_index()
)

print(
    "Aromaticity :",
    stats.aromaticity()
)


# ============================================================
# SECONDARY STRUCTURE
# ============================================================

print()
print("=" * 60)
print("SECONDARY STRUCTURE")
print("=" * 60)

for k, v in stats.secondary_structure().items():

    print(
        f"{k:20} {v}%"
    )


# ============================================================
# EXTINCTION COEFFICIENT
# ============================================================

print()
print("=" * 60)
print("EXTINCTION COEFFICIENT")
print("=" * 60)

for k, v in stats.extinction_coefficients().items():

    print(k, ":", v)


# ============================================================
# AMINO ACID COMPOSITION
# ============================================================

print()
print("=" * 60)
print("AMINO ACID COMPOSITION")
print("=" * 60)

for aa, percent in stats.amino_acid_percent().items():

    print(
        f"{aa:2} {percent:6.2f}%"
    )


# ============================================================
# MOTIF SEARCH
# ============================================================

motifs_db = MotifDatabase()

Motif_hits = motifs_db.search(seq)

print()
print("=" * 60)
print("MOTIF SEARCH")
print("=" * 60)

if not Motif_hits:

    print("No motifs detected")

else:

    for hit in Motif_hits:

        print()

        print(hit["name"])

        print(
            "Family :",
            hit["family"]
        )

        print(
            "Ligand :",
            hit["ligand"]
        )

        print(
            "Confidence :",
            hit["confidence"]
        )

        print(
            "Region :",
            hit["start"],
            "-",
            hit["end"]
        )

        print(
            "Matched :",
            hit["sequence"]
        )


# ============================================================
# CATALYTIC RESIDUE ANALYSIS
# ============================================================

predictor = CatalyticPredictor()

Catalytic_hits = predictor.predict(seq)

print()
print("=" * 60)
print("CATALYTIC RESIDUE ANALYSIS")
print("=" * 60)

if not Catalytic_hits:

    print("No catalytic signatures detected.")

else:

    for hit in Catalytic_hits:

        print()

        print(
            "Catalytic System :",
            hit["name"]
        )

        print(
            "Function :",
            hit["function"]
        )

        print(
            "Class :",
            hit["enzyme_class"]
        )

        print(
            "Confidence :",
            hit["confidence"]
        )


# ============================================================
# RULE / MOTIF BASED ENZYME CLASSIFICATION
# ============================================================

classifier = EnzymeClassifier()

enzyme_predictions = classifier.classify(

    "unknown protein",

    Motif_hits

)

print()
print("=" * 60)
print("ENZYME CLASSIFICATION")
print("=" * 60)

if not enzyme_predictions:

    print("No enzyme class predicted.")

else:

    for item in enzyme_predictions:

        print(
            f"{item['class']}  "
            f"Score: {item['score']}"
        )


# ============================================================
# MACHINE LEARNING ENZYME CLASSIFICATION
# ============================================================

print()
print("=" * 60)
print("MACHINE LEARNING ENZYME CLASSIFICATION")
print("=" * 60)

ml_result = None

# Automatically locate the trained model. XGBoost is both more accurate
# and far smaller on disk than the RandomForest alternative (see
# ml/evaluation.py / README limitations) — use it as the default.
model_path = (

    Path(__file__).resolve().parent
    / "ml"
    / "models"
    / "enzyme_classifier_xgb.pkl"

)

try:

    ml_predictor = EnzymePredictor(
        str(model_path)
    )

    ml_result = ml_predictor.predict_sequence(
        seq
    )

    print()

    print(
        "Predicted Enzyme Class :",
        ml_result["prediction"]
    )

    print()

    print("CLASS PROBABILITIES")

    print("-" * 40)

    for enzyme_class, probability in sorted(

        ml_result["probabilities"].items(),

        key=lambda item: item[1],

        reverse=True

    ):

        print(
            f"{enzyme_class:10} "
            f"{probability * 100:8.2f}%"
        )

except FileNotFoundError as error:

    print()
    print("ML MODEL NOT FOUND")

    print(error)

except Exception as error:

    print()
    print("ML PREDICTION FAILED")

    print(
        type(error).__name__,
        ":",
        error
    )


# ============================================================
# CONFIDENCE SCORING
# ============================================================

scorer = ConfidenceScorer()

confidence = scorer.calculate(

    motif_hits=Motif_hits,

    catalytic_hits=Catalytic_hits,

    enzyme_predictions=enzyme_predictions,

    statistics=stats

)


# ============================================================
# REPORTING
# ============================================================

Report.motif_report(
    Motif_hits
)

Report.catalytic_report(
    Catalytic_hits
)

Report.enzyme_report(
    enzyme_predictions
)

Report.confidence_report(
    confidence
)


# ============================================================
# FUNCTION PREDICTION
# ============================================================

function_predictor = FunctionPredictor()

function_predictions = function_predictor.predict(

    Motif_hits,

    enzyme_predictions,

    Catalytic_hits

)

Report.function_report(
    function_predictions
)


# ============================================================
# ACTIVE SITE PREDICTION
# ============================================================

active_predictor = ActiveSitePredictor()

active_sites = active_predictor.predict(

    Motif_hits,

    len(seq)

)

Report.active_site_report(
    active_sites
)


# ============================================================
# LIGAND PREDICTION
# ============================================================

ligand_predictor = LigandPredictor()

ligand_predictions = ligand_predictor.predict(

    Motif_hits,

    enzyme_predictions,

    function_predictions,

    Catalytic_hits

)

Report.ligand_report(
    ligand_predictions
)


# ============================================================
# EVIDENCE INTEGRATION
# ============================================================

integrator = EvidenceIntegrator()

evidence = integrator.integrate(

    Motif_hits,

    Catalytic_hits,

    enzyme_predictions,

    function_predictions,

    ligand_predictions,

    active_sites

)

Report.evidence_report(
    evidence
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

interpreter = InterpretationEngine()

interpretation = interpreter.generate(

    enzyme_predictions,

    function_predictions,

    ligand_predictions,

    active_sites,

    Motif_hits,

    Catalytic_hits

)

Report.interpretation_report(
    interpretation
)


# ============================================================
# EXPORT RESULTS
# ============================================================

results = {

    "Motifs": Motif_hits,

    "Catalytic Residues": Catalytic_hits,

    "Rule Based Enzyme Prediction": enzyme_predictions,

    "Machine Learning Prediction": ml_result,

    "Function Prediction": function_predictions,

    "Ligand Prediction": ligand_predictions,

    "Active Sites": active_sites,

    "Confidence": confidence,

    "Evidence": evidence,

    "Interpretation": interpretation

}

exporter = ExportEngine()

exporter.export_json(

    "protein_report",

    results

)

exporter.export_csv(

    "protein_report",

    evidence

)

exporter.export_text(

    "protein_report",

    interpretation

)

exporter.export_html(

    "protein_report",

    results

)


# ============================================================
# OPTIONAL STRUCTURE ANALYSIS
# ============================================================

structure_result = None

if len(sys.argv) > 1:

    arg = sys.argv[1]

    pdb_file = Path(arg)

    fetch_error = None

    if not pdb_file.exists() and looks_like_pdb_id(arg):

        print()

        print(f"Fetching {arg.upper()} from RCSB...")

        try:

            pdb_file = fetch_pdb(arg)

        except ValueError as error:

            fetch_error = str(error)

    if fetch_error:

        print()

        print(
            "Structure analysis skipped:"
        )

        print(fetch_error)

    elif not pdb_file.exists():

        print()

        print(
            "Structure analysis skipped:"
        )

        print(
            f"PDB file not found: {pdb_file}"
        )

    else:

        structure = StructureAnalyzer(

            cutoff=3.5

        )

        structure_result = structure.analyze(

            str(pdb_file)

        )

        Report.structure_report(
            structure_result
        )

        Report.structure_summary(
            structure_result
        )

        Report.contact_report(
            structure_result["contacts"]
        )

        Report.binding_site_report(
            structure_result["binding_sites"]
        )

        Report.atom_interaction_report(
            structure_result["contacts"]
        )

        Report.pocket_score_report(
            structure_result["pocket_scores"]
        )

        Report.spatial_catalytic_report(
            structure_result["spatial_catalytic_sites"]
        )


# ============================================================
# STRUCTURAL EVIDENCE
# ============================================================

if structure_result is not None:

    Report.structural_active_site_report(

        structure_result["active_sites"]

    )

    Report.structural_evidence_report(

        structure_result["final_predictions"]

    )

    Report.structural_interpretation_report(

        structure_result["interpretation"]

    )