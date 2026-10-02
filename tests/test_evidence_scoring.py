from core.enzymes_classifier import EnzymeClassifier
from core.scoring import ConfidenceScorer
from core.evidence import EvidenceIntegrator
from core.ligand_prediction import LigandPredictor


class _FakeStats:
    def gravy(self):
        return 0.0

    def instability_index(self):
        return 10.0


def test_enzyme_classifier_reports_max_score():
    predictions = EnzymeClassifier().classify(
        "putative protease", []
    )
    assert predictions
    assert predictions[0]["max_score"] >= predictions[0]["score"] > 0


def test_scoring_weights_enzyme_evidence_by_strength():
    weak = [{"class": "Hydrolase", "score": 2, "max_score": 11}]
    strong = [{"class": "Hydrolase", "score": 11, "max_score": 11}]

    scorer = ConfidenceScorer()
    weak_result = scorer.calculate(
        motif_hits=[], catalytic_hits=[],
        enzyme_predictions=weak, statistics=_FakeStats(),
    )
    strong_result = scorer.calculate(
        motif_hits=[], catalytic_hits=[],
        enzyme_predictions=strong, statistics=_FakeStats(),
    )
    assert weak_result["breakdown"]["Enzyme"] < strong_result["breakdown"]["Enzyme"]
    assert strong_result["breakdown"]["Enzyme"] == 20


def test_evidence_integrator_weights_enzyme_evidence_by_strength():
    weak = [{"class": "Hydrolase", "score": 2, "max_score": 11}]
    integrator = EvidenceIntegrator()
    result = integrator.integrate(
        motif_hits=[], catalytic_hits=[], enzyme_predictions=weak,
        function_predictions=[], ligand_predictions=[], active_sites=[],
    )
    assert 0 < result["score"] < 20


def test_ligand_correlated_single_signal_does_not_reach_max():
    predictor = LigandPredictor()
    ligand = predictor.database[0]

    predictions = predictor.predict(
        motif_hits=[{"name": m} for m in ligand["motifs"]],
        enzyme_predictions=[{"class": c, "score": 1, "max_score": 1}
                             for c in ligand["enzyme_classes"]],
        function_predictions=[{"function": f} for f in ligand["functions"]],
    )
    top = next(p for p in predictions if p["ligand"] == ligand["name"])
    assert top["confidence"] < 100
