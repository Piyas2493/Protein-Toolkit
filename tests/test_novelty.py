import json
import random

import pytest

from core import reliability
from ml.novelty import NoveltyIndex, NoveltyModel, load_if_built

BASE = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
        "HSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSHDGSHVSHVSQEWRSALNSEKLLETSIHHSTGVV")


def _model():
    other = "".join(random.Random(1).choices("ACDEFGHIKLMNPQRSTVWY", k=len(BASE)))
    # intercept/coef chosen so that more similar => more trustworthy
    return NoveltyModel({"fam_a": BASE, "fam_b": other},
                        coef=[0.8, 0.5], intercept=1.0, edges=[0.02, 0.06],
                        tier_stats={"novel": {"accuracy": 0.46}})


def test_similarity_is_one_for_a_family_member_and_near_zero_when_shuffled():
    idx = NoveltyIndex({"fam_a": BASE})
    assert idx.nearest(BASE) == (1.0, "fam_a")
    shuffled = list(BASE)
    random.Random(0).shuffle(shuffled)           # same composition, no homology
    sim, _ = idx.nearest("".join(shuffled))
    assert sim < 0.05


def test_adjusted_confidence_rises_with_similarity_and_tiers_follow_edges():
    m = _model()
    assert m.adjust(0.7, 0.001) < m.adjust(0.7, 0.05) < m.adjust(0.7, 0.9)
    assert m.tier(0.01) == "novel" and m.tier(0.03) == "related"
    assert m.tier(0.06) == "familiar"
    out = m.assess(BASE, 0.9)
    assert out["novelty"] == "familiar" and out["nn_id"] == "fam_a"


def test_save_load_round_trip_and_missing_artifact(tmp_path):
    m = _model()
    path = tmp_path / "novelty.pkl"
    m.save(path)
    again = load_if_built(path)
    assert again.coef == m.coef and again.edges == m.edges
    assert again.assess(BASE, 0.5) == m.assess(BASE, 0.5)
    assert load_if_built(tmp_path / "absent.pkl") is None


def test_triage_adds_novelty_columns_only_when_a_novelty_model_is_given():
    from core.catalytic import CatalyticPredictor
    from core.enzymes_classifier import EnzymeClassifier
    from core.fasta import FASTAParser  # noqa: F401  (import check)
    from core.motifs import MotifDatabase
    from core.scoring import ConfidenceScorer
    from triage import triage_one

    class Rec:
        header, sequence = "q", BASE

    class StubML:
        def predict_sequence(self, seq):
            return {"prediction": "EC3", "probabilities": {"EC3": 0.8, "EC1": 0.2}}

    args = (Rec(), MotifDatabase(), CatalyticPredictor(), EnzymeClassifier(),
            ConfidenceScorer(), StubML())
    with_nv = triage_one(*args, novelty=_model())
    assert with_nv["novelty"] == "familiar" and "adj_confidence" in with_nv
    assert "novelty" not in triage_one(*args)


def test_reliability_notes_read_numbers_from_the_result_files(tmp_path, monkeypatch):
    monkeypatch.setattr(reliability, "RESULTS", tmp_path)
    assert reliability.docking_notes() == [] and reliability.ec_notes() == []
    assert reliability.format_notes([]) == ""

    (tmp_path / "screening_ca2.json").write_text(json.dumps(
        {"pdb": "1CA2", "roc_auc": 0.49, "roc_auc_95ci": [0.38, 0.61], "ef_top10pct": 0.7}))
    (tmp_path / "novelty_calibration.json").write_text(json.dumps(
        {"n_heldout": 100, "accuracy": 0.66, "tiers": {
            "novel": {"accuracy": 0.46}, "related": {"accuracy": 0.65},
            "familiar": {"accuracy": 0.90}}}))
    text = reliability.format_notes(reliability.docking_notes() + reliability.ec_notes())
    assert "1CA2 ROC AUC 0.49" in text and "46% for novel" in text
