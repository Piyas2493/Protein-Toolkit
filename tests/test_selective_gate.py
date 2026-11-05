"""Tests for the risk-controlled triage gate (ml/selective_gate.py)."""
import numpy as np
import pytest

from ml.selective_gate import RiskControlledGate


def _draw(rng, n_pos=400, n_neg=1600):
    pos = np.clip(rng.beta(6, 3, n_pos), 0, 1)
    neg = np.clip(rng.beta(3, 6, n_neg), 0, 1)
    scores = np.concatenate([pos, neg])
    positive = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    return scores, positive


def test_guarantee_holds_across_seeds():
    """Over many fresh draws, realized recall should meet the target in at
    least (1 - delta) of them, up to Monte-Carlo noise."""
    target, delta = 0.95, 0.05
    held = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        gate = RiskControlledGate(target, delta).fit(*_draw(rng))
        res = gate.evaluate(*_draw(np.random.default_rng(seed + 1000)))
        held.append(res["realized_recall"] >= target)
    assert np.mean(held) >= 1 - delta - 0.03  # small slack for finite samples


def test_transform_matches_threshold():
    rng = np.random.default_rng(1)
    gate = RiskControlledGate(0.9, 0.1).fit(*_draw(rng))
    s = np.array([0.0, gate.threshold_, 1.0])
    assert gate.transform(s).tolist() == [False, True, True]


def test_higher_target_sends_more_sequences():
    """A stricter miss-rate target lowers the threshold, so the gate keeps
    at least as many sequences (never fewer)."""
    rng = np.random.default_rng(2)
    scores, positive = _draw(rng)
    loose = RiskControlledGate(0.80, 0.05).fit(scores, positive)
    strict = RiskControlledGate(0.99, 0.05).fit(scores, positive)
    assert strict.threshold_ <= loose.threshold_
    assert strict.report_.kept_fraction >= loose.report_.kept_fraction


def test_infeasible_keeps_all():
    """If the target cannot be guaranteed from the data, the gate keeps every
    sequence and flags itself infeasible rather than silently dropping some."""
    rng = np.random.default_rng(3)
    scores, positive = _draw(rng, n_pos=5, n_neg=50)  # too few positives
    gate = RiskControlledGate(0.999, 0.001).fit(scores, positive)
    assert gate.threshold_ == 0.0
    assert gate.report_.feasible is False
    assert gate.transform(scores).all()


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        RiskControlledGate(target_recall=1.0)
    with pytest.raises(ValueError):
        RiskControlledGate(delta=0.0)


def test_triage_fails_closed_on_unreadable_calibration(tmp_path):
    """A requested gate that can't be built must exit non-zero and write
    nothing, not silently fall back to ungated output."""
    import triage

    fasta = tmp_path / "s.fasta"
    fasta.write_text(">a\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV\n")
    out = tmp_path / "out.csv"
    rc = triage.main([str(fasta), "--out", str(out), "--no-ml",
                      "--gate-calibration", str(tmp_path / "missing.csv")])
    assert rc == 2 and not out.exists()
