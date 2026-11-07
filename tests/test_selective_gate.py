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


# ---- cluster-aware calibration -------------------------------------------

def _clustered(rng, n_clusters):
    """Heavy-tailed cluster sizes; scores within a cluster share a random
    effect, so rows are strongly correlated (like homologous sequences)."""
    sizes = 1 + np.floor(rng.lognormal(0.0, 1.3, n_clusters)).astype(int)
    cid = np.repeat(np.arange(n_clusters), sizes)
    mu = rng.normal(0, 0.8, n_clusters)[cid]
    score = 1 / (1 + np.exp(-(1.2 + mu + rng.normal(0, 0.6, cid.size))))
    return score, cid


def test_cluster_gate_covers_cluster_averaged_recall_row_gate_does_not():
    """With correlated rows the row-level bound undercovers; fitting with
    cluster ids restores the promised coverage of cluster-averaged recall."""
    target, delta = 0.95, 0.05
    pop, pcid = _clustered(np.random.default_rng(12345), 20000)
    sizes = np.bincount(pcid)

    def cluster_recall(tau):
        return float(np.mean(np.bincount(pcid, weights=(pop >= tau)) / sizes))

    rng = np.random.default_rng(0)
    hit_cluster, hit_row = [], []
    for r in range(150):
        s, cid = _clustered(rng, 150)
        ones = np.ones(s.size, bool)
        gc = RiskControlledGate(target, delta, seed=r).fit(s, ones, groups=cid)
        gr = RiskControlledGate(target, delta).fit(s, ones)
        hit_cluster.append(cluster_recall(gc.threshold_) >= target)
        hit_row.append(cluster_recall(gr.threshold_) >= target)
    assert np.mean(hit_cluster) >= 1 - delta - 0.05
    assert np.mean(hit_row) < np.mean(hit_cluster) - 0.1


def test_groups_use_one_unit_per_cluster_and_are_reproducible():
    s, cid = _clustered(np.random.default_rng(1), 80)
    ones = np.ones(s.size, bool)
    a = RiskControlledGate(0.9, 0.1, seed=3).fit(s, ones, groups=cid)
    b = RiskControlledGate(0.9, 0.1, seed=3).fit(s, ones, groups=cid)
    assert a.threshold_ == b.threshold_
    assert a.report_.clustered and a.report_.n_units == 80
    assert a.report_.n_positive == s.size
    assert "80 clusters" in a.report_.summary()
    with pytest.raises(ValueError):
        RiskControlledGate().fit(s, ones, groups=cid[:-1])


def test_too_few_clusters_cannot_be_certified_and_keeps_everything():
    s, cid = _clustered(np.random.default_rng(2), 20)   # 20 units < ~58 needed
    gate = RiskControlledGate(0.95, 0.05).fit(s, np.ones(s.size, bool), groups=cid)
    assert gate.report_.feasible is False and gate.threshold_ == 0.0
    assert gate.transform(s).all()


def test_evaluate_reports_cluster_averaged_recall():
    scores = np.array([0.9, 0.9, 0.9, 0.1])     # cluster A: 3 kept, cluster B: 0 kept
    positive = np.ones(4, bool)
    gate = RiskControlledGate(0.5, 0.5)
    gate.threshold_ = 0.5
    res = gate.evaluate(scores, positive, groups=np.array(["A", "A", "A", "B"]))
    assert res["realized_recall"] == 0.75
    assert res["realized_cluster_recall"] == 0.5
    assert res["guarantee_held"] is True    # judged on cluster recall: 0.5 >= 0.5


def test_calibration_export_carries_clusters_and_triage_uses_them(tmp_path, capsys):
    import triage
    from ml.evaluation import _write_gate_calibration

    s, cid = _clustered(np.random.default_rng(4), 120)
    ok = np.random.default_rng(4).random(s.size) < 0.7
    path = tmp_path / "cal.csv"
    _write_gate_calibration(str(path), s, ok.astype(float), cid)
    gate = triage._fit_gate(str(path), 0.9, 0.1)
    assert gate.report_.clustered and gate.report_.n_units <= 120

    plain = tmp_path / "plain.csv"
    _write_gate_calibration(str(plain), s, ok.astype(float))
    gate2 = triage._fit_gate(str(plain), 0.9, 0.1)
    assert not gate2.report_.clustered
    assert "no 'cluster' column" in capsys.readouterr().err
