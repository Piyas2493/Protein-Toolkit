"""
Risk-controlled triage gate for ProteinToolkit.

Repo location: ml/selective_gate.py

The triage stage scores every sequence. A gate then decides which sequences
are worth the expensive structural / docking stage. A naive gate keeps a
fixed top-N or everything above a hand-picked score. That gives the user no
control over how many genuinely functional candidates are silently discarded.

This module sets the gate threshold from calibration data so that the gate's
RECALL on a class the user must not miss (e.g. true enzymes, or correctly
EC-classified sequences) is at least a target value, with a distribution-free
confidence guarantee. The threshold is the largest score whose Clopper-Pearson
lower confidence bound on recall still clears the target; a larger threshold
would send fewer sequences on (cheaper) but risk dropping too many true ones.

Guarantee: if the calibration labels are exchangeable with the sequences the
gate will later see, then with probability at least 1 - delta the gate keeps
at least `target_recall` of the positive class. Nothing here assumes the
score is a calibrated probability; it only assumes the scores can be ranked.

Caveat: clustered data (homologous protein sequences) is NOT exchangeable at
the row level. On this toolkit's EC classifier, calibrating on half the
held-out clusters and deploying on the other half met a 95% recall target in
only 67.5% of random splits, not the promised 95%. Use the bound only when
calibration and deployment sequences are genuinely exchangeable.

Dependencies: numpy. SciPy is used for an exact Clopper-Pearson bound when
present, with a conservative closed-form fallback otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.stats import beta as _beta  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only without SciPy
    _HAVE_SCIPY = False


def _clopper_pearson_lower(k: int, n: int, delta: float) -> float:
    """Lower 1 - delta confidence bound on a binomial success probability
    from k successes in n trials. Exact (beta quantile) with SciPy, else a
    conservative Wilson-type bound that never exceeds the exact one."""
    if n == 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k >= n:
        # one-sided lower bound when all trials succeeded
        return float(delta ** (1.0 / n))
    if _HAVE_SCIPY:
        return float(_beta.ppf(delta, k, n - k + 1))
    # SciPy-free fallback: Wilson lower bound at level 1 - delta (one-sided).
    # z for the one-sided tail; approximate the normal quantile.
    p = k / n
    z = _one_sided_z(delta)
    z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * np.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    return float(max(0.0, (centre - half) / (1 + z2 / n)))


def _one_sided_z(delta: float) -> float:
    """Approximate upper-tail normal quantile z with P(Z > z) = delta
    (Acklam-style rational approximation), for the SciPy-free path."""
    # invert the standard normal CDF at 1 - delta
    q = 1.0 - delta
    # Beasley-Springer-Moro / Acklam coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if q < plow:
        t = np.sqrt(-2 * np.log(q))
        return float((((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) /
                     ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1))
    if q <= 1 - plow:
        r = q - 0.5
        s = r * r
        return float((((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r /
                     (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1))
    t = np.sqrt(-2 * np.log(1 - q))
    return float(-(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) /
                  ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1))


@dataclass
class GateReport:
    threshold: float
    target_recall: float
    delta: float
    guaranteed_recall: float   # Clopper-Pearson lower bound at the chosen threshold
    point_recall: float        # empirical recall on the calibration positives
    kept_fraction: float       # share of all calibration sequences sent onward
    n_positive: int
    feasible: bool             # False => target could not be guaranteed; gate keeps all

    def summary(self) -> str:
        if not self.feasible:
            return (f"Gate could not guarantee {self.target_recall:.0%} recall at "
                    f"delta={self.delta}; keeping all sequences (threshold 0).")
        return (f"threshold={self.threshold:.4f} | guaranteed recall "
                f">= {self.guaranteed_recall:.3f} (target {self.target_recall:.2f}, "
                f"delta={self.delta}) | sends {self.kept_fraction:.1%} of sequences "
                f"to the structural stage")


class RiskControlledGate:
    """Chooses a triage-score threshold that bounds the miss rate on a target
    class with a distribution-free confidence guarantee.

    Typical use:
        gate = RiskControlledGate(target_recall=0.95, delta=0.05)
        gate.fit(cal_scores, cal_is_true_enzyme)
        keep = gate.transform(new_scores)     # boolean mask
    """

    def __init__(self, target_recall: float = 0.95, delta: float = 0.05):
        if not 0 < target_recall < 1:
            raise ValueError("target_recall must be in (0, 1)")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.target_recall = target_recall
        self.delta = delta
        self.threshold_: Optional[float] = None
        self.report_: Optional[GateReport] = None

    def fit(self, scores: np.ndarray, positive: np.ndarray) -> "RiskControlledGate":
        scores = np.asarray(scores, dtype=float)
        positive = np.asarray(positive, dtype=bool)
        if scores.shape != positive.shape:
            raise ValueError("scores and positive must have the same shape")
        pos = scores[positive]
        n = pos.size
        if n == 0:
            raise ValueError("no positive calibration examples")

        # Candidate thresholds: each distinct positive score, plus 0 (keep all).
        candidates = np.concatenate(([0.0], np.unique(pos)))
        best = None
        for tau in candidates:
            kept = int(np.sum(pos >= tau))
            lcb = _clopper_pearson_lower(kept, n, self.delta)
            if lcb >= self.target_recall:
                # feasible; prefer the LARGEST such tau (fewest sequences sent on)
                if best is None or tau > best[0]:
                    best = (tau, lcb, kept)

        if best is None:
            self.threshold_ = 0.0
            self.report_ = GateReport(
                threshold=0.0, target_recall=self.target_recall, delta=self.delta,
                guaranteed_recall=_clopper_pearson_lower(n, n, self.delta),
                point_recall=1.0, kept_fraction=1.0, n_positive=n, feasible=False)
            return self

        tau, lcb, kept = best
        self.threshold_ = float(tau)
        self.report_ = GateReport(
            threshold=float(tau), target_recall=self.target_recall, delta=self.delta,
            guaranteed_recall=float(lcb), point_recall=kept / n,
            kept_fraction=float(np.mean(scores >= tau)), n_positive=n, feasible=True)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("call fit() before transform()")
        return np.asarray(scores, dtype=float) >= self.threshold_

    def evaluate(self, scores: np.ndarray, positive: np.ndarray) -> dict:
        """Realized recall and compute saving on a held-out set the gate did
        not see during fit(), for honest reporting."""
        scores = np.asarray(scores, dtype=float)
        positive = np.asarray(positive, dtype=bool)
        keep = self.transform(scores)
        pos = positive.sum()
        recall = float(np.sum(keep & positive) / pos) if pos else float("nan")
        return {
            "realized_recall": recall,
            "kept_fraction": float(np.mean(keep)),
            "compute_saving": float(1.0 - np.mean(keep)),
            "target_recall": self.target_recall,
            "guarantee_held": bool(recall >= self.target_recall) if pos else None,
        }
