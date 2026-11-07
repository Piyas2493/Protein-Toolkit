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

Guarantee: if the calibration units are exchangeable with the units the gate
will later see, then with probability at least 1 - delta the gate keeps at
least `target_recall` of the positive class. Nothing here assumes the score is
a calibrated probability; it only assumes the scores can be ranked.

What counts as a "unit" depends on `groups`:

* No `groups`: every positive row is a unit. Fine for independent rows, but
  homologous protein sequences are not independent, and the bound is then
  over-confident. On this toolkit's EC classifier, calibrating on half the
  held-out clusters and deploying on the other half met a 95% row-weighted
  recall target in only 67% of random splits (1.5% for cluster-averaged
  recall); see benchmarks/gate_coverage.py.
* With `groups` (one cluster label per row): ONE positive is drawn uniformly
  at random from each cluster and only those draws are used. They are
  independent draws, so the same exact bound applies with the cluster as the
  unit and no asymptotic approximation is needed. The price is a change of
  estimand: the guarantee is on CLUSTER-AVERAGED recall (every homologous
  family counts equally, so large families cannot dominate) and information
  from the other cluster members is discarded. Row-weighted recall on a new
  batch can be higher or lower than that.

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


def _largest_certified_threshold(units: np.ndarray, target: float, delta: float):
    """(tau, lower_bound, n_kept) for the largest tau whose Clopper-Pearson
    lower bound on recall clears `target`, or None if none (not even tau = 0)
    does. `units` are the scores of the independent positive draws."""
    n = units.size
    srt = np.sort(units)
    # Candidate thresholds: each distinct positive score, plus 0 (keep all).
    cands = np.concatenate(([0.0], np.unique(srt)))
    kept = n - np.searchsorted(srt, cands, side="left")       # units >= tau
    # Raising tau only lowers `kept`, and the bound only falls with it, so the
    # feasible thresholds form a prefix: bisect for its end (the LARGEST tau,
    # i.e. the fewest sequences sent on).
    if _clopper_pearson_lower(int(kept[0]), n, delta) < target:
        return None
    lo, hi = 0, cands.size - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _clopper_pearson_lower(int(kept[mid]), n, delta) >= target:
            lo = mid
        else:
            hi = mid - 1
    return (float(cands[lo]), _clopper_pearson_lower(int(kept[lo]), n, delta),
            int(kept[lo]))


@dataclass
class GateReport:
    threshold: float
    target_recall: float
    delta: float
    guaranteed_recall: float   # Clopper-Pearson lower bound at the chosen threshold
    point_recall: float        # empirical recall over the calibration units
    kept_fraction: float       # share of all calibration sequences sent onward
    n_positive: int            # positive calibration rows
    feasible: bool             # False => target could not be guaranteed; gate keeps all
    n_units: int = 0           # independent units behind the bound (rows or clusters)
    clustered: bool = False    # True => one random positive per cluster was used

    def summary(self) -> str:
        unit = (f"{self.n_units} clusters, cluster-averaged recall"
                if self.clustered else f"{self.n_units} rows assumed independent")
        if not self.feasible:
            return (f"Gate could not guarantee {self.target_recall:.0%} recall at "
                    f"delta={self.delta} from {unit}; keeping all sequences "
                    "(threshold 0).")
        return (f"threshold={self.threshold:.4f} | guaranteed recall "
                f">= {self.guaranteed_recall:.3f} (target {self.target_recall:.2f}, "
                f"delta={self.delta}) | sends {self.kept_fraction:.1%} of sequences "
                f"to the structural stage | {unit}")


class RiskControlledGate:
    """Chooses a triage-score threshold that bounds the miss rate on a target
    class with a distribution-free confidence guarantee.

    Typical use:
        gate = RiskControlledGate(target_recall=0.95, delta=0.05)
        gate.fit(cal_scores, cal_is_true_enzyme)                  # independent rows
        gate.fit(cal_scores, cal_is_true_enzyme, groups=cluster)  # clustered rows
        keep = gate.transform(new_scores)     # boolean mask

    Pass `groups` (one cluster label per row) whenever rows are not
    independent, e.g. homologous protein sequences; see the module docstring
    for what that changes. `seed` fixes which positive is drawn per cluster, so
    a fit is reproducible.
    """

    def __init__(self, target_recall: float = 0.95, delta: float = 0.05,
                 seed: int = 0):
        if not 0 < target_recall < 1:
            raise ValueError("target_recall must be in (0, 1)")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.target_recall = target_recall
        self.delta = delta
        self.seed = seed
        self.threshold_: Optional[float] = None
        self.report_: Optional[GateReport] = None

    def fit(self, scores: np.ndarray, positive: np.ndarray,
            groups: Optional[np.ndarray] = None) -> "RiskControlledGate":
        scores = np.asarray(scores, dtype=float)
        positive = np.asarray(positive, dtype=bool)
        if scores.shape != positive.shape:
            raise ValueError("scores and positive must have the same shape")
        pos = scores[positive]
        n = pos.size
        if n == 0:
            raise ValueError("no positive calibration examples")

        units = pos
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != scores.shape:
                raise ValueError("groups must have the same shape as scores")
            # One uniformly random positive per cluster: shuffle, then keep the
            # first row of each cluster. The draws are independent across
            # clusters, so the bound below is exact with the cluster as the unit.
            shuffled = np.random.default_rng(self.seed).permutation(
                np.flatnonzero(positive))
            _, first = np.unique(groups[shuffled], return_index=True)
            units = scores[shuffled[first]]

        best = _largest_certified_threshold(units, self.target_recall, self.delta)
        common = dict(target_recall=self.target_recall, delta=self.delta,
                      n_positive=n, n_units=int(units.size),
                      clustered=groups is not None)
        if best is None:
            self.threshold_ = 0.0
            self.report_ = GateReport(
                threshold=0.0,
                guaranteed_recall=_clopper_pearson_lower(units.size, units.size,
                                                         self.delta),
                point_recall=1.0, kept_fraction=1.0, feasible=False, **common)
            return self

        tau, lcb, kept = best
        self.threshold_ = float(tau)
        self.report_ = GateReport(
            threshold=float(tau), guaranteed_recall=float(lcb),
            point_recall=kept / units.size,
            kept_fraction=float(np.mean(scores >= tau)), feasible=True, **common)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("call fit() before transform()")
        return np.asarray(scores, dtype=float) >= self.threshold_

    def evaluate(self, scores: np.ndarray, positive: np.ndarray,
                 groups: Optional[np.ndarray] = None) -> dict:
        """Realized recall and compute saving on a held-out set the gate did
        not see during fit(), for honest reporting. With `groups` it also
        reports cluster-averaged recall, the quantity a clustered fit bounds."""
        scores = np.asarray(scores, dtype=float)
        positive = np.asarray(positive, dtype=bool)
        keep = self.transform(scores)
        pos = positive.sum()
        recall = float(np.sum(keep & positive) / pos) if pos else float("nan")
        out = {
            "realized_recall": recall,
            "kept_fraction": float(np.mean(keep)),
            "compute_saving": float(1.0 - np.mean(keep)),
            "target_recall": self.target_recall,
        }
        if groups is not None and pos:
            g = np.asarray(groups)[positive]
            _, inv = np.unique(g, return_inverse=True)
            per_cluster = (np.bincount(inv, weights=keep[positive].astype(float))
                           / np.bincount(inv))
            out["realized_cluster_recall"] = float(per_cluster.mean())
            target_stat = out["realized_cluster_recall"]
        else:
            target_stat = recall
        out["guarantee_held"] = bool(target_stat >= self.target_recall) if pos else None
        return out
