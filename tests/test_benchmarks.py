import random

from benchmarks.screening import auc_with_ci, size_matched_negatives


def test_size_matched_negatives_match_sizes_and_never_reuse():
    pool = {"a": 10, "b": 20, "c": 21, "d": 30}
    chosen = size_matched_negatives([21, 21, 10], pool, random.Random(0))
    assert chosen[0] == "c" and chosen[2] == "a"       # nearest size
    assert chosen[1] == "b"                            # "c" already used
    assert len(set(chosen)) == len(chosen)


def test_auc_with_ci_perfect_random_and_inverted():
    y = [1] * 20 + [0] * 20
    auc, lo, hi = auc_with_ci(y, list(range(40, 0, -1)))
    assert (auc, lo, hi) == (1.0, 1.0, 1.0)
    assert auc_with_ci(y, list(range(40)))[0] == 0.0
    mixed = [0, 1] * 20                                # uninformative score
    auc, lo, hi = auc_with_ci(y, mixed)
    assert lo <= auc <= hi and 0.3 < auc < 0.7
