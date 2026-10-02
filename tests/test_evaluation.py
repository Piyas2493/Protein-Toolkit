import numpy as np

from ml.evaluation import (
    calibration_bins, expected_calibration_error, brier_score,
)


def test_perfectly_calibrated_has_zero_ece():
    # 10 predictions at 0.9 confidence, 9 correct -> matches its own bin.
    confidences = np.array([0.9] * 10)
    correct = np.array([1.0] * 9 + [0.0])
    bins = calibration_bins(confidences, correct, n_bins=10)
    ece = expected_calibration_error(bins, len(correct))
    assert ece < 1e-6


def test_overconfident_model_has_positive_ece():
    # Always 0.99 confident, but only right half the time.
    confidences = np.array([0.99] * 10)
    correct = np.array([1.0] * 5 + [0.0] * 5)
    bins = calibration_bins(confidences, correct, n_bins=10)
    ece = expected_calibration_error(bins, len(correct))
    assert ece > 0.4


def test_brier_score_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_true_idx = np.array([0, 1])
    assert brier_score(probs, y_true_idx, n_classes=3) == 0.0


def test_brier_score_uniform_guess_is_worse():
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    y_true_idx = np.array([0])
    score = brier_score(probs, y_true_idx, n_classes=3)
    assert score > 0.0
