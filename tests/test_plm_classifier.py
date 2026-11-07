import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.plm_classifier import PLMEnzymeWrapper, find_weights
from ml.predictor import EnzymePredictor


def _wrapper(temperature=1.0):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 1280)).astype(np.float32)
    y = rng.integers(0, 3, 300)
    X[np.arange(300), y] += 4.0                       # make the classes separable
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)
    return PLMEnzymeWrapper(scaler, clf, temperature, ["EC1", "EC2", "EC3"]), X, y


def test_probabilities_sum_to_one_and_temperature_softens_them():
    w, X, y = _wrapper(1.0)
    p = w.predict_proba_embeddings(X[:20])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == y[:20]).mean() > 0.9

    hot, _, _ = _wrapper(3.0)
    hot.classifier, hot.scaler = w.classifier, w.scaler
    soft = hot.predict_proba_embeddings(X[:20])
    assert np.allclose(soft.sum(axis=1), 1.0)
    assert soft.max(axis=1).mean() < p.max(axis=1).mean()   # T > 1 lowers confidence
    assert (soft.argmax(axis=1) == p.argmax(axis=1)).all()   # but never changes the call


def test_pickle_drops_the_loaded_language_model():
    w, _, _ = _wrapper()
    w._esm = object()                                  # pretend a model is loaded
    again = pickle.loads(pickle.dumps(w))
    assert again._esm is None and list(again.classes_) == ["EC1", "EC2", "EC3"]


def test_predictor_routes_embedding_models_through_the_sequence(tmp_path, monkeypatch):
    w, X, _ = _wrapper()
    monkeypatch.setattr(w, "embed_sequence", lambda seq: X[:1])   # skip ESM itself
    path = tmp_path / "plm.pkl"
    path.write_bytes(b"x")                                         # only has to exist
    monkeypatch.setattr("ml.predictor.ModelLoader.load", staticmethod(lambda p: w))
    out = EnzymePredictor(str(path)).predict_sequence("MKTAYIAKQR")
    assert out["prediction"] in ("EC1", "EC2", "EC3")
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6


def test_missing_weights_give_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ESM1B_WEIGHTS", raising=False)
    monkeypatch.setattr("ml.plm_classifier.Path.home", staticmethod(lambda: tmp_path))
    with pytest.raises(FileNotFoundError, match="ESM1B_WEIGHTS"):
        find_weights(str(tmp_path / "nowhere.pt"))
