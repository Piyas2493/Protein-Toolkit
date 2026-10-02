"""
ProteinToolkit Enzyme Predictor

Two feature tiers are supported:
    - Base (30 columns) — RandomForest at ml/models/enzyme_classifier.pkl
    - Extended (430 columns) — XGBoost at ml/models/enzyme_classifier_xgb.pkl

The predictor auto-detects whether the loaded model needs dipeptides
by checking `feature_names_in_`. If any feature starts with "DIPEP_",
the extended schema is requested from FeatureExtractor.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .feature_extractor import FeatureExtractor
from .model_loader import ModelLoader


class XGBoostEnzymeWrapper:
    """sklearn-compatible wrapper around an XGBoostClassifier + LabelEncoder.

    Exposes:
        - .predict(X)            -> np.ndarray of string labels
        - .predict_proba(X)      -> np.ndarray (n, n_classes) of probabilities
        - .classes_              -> np.ndarray of string class labels
        - .feature_names_in_     -> np.ndarray of feature names

    Internally encodes string labels via LabelEncoder so XGBoost sees
    integer targets, then decodes back at inference.
    """

    def __init__(self, model, label_encoder, feature_names_in_):
        self.model = model
        self.label_encoder = label_encoder
        self.feature_names_in_ = feature_names_in_
        self.classes_ = np.asarray(label_encoder.classes_, dtype=object)

    def predict(self, X):
        enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(enc.astype(int))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class EnzymePredictor:

    def __init__(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.model = ModelLoader.load(model_path)

        self.feature_extractor = FeatureExtractor()
        self.include_dipeptides = self._model_needs_dipeptides()

    def _model_needs_dipeptides(self) -> bool:
        names = getattr(self.model, "feature_names_in_", None)
        if names is None:
            return False
        return any(str(n).startswith("DIPEP_") for n in names)

    def predict_sequence(self, sequence):

        features = self.feature_extractor.extract(
            sequence, include_dipeptides=self.include_dipeptides,
        )

        X = pd.DataFrame([features])

        if hasattr(self.model, "feature_names_in_"):
            X = X.reindex(
                columns=self.model.feature_names_in_
            )

        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]

        probability_dict = {
            class_name: float(probability)
            for class_name, probability in zip(
                self.model.classes_,
                probabilities
            )
        }

        return {
            "prediction": prediction,
            "probabilities": probability_dict,
            "features": features,
        }