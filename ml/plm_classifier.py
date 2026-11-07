"""
EC classifier on frozen ESM-1b embeddings.

`PLMEnzymeWrapper` bundles a scaler, a classifier fitted on 1280-d mean ESM-1b
embeddings, and a temperature that calibrates its probabilities. Unlike the
feature-based models it needs the sequence itself (to embed it), so it exposes
`predict_proba_sequence`; `ml.predictor.EnzymePredictor` uses that when present.

Optional dependencies: `torch` and `fair-esm`, plus the ESM-1b weights file
(esm1b_t33_650M_UR50S.pt, ~7.8 GB, from https://dl.fbaipublicfiles.com/fair-esm/models/).
Weights are looked up from, in order: the ESM1B_WEIGHTS environment variable,
the torch hub cache (~/.cache/torch/hub/checkpoints/), then the path stored at
training time. Embedding runs on a GPU if one is available, else on the CPU
(slow: several seconds per sequence).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

MAXLEN = 1022
WEIGHTS_NAME = "esm1b_t33_650M_UR50S.pt"


def find_weights(stored: Optional[str] = None) -> Path:
    candidates = [os.environ.get("ESM1B_WEIGHTS"),
                  str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / WEIGHTS_NAME),
                  stored]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        f"ESM-1b weights ({WEIGHTS_NAME}) not found. Set ESM1B_WEIGHTS or place the file in "
        "~/.cache/torch/hub/checkpoints/ (download: "
        "https://dl.fbaipublicfiles.com/fair-esm/models/" + WEIGHTS_NAME + ").")


class PLMEnzymeWrapper:
    """sklearn-style wrapper; see module docstring."""

    needs_sequence_embedding = True

    def __init__(self, scaler, classifier, temperature: float, classes: List[str],
                 weights_path: Optional[str] = None, source: str = ""):
        self.scaler = scaler
        self.classifier = classifier
        self.temperature = float(temperature)
        self.classes_ = np.asarray(classes, dtype=object)
        self.weights_path = weights_path
        self.source = source
        self._esm = None                      # lazily loaded, never pickled

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_esm"] = None
        return state

    # ---- probabilities from embeddings -----------------------------------
    def predict_proba_embeddings(self, emb: np.ndarray) -> np.ndarray:
        """Temperature-calibrated class probabilities for (n, 1280) embeddings."""
        p = self.classifier.predict_proba(self.scaler.transform(np.asarray(emb, dtype=np.float32)))
        z = np.log(np.clip(p, 1e-12, 1.0)) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---- from a sequence -------------------------------------------------
    def embed_sequence(self, sequence: str) -> np.ndarray:
        from ml.training.embed_esm1b_dataset import embed_batch, load_model
        if self._esm is None:
            self._esm = load_model(find_weights(self.weights_path))
        model, alphabet, device = self._esm
        return embed_batch(model, alphabet, device, [("query", sequence.upper()[:MAXLEN])])

    def predict_proba_sequence(self, sequence: str) -> np.ndarray:
        return self.predict_proba_embeddings(self.embed_sequence(sequence))[0]
