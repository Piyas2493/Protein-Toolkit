"""
Sprint 10: Protein Language Model (PLM) scaffold.

The prompt specifies ESM-2 and ProtT5 with frozen embeddings.

We provide:
    1) A `PLMEmbedder` interface that exposes `embed(sequence)`.
    2) A hashing fallback that produces a deterministic 64-d vector
       from the sequence — useful for testing without the heavy
       `transformers` dependency.
    3) Lazy imports of `transformers` so the module is importable in
       the current environment. A real user wires in a real model:

            embedder = ESM2Embedder()
            emb = embedder.embed(seq)   # np.ndarray of shape (L, 480)

The downstream classifier (RF / XGBoost / shallow MLP) is the
responsibility of the calling code; this module only produces the
embedding.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np


class PLMEmbedder:
    """Interface for protein-language-model embedders."""

    name: str = "base"
    dim: int = 64  # default fallback

    def is_available(self) -> bool:
        raise NotImplementedError

    def embed(self, sequence: str) -> np.ndarray:
        raise NotImplementedError


class HashingFallbackEmbedder(PLMEmbedder):
    """Deterministic 64-d embedding from sequence composition.

    NOT a real protein language model. Intended for unit tests and
    for users without GPU / transformers installed.
    """

    name = "hashing-fallback"
    dim = 64

    def is_available(self) -> bool:
        return True

    def embed(self, sequence: str) -> np.ndarray:
        # Per-residue one-hot (20 aa) projected into a fixed 64-d
        # vector via a seeded hash. Deterministic and reproducible.
        rng = np.random.default_rng(42)
        projection = rng.standard_normal((20, self.dim)).astype(np.float32)
        aa_index = {
            a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")
        }
        out = np.zeros(self.dim, dtype=np.float32)
        for ch in sequence.upper():
            idx = aa_index.get(ch)
            if idx is None:
                continue
            out += projection[idx]
        if len(sequence) > 0:
            out /= len(sequence)
        # Sequence-level fingerprint (sum of k-mers, hashed)
        h = hashlib.sha256(sequence.encode("utf-8")).digest()
        for i in range(self.dim):
            out[i] += (h[i % len(h)] - 128) / 256.0
        return out


class ESM2Embedder(PLMEmbedder):
    """ESM-2 wrapper (Sprint 10). Imports `transformers` lazily."""

    name = "esm2"
    dim = 480   # ESM-2 t6 / t12 hidden size; real size depends on model

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def embed(self, sequence: str) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                "transformers is not installed; install it or use "
                "HashingFallbackEmbedder for testing."
            )
        # Lazy import keeps the module importable in slim environments.
        from transformers import AutoTokenizer, AutoModel  # type: ignore
        import torch  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors="pt").to(self.device)
            outputs = model(**inputs)
        # Mean-pool over the sequence dimension
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return emb.astype(np.float32)


class ProtT5Embedder(PLMEmbedder):
    """ProtT5 wrapper (Sprint 10)."""

    name = "prot_t5"
    dim = 1024  # ProtT5 base hidden size

    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_uniref50",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def embed(self, sequence: str) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError(
                "transformers is not installed; install it or use "
                "HashingFallbackEmbedder for testing."
            )
        from transformers import T5Tokenizer, T5EncoderModel  # type: ignore
        import torch  # type: ignore

        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
        model.eval()
        with torch.no_grad():
            seq = " ".join(list(sequence))
            ids = tokenizer.batch_encode_plus(
                [seq], add_special_tokens=True, return_tensors="pt"
            )
            ids = {k: v.to(self.device) for k, v in ids.items()}
            emb = model(**ids)
        emb = emb.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return emb.astype(np.float32)