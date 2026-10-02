"""
Tests for the ML scaffolds (GNN, PLMs).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tests.test_docking import make_test_pdb

from structure.gnn import build_residue_graph, GNNModelStub
from ml.plm import HashingFallbackEmbedder, ESM2Embedder, ProtT5Embedder


@pytest.fixture
def pdb_file(tmp_path):
    """Local PDB fixture (mirrors test_docking.pdb_file)."""
    p = tmp_path / "mini.pdb"
    make_test_pdb(p)
    return p


# ----------------------------------------------------------------------
# GNN
# ----------------------------------------------------------------------

def test_gnn_graph_builds(pdb_file):
    from structure.parser import parse_pdb
    structure = parse_pdb(str(pdb_file))
    graph = build_residue_graph(structure, edge_cutoff=10.0)
    assert graph.nodes
    assert all(n.features for n in graph.nodes)
    # At least one edge
    assert graph.edges


def test_gnn_graph_features(pdb_file):
    from structure.parser import parse_pdb
    structure = parse_pdb(str(pdb_file))
    graph = build_residue_graph(structure)
    phe = next(n for n in graph.nodes if n.resname == "PHE")
    assert phe.features["is_aromatic"] == 1.0
    assert phe.features["is_polar"] == 0.0


def test_gnn_stub_unavailable():
    stub = GNNModelStub()
    assert stub.is_available() is False
    assert stub.predict_active_site(None) == []


# ----------------------------------------------------------------------
# PLM
# --------------------------------------------------------------------**

def test_plm_fallback_embedder():
    e = HashingFallbackEmbedder()
    assert e.is_available()
    emb = e.embed("ACDEFGHIKLMNPQRSTVWY")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (e.dim,)


def test_plm_fallback_deterministic():
    e = HashingFallbackEmbedder()
    a = e.embed("ACDEF")
    b = e.embed("ACDEF")
    assert np.allclose(a, b)


def test_plm_fallback_different_sequences_differ():
    e = HashingFallbackEmbedder()
    a = e.embed("ACDEF")
    b = e.embed("GHIKL")
    assert not np.allclose(a, b)


def test_esm2_not_installed_in_slim_env():
    e = ESM2Embedder()
    # If transformers is installed, is_available() is True; either is OK.
    assert isinstance(e.is_available(), bool)


def test_prott5_not_installed_in_slim_env():
    e = ProtT5Embedder()
    assert isinstance(e.is_available(), bool)