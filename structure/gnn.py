"""
Graph Neural Network scaffold (Sprint 11).

The prompt introduces GNNs for:
    - active-site prediction
    - binding-site prediction
    - ligand interaction prediction
    - structural compatibility

We do NOT implement a full GNN here. The scaffold provides:

    1) Graph construction from residues + spatial edges.
    2) Pluggable model interface so the user can drop in PyG / DGL.

The graph representation:
    Nodes: residues (one per (chain, residue_number))
    Edges: spatial proximity (< 8 Å heavy-atom distance between residues)

This module does not require torch / dgl / torch_geometric. The
graph itself is just plain Python; only a real GNN consumer needs
PyTorch. The module is importable in the current environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from structure.models import Atom, ProteinStructure


@dataclass
class ResidueNode:
    chain: str
    resname: str
    resnum: int
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.chain, self.resname, self.resnum)


@dataclass
class ResidueGraph:
    nodes: List[ResidueNode] = field(default_factory=list)
    edges: List[Tuple[int, int, float]] = field(default_factory=list)  # (i, j, distance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "node_keys": [n.key for n in self.nodes],
        }


def build_residue_graph(
    structure: ProteinStructure,
    *,
    edge_cutoff: float = 8.0,
    include_features: bool = True,
) -> ResidueGraph:
    """Construct a residue graph from a ProteinStructure.

    Each residue becomes one node; an edge connects two residues if
    any heavy-atom pair is within `edge_cutoff` Å.
    """
    # 1) Aggregate atoms by residue
    residues: Dict[Tuple[str, str, int], List[Atom]] = {}
    for atom in structure.atoms:
        if atom.element == "H":
            continue
        key = (atom.chain_id, atom.residue_name, atom.residue_number)
        residues.setdefault(key, []).append(atom)

    # 2) Build nodes
    nodes: List[ResidueNode] = []
    for (chain, resname, resnum), atoms in residues.items():
        n = len(atoms)
        cx = sum(a.x for a in atoms) / n
        cy = sum(a.y for a in atoms) / n
        cz = sum(a.z for a in atoms) / n
        node = ResidueNode(
            chain=chain, resname=resname, resnum=resnum,
            centroid=(cx, cy, cz),
        )
        if include_features:
            node.features = _residue_features(resname, atoms)
        nodes.append(node)

    # 3) Edges
    edges: List[Tuple[int, int, float]] = []
    for i in range(len(nodes)):
        ci = nodes[i].centroid
        for j in range(i + 1, len(nodes)):
            cj = nodes[j].centroid
            d = math.dist(ci, cj)
            if d <= edge_cutoff:
                edges.append((i, j, d))

    return ResidueGraph(nodes=nodes, edges=edges)


# ----------------------------------------------------------------------
# Residue features (very lightweight)
# ----------------------------------------------------------------------

_RESIDUE_HYDROPHOBICITY = {
    "ALA": 1.8, "VAL": 4.2, "LEU": 3.8, "ILE": 4.5, "MET": 1.9,
    "PHE": 2.8, "TRP": -0.9, "PRO": -1.6, "TYR": -1.3,
    "GLY": -0.4, "CYS": 2.5, "SER": -0.8, "THR": -0.7,
    "ASN": -3.5, "GLN": -3.5, "ASP": -3.5, "GLU": -3.5,
    "HIS": -3.2, "LYS": -3.9, "ARG": -4.5,
}

_AROMATIC = {"PHE", "TYR", "TRP", "HIS"}
_CHARGED_POS = {"LYS", "ARG"}
_CHARGED_NEG = {"ASP", "GLU"}
_POLAR = {"SER", "THR", "ASN", "GLN", "HIS", "CYS", "TYR"}


def _residue_features(resname: str, atoms: List[Atom]) -> Dict[str, float]:
    return {
        "hydrophobicity": _RESIDUE_HYDROPHOBICITY.get(resname, 0.0),
        "is_aromatic": 1.0 if resname in _AROMATIC else 0.0,
        "is_charged_pos": 1.0 if resname in _CHARGED_POS else 0.0,
        "is_charged_neg": 1.0 if resname in _CHARGED_NEG else 0.0,
        "is_polar": 1.0 if resname in _POLAR else 0.0,
        "n_heavy_atoms": float(len(atoms)),
    }


# ----------------------------------------------------------------------
# GNN consumer (stubs)
# ----------------------------------------------------------------------

class GNNModelStub:
    """Placeholder for a real GNN model.

    Drop in a real PyG/DGL model here when ready. The interface
    (predict_active_site, predict_binding_site) is intentionally
    minimal so a real model can replace this stub without touching
    downstream code.
    """

    name = "gnn-stub"

    def is_available(self) -> bool:
        """Real model would check for installed torch / PyG."""
        return False

    def predict_active_site(self, graph: ResidueGraph):
        return []

    def predict_binding_site(self, graph: ResidueGraph, ligand_centroid=None):
        return []