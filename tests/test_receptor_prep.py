"""
Receptor preparation for docking: protonation at pH 7.4, charges and
AutoDock atom types (docking.preparation.write_prepared_receptor_pdbqt),
and the fallback to the old unprotonated writer.

The receptor is a synthetic octapeptide (ACDEKHSY: carboxylates, a Lys, a
His, hydroxyls) built with RDKit, plus a distant Zn ion — small enough to
prepare in ~1 s, with residues whose protonation at pH 7.4 is unambiguous.
"""
import math
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem

import docking.preparation as preparation
from docking import ReceptorPreparation

ZN_LINE = ("HETATM  999 ZN    ZN A 100      40.000  40.000  40.000"
           "  1.00  0.00          ZN")


@pytest.fixture(scope="module")
def peptide_pdb(tmp_path_factory):
    mol = Chem.AddHs(Chem.MolFromSequence("ACDEKHSY"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    block = Chem.MolToPDBBlock(Chem.RemoveHs(mol), flavor=4)
    atoms = [l for l in block.splitlines() if l.startswith("ATOM")]
    path = tmp_path_factory.mktemp("recep") / "peptide.pdb"
    path.write_text("\n".join(atoms + [ZN_LINE, "END"]) + "\n")
    return str(path)


def _read_pdbqt(path):
    return [{"name": ln[12:16].strip(), "res": int(ln[22:26]),
             "xyz": (float(ln[30:38]), float(ln[38:46]), float(ln[46:54])),
             "q": float(ln[70:76]), "type": ln[77:79].strip()}
            for ln in Path(path).read_text().splitlines()
            if ln.startswith("ATOM")]


def _n_hd(atoms, name, res):
    heavy = next(a for a in atoms if a["name"] == name and a["res"] == res)
    return sum(1 for a in atoms if a["type"] == "HD"
               and math.dist(a["xyz"], heavy["xyz"]) < 1.3)


def test_prepared_receptor_is_protonated_charged_and_typed(peptide_pdb):
    pytest.importorskip("pdb2pqr")
    pytest.importorskip("propka")
    pytest.importorskip("meeko")

    prep = ReceptorPreparation().prepare(peptide_pdb)
    assert len(prep.hetero_atoms) == 1 and prep.hetero_atoms[0].residue_name == "ZN"
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # the fallback warns; this must not
        atoms = _read_pdbqt(prep.pdbqt())
    info = prep.pdbqt_info

    assert info["route"] == "pdb2pqr+meeko" and info["ph"] == 7.4
    assert info["n_hd"] >= 15 and info["residues_left_out"] == []
    assert sum(a["type"] == "HD" for a in atoms) == info["n_hd"]

    # pH 7.4 states: Lys NH3+, Asp carboxylate (no H), Ser/Tyr hydroxyl H,
    # amide NH on an interior backbone N.
    assert _n_hd(atoms, "NZ", 5) == 3
    assert _n_hd(atoms, "OD1", 3) == 0 and _n_hd(atoms, "OD2", 3) == 0
    assert _n_hd(atoms, "OG", 7) == 1
    assert _n_hd(atoms, "OH", 8) == 1
    assert _n_hd(atoms, "N", 4) == 1
    # N-term NH3+ + Lys+ vs C-term, Asp, Glu: net -1.
    assert round(info["net_charge"]) == -1
    assert any(abs(a["q"]) > 0.1 for a in atoms)

    by = {(a["res"], a["name"]): a["type"] for a in atoms}
    assert by[(8, "CZ")] == "A" and by[(6, "CG")] == "A"     # aromatic C
    assert by[(3, "OD1")] == "OA"

    zn = atoms[-1]                                  # metals are appended last
    assert (zn["type"], zn["q"]) == ("Zn", 2.0)
    assert info["hd_near_metal"] == 0


def test_residue_without_backbone_is_left_out_not_fatal(peptide_pdb):
    pytest.importorskip("pdb2pqr")
    pytest.importorskip("propka")
    pytest.importorskip("meeko")

    lines = Path(peptide_pdb).read_text().splitlines()
    # a trailing residue resolved only to its N (as in PDB 3ERT, PRO 552)
    lines.insert(-2, "ATOM    900  N   PRO A   9      20.000  20.000  20.000"
                     "  1.00  0.00           N")
    path = Path(peptide_pdb).with_name("with_stub.pdb")
    path.write_text("\n".join(lines) + "\n")

    prep = ReceptorPreparation().prepare(str(path))
    prep.pdbqt()
    assert prep.pdbqt_info["route"] == "pdb2pqr+meeko"
    assert prep.pdbqt_info["residues_left_out"] == ["A:9 PRO"]


def test_falls_back_to_unprotonated_zero_charge_writer(peptide_pdb, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ImportError("No module named 'pdb2pqr'")
    monkeypatch.setattr(preparation, "_protonate", unavailable)

    prep = ReceptorPreparation().prepare(peptide_pdb)
    with pytest.warns(RuntimeWarning, match="unprotonated zero-charge"):
        atoms = _read_pdbqt(prep.pdbqt())
    assert prep.pdbqt_info["route"] == "legacy"
    assert not any(a["type"] == "HD" for a in atoms)
    assert all(a["q"] == 0.0 for a in atoms)
    assert len(atoms) == len(prep.receptor_atoms)


def test_pdbqt_is_built_once_per_receptor(peptide_pdb, monkeypatch):
    calls = []

    def fake(prep, out_path, *, ph):
        calls.append(ph)
        Path(out_path).write_text("END\n")
        return {"route": "fake"}
    monkeypatch.setattr(preparation, "write_prepared_receptor_pdbqt", fake)

    prep = ReceptorPreparation().prepare(peptide_pdb)
    assert prep.pdbqt() == prep.pdbqt()
    assert calls == [preparation.DEFAULT_PH]
