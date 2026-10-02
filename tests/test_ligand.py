"""
Tests for the ligand engine (Sprint 4 / Sprint 5).

Covers: SMILES loading, validation, preparation, edge cases from
Section 32 (invalid ligand, charged ligands, tautomer-labile,
missing-bonds PDB input).
"""
import os
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ligand import (
    load_ligand,
    LigandValidator,
    LigandPreparator,
    LigandLoadError,
)
from ligand.validation import ValidationReport
from ligand.ligand import Ligand, _parse_smiles_fallback, _parse_sdf, _parse_mol2


# ----------------------------------------------------------------------
# SMILES loading
# ----------------------------------------------------------------------

def test_load_smiles_aspirin():
    lig = load_ligand(smiles="CC(=O)Oc1ccccc1C(=O)O")
    assert lig.atom_count() == 13
    assert lig.source_format == "smiles"
    assert lig.smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert len(lig.provenance) == 1


def test_load_smiles_branch_and_ring():
    lig = load_ligand(smiles="c1ccncc1")
    assert lig.atom_count() == 6
    # Ring-closure bonds must exist
    assert any(
        {b.a1, b.a2} == {1, 6} for b in lig.bonds
    )


def test_load_smiles_brackets():
    # A disconnected counterion has no bond tying its 3D position to
    # the real molecule once embedded, which silently breaks downstream
    # docking geometry — load_ligand keeps only the largest fragment
    # (standard salt-stripping), so only the cation (4 heavy atoms)
    # survives, not the 1-atom chloride.
    lig = load_ligand(smiles="CC(C)[NH3+].[Cl-]")
    assert lig.atom_count() == 4
    charges = [a.formal_charge for a in lig.atoms]
    assert 1 in charges
    assert -1 not in charges


def test_load_smiles_invalid_raises():
    with pytest.raises(LigandLoadError):
        load_ligand(smiles="C((((((")
    with pytest.raises(LigandLoadError):
        load_ligand(smiles="")
    with pytest.raises(LigandLoadError):
        load_ligand(smiles="Xyz")


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def test_validate_simple_alcohol():
    lig = load_ligand(smiles="CCO")
    rep = LigandValidator().validate(lig)
    assert rep.chemical_valid is True
    assert rep.heavy_atoms == 3
    assert rep.molecular_weight > 40
    assert rep.molecular_weight < 50


def test_validate_substituted_fused_heteroaromatic():
    # Caffeine: N-methylated fused imidazole/pyrimidinedione rings.
    # The flat "1.5 per aromatic bond" valence sum falsely flags its
    # ring nitrogens as over-valent; RDKit already validated this
    # structure during parsing, so chemistry_verified must skip the
    # redundant (and here, wrong) heuristic re-check.
    lig = load_ligand(smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    assert lig.chemistry_verified is True
    rep = LigandValidator().validate(lig)
    assert rep.chemical_valid is True
    assert rep.errors == []


def test_validate_charged_ligand_warning():
    # After salt-stripping, only the cation remains — its charge is the
    # ligand's real net charge, not "neutralized" by the dropped
    # chloride counterion.
    lig = load_ligand(smiles="CC(C)[NH3+].[Cl-]")
    rep = LigandValidator().validate(lig)
    assert rep.net_charge == 1
    assert any(a.formal_charge != 0 for a in lig.atoms)


def test_validate_empty_raises_error():
    empty = Ligand(name="empty")
    rep = LigandValidator().validate(empty)
    assert rep.valid is False
    assert rep.chemical_valid is False


def test_validate_tautomer_flag():
    lig = load_ligand(smiles="NC(=O)O")  # carbamic acid — tautomer-labile
    # has_tautomer_labile depends on h_count on N, which RDKit already
    # computes correctly at parse time (unlike the no-RDKit fallback,
    # which defers implicit-H accounting to preparation). Either way,
    # after preparation the flag must correctly reflect the labile N-H.
    prep = LigandPreparator().prepare(lig)
    rep = LigandValidator().validate(prep.ligand)
    assert rep.has_tautomer_labile is True


def test_validation_report_to_dict_roundtrip():
    lig = load_ligand(smiles="CCO")
    rep = LigandValidator().validate(lig)
    d = rep.to_dict()
    assert d["valid"] is True
    assert "molecular_weight" in d
    assert "errors" in d
    assert "warnings" in d


# ----------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------**

def test_prepare_adds_hydrogens():
    # Ethanol's -OH is a donor. With RDKit, implicit H count (and thus
    # hbd) is already correct right after parsing; the no-RDKit fallback
    # only fills it in during preparation. Either way, after preparation
    # hbd must be correct.
    lig = load_ligand(smiles="CCO")
    prep = LigandPreparator(add_hydrogens=True).prepare(lig)
    post_rep = LigandValidator().validate(prep.ligand)
    assert post_rep.hbd >= 1
    assert any(s.name == "hydrogens" and s.applied for s in prep.steps)


def test_prepare_records_provenance():
    lig = load_ligand(smiles="CCO")
    prep = LigandPreparator().prepare(lig)
    step_names = [s.name for s in prep.steps]
    for required in ("standardize", "protonation", "tautomer",
                     "hydrogens", "3d_conformer", "minimize"):
        assert required in step_names


def test_prepare_3d_coordinates():
    lig = load_ligand(smiles="CCO")
    prep = LigandPreparator().prepare(lig)
    assert prep.has_3d is True
    # At least some atoms should have nonzero coords
    nonzero = sum(
        1 for a in prep.ligand.atoms
        if (a.x, a.y, a.z) != (0.0, 0.0, 0.0)
    )
    assert nonzero > 0


def test_prepare_protonation_changes_charge():
    lig = load_ligand(smiles="CC(=O)O")  # acetic acid
    before = sum(a.formal_charge for a in lig.atoms)
    prep = LigandPreparator(protonation="deprotonate-acid").prepare(lig)
    after = sum(a.formal_charge for a in prep.ligand.atoms)
    assert after < before  # one -1 charge introduced


def test_prepare_invalid_mode_raises():
    lig = load_ligand(smiles="CCO")
    with pytest.raises(ValueError):
        LigandPreparator(protonation="magic").prepare(lig)


def test_prepare_reproducible_seed():
    lig = load_ligand(smiles="CCO")
    p1 = LigandPreparator(seed=42).prepare(lig)
    p2 = LigandPreparator(seed=42).prepare(lig)
    coords1 = [(a.x, a.y, a.z) for a in p1.ligand.atoms]
    coords2 = [(a.x, a.y, a.z) for a in p2.ligand.atoms]
    # Without RDKit fallback tree-builder is deterministic
    assert coords1 == coords2


# ----------------------------------------------------------------------
# SDF / MOL2 / PDB I/O
# ----------------------------------------------------------------------

def test_parse_sdf_minimal(tmp_path):
    """A minimal V2000 SDF block — checked-in as inline string."""
    sdf = """\
Test
  Mrv2014

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
M  END
"""
    p = tmp_path / "test.sdf"
    p.write_text(sdf, encoding="utf-8")
    lig = load_ligand(source=str(p))
    assert lig.atom_count() == 3
    assert lig.bond_count() == 2
    assert lig.source_format == "sdf"
    # Validation must pass — valence sums correct
    rep = LigandValidator().validate(lig)
    assert rep.chemical_valid is True


def test_parse_mol2_minimal(tmp_path):
    mol2 = """\
@<TRIPOS>MOLECULE
test
SMALL
USER_CHARGES


@<TRIPOS>ATOM
1 C 0.0 0.0 0.0 C.alkane 1.0
2 C 1.5 0.0 0.0 C.alkane 1.0
@<TRIPOS>BOND
1 1 2 1
"""
    p = tmp_path / "test.mol2"
    p.write_text(mol2, encoding="utf-8")
    lig = load_ligand(source=str(p))
    assert lig.atom_count() == 2
    assert lig.bond_count() == 1


def _hetatm_line(serial: int, atom_name: str, resname: str, chain: str,
                 resnum: int, x: float, y: float, z: float, element: str) -> str:
    """Build a properly aligned HETATM record (80 cols)."""
    return (
        f"HETATM{serial:>5d} {atom_name:<4s} {resname:<3s} "
        f"{chain:1s}{resnum:>4d}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}          "
        f"{element:>2s}"
    )


def test_parse_ligand_pdb_no_bonds(tmp_path):
    pdb = "\n".join([
        _hetatm_line(1, " C1 ", "UNL", "A", 1, 0.0, 0.0, 0.0, "C"),
        _hetatm_line(2, " C2 ", "UNL", "A", 1, 1.5, 0.0, 0.0, "C"),
        _hetatm_line(3, " O3 ", "UNL", "A", 1, 3.0, 0.0, 0.0, "O"),
        "",
    ])
    p = tmp_path / "test.pdb"
    p.write_text(pdb, encoding="utf-8")
    lig = load_ligand(source=str(p))
    assert lig.atom_count() == 3
    assert lig.bond_count() == 0  # HETATM-only input
    rep = LigandValidator().validate(lig)
    assert rep.chemical_valid is True


def test_parse_malformed_pdb_warns(tmp_path):
    # Malformed: missing coordinate columns.
    bad = "HETATM    1  C1   UNL A   1\n"
    p = tmp_path / "bad.pdb"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(LigandLoadError):
        load_ligand(source=str(p))


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "test.xyz"
    p.write_text("garbage", encoding="utf-8")
    with pytest.raises(LigandLoadError):
        load_ligand(source=str(p))


def test_load_ligand_no_input_raises():
    with pytest.raises(LigandLoadError):
        load_ligand()


def test_load_ligand_missing_file_raises(tmp_path):
    with pytest.raises(LigandLoadError):
        load_ligand(source=str(tmp_path / "does_not_exist.sdf"))


# ----------------------------------------------------------------------
# Ligand center / bounding box
# ----------------------------------------------------------------------

def test_center_and_bbox():
    lig = load_ligand(smiles="CCO")
    prep = LigandPreparator().prepare(lig)
    cx, cy, cz = prep.ligand.center()
    bbox = prep.ligand.bounding_box()
    assert bbox[0] <= cx <= bbox[3]
    assert bbox[1] <= cy <= bbox[4]
    assert bbox[2] <= cz <= bbox[5]
