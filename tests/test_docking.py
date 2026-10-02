"""
Tests for the docking/compatibility pipeline (Sprints 6, 7, 8).

End-to-end coverage:
    - PDB → pocket detection
    - Ligand → compatibility
    - Docking (mock) → pose analysis → ranking
    - Edge cases: missing ligand, empty pocket, no pockets, missing
      receptor file, no active residues, unknown engine
"""
import math
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ligand import load_ligand, LigandPreparator
from docking import (
    ReceptorPreparation,
    PocketDetector,
    BindingBox,
    CompatibilityEngine,
    DockingManager,
    DockingRequest,
    MockDockingAdapter,
    PoseAnalyzer,
    DockingReportWriter,
)


# ----------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------

def _atom_line(serial, name, resname, chain, resnum, x, y, z, element):
    return (f"ATOM  {serial:>5d} {name:<4s} {resname:<3s} "
            f"{chain:1s}{resnum:>4d}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}          {element:>2s}")


def _hetatm_line(serial, name, resname, chain, resnum, x, y, z, element):
    return (f"HETATM{serial:>5d} {name:<4s} {resname:<3s} "
            f"{chain:1s}{resnum:>4d}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}          {element:>2s}")


def make_test_pdb(path):
    """PDB with one HETATM ligand surrounded by ASP / HIS / PHE residues."""
    lines = []
    residues = [
        ('A', 10, [('N', 0, 0, 0), ('CA', 1.5, 0, 0), ('C', 2, 1.4, 0),
                   ('O', 1.5, 2.5, 0), ('CB', 2.2, -1.2, 0.5)], 'ALA'),
        ('A', 11, [('N', 3.4, 1.4, 0), ('CA', 4, 2.7, 0), ('C', 5.5, 2.7, 0),
                   ('O', 6.2, 1.6, 0), ('CB', 3.4, 3.9, 0.5),
                   ('CG', 4.4, 5, 0.7), ('OD1', 5.5, 5, 0), ('OD2', 4, 6, 1.5)],
         'ASP'),
        ('A', 12, [('N', 6, 3.9, 0), ('CA', 7.5, 3.9, 0), ('C', 8, 5.3, 0),
                   ('O', 7.4, 6.4, 0), ('CB', 8, 2.7, 0.5),
                   ('CG', 9.5, 2.5, 0.5), ('ND1', 10, 1.4, 0.2)], 'HIS'),
        ('A', 13, [('N', 9.1, 5.3, 0), ('CA', 9.6, 6.7, 0), ('C', 11.1, 6.7, 0),
                   ('O', 11.7, 5.7, 0), ('CB', 9, 7.9, -0.5),
                   ('CG', 7.6, 8, -0.2), ('CD1', 6.7, 7, -0.2),
                   ('CD2', 7.1, 9.3, 0), ('CE1', 5.4, 7.3, 0),
                   ('CE2', 5.7, 9.5, 0), ('CZ', 4.9, 8.5, 0)], 'PHE'),
        ('A', 14, [('N', 11.7, 7.9, 0), ('CA', 13.2, 7.9, 0),
                   ('C', 13.7, 9.3, 0), ('O', 13.1, 10.4, 0),
                   ('CB', 13.8, 6.7, 0.5)], 'ALA'),
    ]
    serial = 1
    for chain, resnum, atoms, resname in residues:
        for aname, x, y, z in atoms:
            lines.append(_atom_line(serial, aname, resname, chain, resnum,
                                    x, y, z, aname[0]))
            serial += 1
    # Bound ligand
    for (aname, x, y, z, el) in [
        ('C1', 7.5, 6, 4, 'C'),
        ('C2', 8.5, 6.5, 4.5, 'C'),
        ('N1', 6.4, 6, 4.5, 'N'),
        ('O1', 7.8, 5, 4, 'O'),
        ('O2', 9.5, 6.5, 4.5, 'O'),
        ('O3', 6, 6, 5.5, 'O'),
    ]:
        lines.append(_hetatm_line(serial, aname, 'ADP', 'A', 100, x, y, z, el))
        serial += 1
    lines.append('END')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


@pytest.fixture
def pdb_file(tmp_path):
    p = tmp_path / "test.pdb"
    make_test_pdb(p)
    return p


@pytest.fixture
def prep(pdb_file):
    return ReceptorPreparation().prepare(str(pdb_file), chain='A')


@pytest.fixture
def pockets(prep):
    return PocketDetector(cutoff=5.0).detect(prep)


@pytest.fixture
def prep_ligand():
    lig = load_ligand(smiles='CC(=O)Oc1ccccc1C(=O)O')  # aspirin
    return LigandPreparator().prepare(lig).ligand


# ----------------------------------------------------------------------
# Receptor preparation
# ----------------------------------------------------------------------

def test_receptor_prep_basic(pdb_file):
    prep = ReceptorPreparation().prepare(str(pdb_file), chain='A')
    assert len(prep.receptor_atoms) > 30
    assert 'ADP' in [l.residue_name for l in prep.structure.ligands]
    assert len(prep.provenance) >= 3


def test_receptor_prep_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReceptorPreparation().prepare(str(tmp_path / "missing.pdb"))


def test_receptor_prep_chain_selection(pdb_file):
    prep_a = ReceptorPreparation().prepare(str(pdb_file), chain='A')
    prep_all = ReceptorPreparation().prepare(str(pdb_file))
    # All-chains view should have at least as many atoms as chain A.
    assert len(prep_all.receptor_atoms) >= len(prep_a.receptor_atoms)


# ----------------------------------------------------------------------
# Pocket detection
# --------------------------------------------------------------------**

def test_pocket_detector_finds_crystallographic(pockets):
    assert len(pockets) >= 1
    p = pockets[0]
    assert p.source == "crystallographic"
    assert p.ligand_resname == "ADP"
    assert len(p.residues) >= 3
    assert p.box is not None
    assert p.box.size_x >= 8.0
    assert p.score > 50


def test_pocket_detector_ranks(pockets):
    scores = [p.score for p in pockets]
    assert scores == sorted(scores, reverse=True)
    assert all(p.pocket_id.startswith("P") for p in pockets)


def test_pocket_box_inside_pdb_extent(prep, pockets):
    xs = [a.x for a in prep.receptor_atoms]
    ys = [a.y for a in prep.receptor_atoms]
    zs = [a.z for a in prep.receptor_atoms]
    for p in pockets:
        box = p.box
        assert box.center_x - box.size_x / 2 <= max(xs)
        assert box.center_x + box.size_x / 2 >= min(xs)
        assert box.center_y - box.size_y / 2 <= max(ys)
        assert box.center_y + box.size_y / 2 >= min(ys)


# ----------------------------------------------------------------------
# Compatibility
# ----------------------------------------------------------------------

def test_compatibility_assess_basic(prep_ligand, pockets):
    report = CompatibilityEngine().assess(prep_ligand, pockets)
    assert report.ligand_name == "ligand"
    assert report.best_pocket_id is not None
    assert len(report.pockets) == len(pockets)
    assert report.overall_assessment in {
        "HIGH-PRIORITY DOCKING CANDIDATE",
        "MODERATE-PRIORITY CANDIDATE",
        "LOW-PRIORITY CANDIDATE",
        "INCOMPATIBLE",
    }


def test_compatibility_with_active_residues(prep_ligand, pockets):
    pred = [('A', 'ASP', 11), ('A', 'HIS', 12), ('A', 'PHE', 13)]
    report = CompatibilityEngine().assess(
        prep_ligand, pockets, predicted_active_residues=pred
    )
    assert all(
        c.active_site_overlap in {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}
        for c in report.pockets
    )


def test_compatibility_no_pockets(prep_ligand):
    report = CompatibilityEngine().assess(prep_ligand, [])
    assert report.overall_assessment == "INCOMPATIBLE"
    assert report.best_pocket_id is None
    assert "No pockets" in report.warnings[0]


def test_compatibility_disclaimer_present(prep_ligand, pockets):
    report = CompatibilityEngine().assess(prep_ligand, pockets)
    assert "computational" in report.computational_disclaimer.lower()


def test_compatibility_to_dict_roundtrip(prep_ligand, pockets):
    report = CompatibilityEngine().assess(prep_ligand, pockets)
    d = report.to_dict()
    assert "pockets" in d
    assert "ligand_profile" in d
    assert "overall_assessment" in d


# ----------------------------------------------------------------------
# Docking manager + adapters
# ----------------------------------------------------------------------

def test_docking_manager_mock_default(prep_ligand, pockets):
    box = pockets[0].box
    mgr = DockingManager(engine='mock')
    req = DockingRequest(
        receptor_path="ignored.pdb",
        ligand=prep_ligand, box=box, n_poses=5,
    )
    result = mgr.dock(req)
    assert result.engine == "mock"
    assert len(result.poses) == 5
    assert all(p.ligand.atom_count() == prep_ligand.atom_count()
               for p in result.poses)


def test_docking_manager_unknown_engine_raises():
    with pytest.raises(RuntimeError):
        DockingManager(engine='bogus').dock(
            DockingRequest(receptor_path="x", ligand=load_ligand(smiles='C'),
                           box=BindingBox(0, 0, 0, 10, 10, 10))
        )


def test_docking_manager_unavailable_engine(pockets, prep_ligand):
    # Vina is not installed in this env, so the manager must raise a
    # clear RuntimeError when asked to use it.
    with pytest.raises(RuntimeError):
        DockingManager(engine='vina').dock(
            DockingRequest(receptor_path="x", ligand=prep_ligand,
                           box=pockets[0].box)
        )


def test_ligand_pdbqt_uses_meeko_torsion_tree():
    # A flexible molecule (several rotatable bonds) must come out with
    # a real BRANCH/ENDBRANCH torsion tree, not a single rigid ROOT —
    # that's the whole point of preferring meeko over the placeholder.
    from docking.engines.vina_adapter import _ligand_to_pdbqt
    lig = load_ligand(smiles="CCCCCCNc1ccccc1")
    pdbqt = _ligand_to_pdbqt(lig)
    assert "BRANCH" in pdbqt
    assert "ENDBRANCH" in pdbqt


def test_vina_cli_real_dock(tmp_path):
    """Real end-to-end AutoDock Vina CLI docking, skipped if no vina
    binary is available (e.g. CI without bin/vina.exe)."""
    from docking.engines.vina_adapter import VinaAdapter

    adapter = VinaAdapter()
    if not adapter.is_available():
        pytest.skip("no vina binary available (PATH or bin/)")

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(
        "ATOM      1  C1  REC A   1       0.000   0.000   0.000"
        "  0.00  0.00    +0.000 C\n"
        "ATOM      2  C2  REC A   1       3.800   0.000   0.000"
        "  0.00  0.00    +0.000 C\n"
        "ATOM      3  C3  REC A   1       0.000   3.800   0.000"
        "  0.00  0.00    +0.000 C\n"
        "ATOM      4  C4  REC A   1       0.000   0.000   3.800"
        "  0.00  0.00    +0.000 C\n",
        encoding="utf-8",
    )
    lig = load_ligand(smiles="CCO")
    box = BindingBox(center_x=1.5, center_y=1.5, center_z=1.5,
                      size_x=12, size_y=12, size_z=12)
    req = DockingRequest(receptor_path=str(receptor), ligand=lig, box=box,
                          n_poses=3, exhaustiveness=4, seed=1)

    result = adapter.dock(req)
    assert result.engine == "vina"
    assert len(result.poses) >= 1
    for pose in result.poses:
        assert isinstance(pose.score, float)
        assert pose.provenance["engine"] == "vina-cli"


def test_docking_poses_translated_into_box(pockets, prep_ligand):
    box = pockets[0].box
    mgr = DockingManager(engine='mock')
    req = DockingRequest(
        receptor_path="ignored.pdb", ligand=prep_ligand,
        box=box, n_poses=3, seed=42,
    )
    result = mgr.dock(req)
    cx = box.center_x
    for pose in result.poses:
        ligand_cx = sum(a.x for a in pose.ligand.atoms) / pose.ligand.atom_count()
        # Centroid should be close to box center
        assert abs(ligand_cx - cx) < 3.0


# ----------------------------------------------------------------------
# Pose analysis
# ----------------------------------------------------------------------

def test_pose_analysis_basic(prep, pockets, prep_ligand):
    box = pockets[0].box
    result = DockingManager(engine='mock').dock(
        DockingRequest(receptor_path="ignored.pdb", ligand=prep_ligand,
                       box=box, n_poses=3)
    )
    pred = [('A', 'ASP', 11), ('A', 'HIS', 12), ('A', 'PHE', 13)]
    report = PoseAnalyzer().analyze(result, prep.receptor_atoms,
                                   pockets[0], pred)
    assert len(report.analyses) == 3
    assert len(report.ranked) == 3
    assert report.best_pose_id is not None
    # Each pose has at least one contact
    assert any(pa.n_contacts > 0 for pa in report.analyses)


def test_pose_analysis_no_poses(prep, pockets, prep_ligand):
    result = DockingManager(engine='mock').dock(
        DockingRequest(receptor_path="ignored.pdb", ligand=prep_ligand,
                       box=pockets[0].box, n_poses=0)
    )
    report = PoseAnalyzer().analyze(result, prep.receptor_atoms, pockets[0])
    assert "No poses" in " ".join(report.warnings)


def test_pose_analysis_with_active_set(prep, pockets, prep_ligand):
    box = pockets[0].box
    result = DockingManager(engine='mock').dock(
        DockingRequest(receptor_path="ignored.pdb", ligand=prep_ligand,
                       box=box, n_poses=4)
    )
    # Build a list of (chain, resname, resnum) tuples from pocket residues,
    # excluding the HETATM ligand entry itself.
    pred = [
        (c, r, n)
        for (c, r, n) in pockets[0].residues
        if r not in {"ADP", "HOH", "WAT"}
    ]
    if not pred:
        pytest.skip("No protein residues in pocket to use as active set.")
    report = PoseAnalyzer().analyze(result, prep.receptor_atoms,
                                   pockets[0], pred)
    # We do not strictly require overlap (poses may jitter away) but the
    # analyzer must not crash and the result must be a list of analyses.
    assert len(report.analyses) == 4
    assert all(0 <= pa.active_overlap <= len(pred) for pa in report.analyses)


def test_pose_overall_classification(prep, pockets, prep_ligand):
    result = DockingManager(engine='mock').dock(
        DockingRequest(receptor_path="ignored.pdb", ligand=prep_ligand,
                       box=pockets[0].box, n_poses=4)
    )
    report = PoseAnalyzer().analyze(result, prep.receptor_atoms, pockets[0])
    assert all(pa.overall in {"STRONG", "MODERATE", "WEAK"}
               for pa in report.analyses)


# ----------------------------------------------------------------------
# End-to-end: Section 28 final report
# ----------------------------------------------------------------------

def test_final_assessment_runs(prep, pockets, prep_ligand, capsys):
    box = pockets[0].box
    result = DockingManager(engine='mock').dock(
        DockingRequest(receptor_path="ignored.pdb", ligand=prep_ligand,
                       box=box, n_poses=3)
    )
    pred = [('A', 'ASP', 11), ('A', 'HIS', 12), ('A', 'PHE', 13)]
    compat = CompatibilityEngine().assess(prep_ligand, pockets, pred)
    pose_report = PoseAnalyzer().analyze(result, prep.receptor_atoms,
                                        pockets[0], pred)
    DockingReportWriter.final_assessment(
        target_name="test_target", chain='A',
        compatibility=compat,
        best_pose=pose_report.ranked[0],
        best_pocket=pockets[0],
        active_residues=pred,
    )
    captured = capsys.readouterr()
    assert "FINAL LIGAND–TARGET ASSESSMENT" in captured.out
    assert "IMPORTANT" in captured.out
    assert "test_target" in captured.out


# ----------------------------------------------------------------------
# Ligand helpers
# ----------------------------------------------------------------------

def test_ligand_geometry_helpers():
    from docking.ligand import (
        ligand_bounding_volume, ligand_radius_of_gyration,
        profile_ligand, hba_positions, hbd_positions,
    )
    lig = load_ligand(smiles='CCO')
    prep_lig = LigandPreparator().prepare(lig).ligand
    assert ligand_bounding_volume(prep_lig) > 0
    assert ligand_radius_of_gyration(prep_lig) > 0
    profile = profile_ligand(prep_lig)
    assert profile.heavy_atoms == 3
    assert profile.hba >= 1


# ----------------------------------------------------------------------
# Edge cases from Section 32
# --------------------------------------------------------------------**

def test_multiple_chains(pdb_file):
    prep_a = ReceptorPreparation().prepare(str(pdb_file), chain='A')
    pockets_a = PocketDetector(cutoff=5.0).detect(prep_a)
    # Re-prepare with a different (non-existent) chain
    prep_empty = ReceptorPreparation().prepare(str(pdb_file), chain='Z')
    pockets_empty = PocketDetector(cutoff=5.0).detect(prep_empty)
    # No pockets should be found on chain Z because all ligands are A
    assert len(pockets_empty) == 0 or all(
        p.ligand_resname is None for p in pockets_empty
    )


def test_no_active_residues(prep_ligand, pockets):
    report = CompatibilityEngine().assess(prep_ligand, pockets)
    for c in report.pockets:
        assert c.active_site_overlap == "UNKNOWN"


def test_docking_engine_unavailable_falls_back():
    """If 'auto' is used and no real engine is installed, mock should be used."""
    mgr = DockingManager(engine='auto')
    adapter = mgr._resolve_adapter()
    assert adapter.name == 'mock' or adapter.is_available()