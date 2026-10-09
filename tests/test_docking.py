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


def test_receptor_atom_typing_acceptors_and_aromaticity():
    from docking.preparation import _autodock_atom_type
    from structure.models import Atom

    def atom(name, resname, element):
        return Atom(serial=1, atom_name=name, residue_name=resname,
                    chain_id="A", residue_number=1, x=0, y=0, z=0,
                    occupancy=1.0, b_factor=0.0, element=element)

    assert _autodock_atom_type(atom("CZ", "PHE", "C")) == "A"
    assert _autodock_atom_type(atom("CB", "ALA", "C")) == "C"
    assert _autodock_atom_type(atom("O", "ALA", "O")) == "OA"
    assert _autodock_atom_type(atom("ND1", "HIS", "N")) == "NA"
    assert _autodock_atom_type(atom("N", "ALA", "N")) == "N"
    assert _autodock_atom_type(atom("SG", "CYS", "S")) == "SA"
    assert _autodock_atom_type(atom("ZN", "ZN", "ZN")) == "Zn"


VINA_STDOUT_TABLE = """\
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -6.185          0          0
   2         -6.1      1.767      2.689
   3       -6.098       3.55      6.624
   4       -5.809      1.851      2.248
   5       -5.663      3.098      6.493
"""


def test_receptor_excludes_bound_ligand_additives_and_altlocs(tmp_path):
    # The receptor must be protein + kept metals only. Building it from
    # every ATOM/HETATM record left the crystal ligand sitting in its own
    # pocket (so docking it back was impossible), duplicated metals, and
    # kept both alternate conformations of a disordered residue.
    def line(record, serial, name, res, resnum, x, alt=" ", el="C"):
        return (f"{record:<6s}{serial:>5d} {name:<4s}{alt}{res:>3s} A{resnum:>4d}"
                f"    {x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.00:>6.2f}{10.00:>6.2f}"
                f"          {el:>2s}")

    pdb = tmp_path / "t.pdb"
    pdb.write_text("\n".join([
        line("ATOM", 1, "CA", "ALA", 1, 0.0),
        line("ATOM", 2, "CB", "SER", 2, 3.0, alt="A"),
        line("ATOM", 3, "CB", "SER", 2, 3.4, alt="B"),     # 2nd conformer
        line("HETATM", 4, "ZN", "ZN", 3, 6.0, el="ZN"),    # metal: keep once
        line("HETATM", 5, "C1", "LIG", 4, 9.0),            # bound ligand
        line("HETATM", 6, "C1", "GOL", 5, 12.0),           # additive
        "END",
    ]) + "\n", encoding="utf-8")

    prep = ReceptorPreparation().prepare(str(pdb), chain="A")
    names = sorted((a.residue_name, a.atom_name) for a in prep.receptor_atoms)

    assert names == [("ALA", "CA"), ("SER", "CB"), ("ZN", "ZN")]
    assert "LIG" in prep.known_ligand_resnames   # still reported, not docked into


def _classify(**overrides):
    from docking.pose_analysis import PoseAnalyzer
    args = dict(score=-3.0, n_clashes=0, n_hbond_residues=0, n_metal=0,
                n_nonpolar_residues=0, active_overlap=0)
    args.update(overrides)
    return PoseAnalyzer()._classify_pose(**args)


def test_pose_label_counts_independent_lines_of_evidence():
    # No support at all, or a single line, is WEAK.
    assert _classify() == "WEAK"
    assert _classify(score=-7.0) == "WEAK"
    # Two independent lines -> MODERATE; three -> STRONG.
    assert _classify(score=-7.0, n_metal=1) == "MODERATE"
    assert _classify(score=-7.0, n_hbond_residues=2,
                     n_nonpolar_residues=3) == "STRONG"
    # STRONG is reachable without user-supplied active-site residues.
    assert _classify(score=-6.5, n_metal=1, n_nonpolar_residues=4) == "STRONG"
    # A strained pose is WEAK regardless of how much else supports it.
    assert _classify(score=-9.0, n_metal=1, n_hbond_residues=4,
                     n_nonpolar_residues=5, n_clashes=6) == "WEAK"


def test_pose_label_not_inflated_by_atom_pair_counts():
    # A tiny molecule touching ONE residue many times must not look
    # supported: the rule counts distinct residues, not atom pairs.
    assert _classify(score=-2.9, n_hbond_residues=1,
                     n_nonpolar_residues=1) == "WEAK"


def test_metal_coordination_ignores_alpha_carbons():
    # "CA" is every residue's alpha carbon (and calcium's residue name):
    # matching on atom name counted each backbone C-alpha as a calcium.
    from docking.pose_analysis import PoseAnalyzer
    from ligand import LigandAtom
    from structure.models import Atom

    alpha_c = Atom(serial=1, atom_name="CA", residue_name="ALA", chain_id="A",
                   residue_number=1, x=0, y=0, z=0, occupancy=1.0,
                   b_factor=0.0, element="C")
    zinc = Atom(serial=2, atom_name="ZN", residue_name="ZN", chain_id="A",
                residue_number=2, x=0, y=0, z=0, occupancy=1.0,
                b_factor=0.0, element="ZN")
    donor = LigandAtom(serial=1, element="N", x=2.0, y=0, z=0)
    far_donor = LigandAtom(serial=2, element="N", x=3.8, y=0, z=0)

    analyzer = PoseAnalyzer()
    assert analyzer._classify_interaction(donor, alpha_c, 2.0) != "Metal Coordination"
    assert analyzer._classify_interaction(donor, zinc, 2.0) == "Metal Coordination"
    # Beyond coordination distance it is not coordination.
    assert analyzer._classify_interaction(far_donor, zinc, 3.8) != "Metal Coordination"


def test_pose_ranking_is_score_dominant():
    # Atom-pair contact counts used to outweigh many kcal/mol of score.
    from docking.pose_analysis import PoseAnalyzer, PoseAnalysis

    def pose(pose_id, score, n_hbonds):
        return PoseAnalysis(
            pose_id=pose_id, score=score, rmsd_to_input=0.0, n_contacts=50,
            n_hbonds=n_hbonds, n_hydrophobic=0, n_ionic=0, n_aromatic=0,
            n_metal=0, n_clashes=0, active_overlap=0, active_overlap_ratio=0.0,
            pocket_occupancy=0.0, overall="WEAK")

    analyzer = PoseAnalyzer()
    better = pose(1, -6.2, n_hbonds=4)
    worse_but_many_pairs = pose(2, -5.8, n_hbonds=9)
    assert analyzer._rank_key(better) > analyzer._rank_key(worse_but_many_pairs)


def test_vina_stdout_parser_keeps_best_pose():
    # Real Vina output. C++ stream formatting prints "0" and "-6.1", not
    # "0.000" / "-6.100"; a regex requiring a decimal point silently
    # dropped mode 1 — the best pose of every run.
    from docking.engines.vina_adapter import _parse_vina_cli_stdout
    poses = _parse_vina_cli_stdout(VINA_STDOUT_TABLE, ligand=None)
    assert [p.pose_id for p in poses] == [1, 2, 3, 4, 5]
    assert poses[0].score == -6.185
    assert poses[1].score == -6.1


def test_docked_coordinates_replace_input_ligand(tmp_path):
    # Poses must carry the DOCKED coordinates from Vina's output PDBQT,
    # not the undocked input conformer — otherwise pose analysis counts
    # contacts for a molecule that is nowhere near the pocket.
    from docking.engines.vina_adapter import _attach_docked_coordinates
    from docking.manager import DockedPose

    out = tmp_path / "out.pdbqt"
    out.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT:     -5.000      0.000      0.000\n"
        "ATOM      1  C   UNL     1      10.000  20.000  30.000  1.00  0.00    +0.000 A \n"
        "ATOM      2  O   UNL     1      11.000  21.000  31.000  1.00  0.00    -0.300 OA\n"
        "ATOM      3  H   UNL     1      11.500  21.500  31.500  1.00  0.00    +0.200 HD\n"
        "ENDMDL\n",
        encoding="utf-8",
    )
    undocked = load_ligand(smiles="CO")
    poses = [DockedPose(pose_id=1, score=-5.0, ligand=undocked)]

    _attach_docked_coordinates(poses, str(out), undocked)

    atoms = poses[0].ligand.atoms
    assert [a.element for a in atoms] == ["C", "O"]   # AD types mapped, H dropped
    assert atoms[0].aromatic is True                   # "A" = aromatic carbon
    assert (atoms[0].x, atoms[0].y, atoms[0].z) == (10.0, 20.0, 30.0)


def test_poses_outside_vina_energy_window_are_dropped_not_fatal(tmp_path):
    # Real case (181L benzene): Vina's stdout listed 8 modes but wrote 6
    # to the PDBQT, because modes 7-8 (+0.08, +24.8) were outside the 3
    # kcal/mol --energy_range. The file's models are a prefix of the table.
    # A real disagreement (scores that don't line up) must still raise.
    import pytest
    from docking.engines.vina_adapter import _attach_docked_coordinates
    from docking.manager import DockedPose

    def model(n, score):
        return (f"MODEL {n}\n"
                f"REMARK VINA RESULT:   {score:8.3f}      0.000      0.000\n"
                "ATOM      1  C   UNL     1       1.000   2.000   3.000"
                "  1.00  0.00    +0.000 C \n"
                "ENDMDL\n")

    out = tmp_path / "out.pdbqt"
    out.write_text(model(1, -5.468) + model(2, -4.273), encoding="utf-8")
    lig = load_ligand(smiles="C")
    table = [-5.468, -4.273, 0.07634, 24.81]
    poses = [DockedPose(pose_id=i + 1, score=sc, ligand=lig)
             for i, sc in enumerate(table)]

    assert _attach_docked_coordinates(poses, str(out), lig) == 2
    assert [p.pose_id for p in poses] == [1, 2]

    out.write_text(model(1, -5.468) + model(2, -3.000), encoding="utf-8")
    poses = [DockedPose(pose_id=i + 1, score=sc, ligand=lig)
             for i, sc in enumerate(table)]
    with pytest.raises(RuntimeError):
        _attach_docked_coordinates(poses, str(out), lig)


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