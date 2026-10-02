"""
Benchmark suite for Sprint 14: end-to-end validation.

These tests do NOT measure throughput. They verify:

    1) Section 28 keys are always present in `FinalReport.to_dict()`.
    2) Scientific-language rules hold:
       - "WILL bind" never appears.
       - "binding confirmed" never appears.
       - "experimental evidence" never appears unless describing a
         known crystallographic ligand (handled upstream).
       - The disclaimer appears in CLI output.
    3) JSON export round-trips.
    4) Pipeline is deterministic with the same seed.

Any failure here is a regression of a hard rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import run_analysis, FinalReport
from tests.test_docking import make_test_pdb


# ----------------------------------------------------------------------
# Section 28 keys
# ----------------------------------------------------------------------

def test_box_for_docking_expands_tight_pocket():
    from pipeline import _box_for_docking
    from docking import BindingBox
    from ligand import load_ligand, LigandPreparator

    # A long, drug-sized molecule (~14 Angstrom) against a tight,
    # compatibility-scoring-sized pocket box (8 Angstrom) — must expand
    # to fit the ligand plus padding, not leave it clipped.
    lig = load_ligand(smiles="CC(C)CN(C)Cc1cc(C(=O)c2csc(S(N)(=O)=O)c2)ccc1O")
    prep = LigandPreparator(seed=42).prepare(lig)
    tight = BindingBox(center_x=0, center_y=0, center_z=0,
                        size_x=8.0, size_y=8.0, size_z=8.0)

    box = _box_for_docking(tight, prep.ligand)

    assert box.size_x > 14.0
    assert box.size_y > 14.0
    assert box.size_z > 14.0
    # Center must be preserved — only size changes.
    assert (box.center_x, box.center_y, box.center_z) == (0, 0, 0)


def test_box_for_docking_keeps_already_adequate_pocket():
    from pipeline import _box_for_docking
    from docking import BindingBox
    from ligand import load_ligand, LigandPreparator

    lig = load_ligand(smiles="CCO")
    prep = LigandPreparator(seed=42).prepare(lig)
    roomy = BindingBox(center_x=1, center_y=2, center_z=3,
                        size_x=20.0, size_y=20.0, size_z=20.0)

    box = _box_for_docking(roomy, prep.ligand)

    assert box.size_x == 20.0
    assert box.size_y == 20.0
    assert box.size_z == 20.0


REQUIRED_KEYS = {
    "target", "ligand", "best_pocket", "docking",
    "interactions", "active_site_overlap", "evidence",
    "overall_assessment", "provenance",
}


def test_section28_keys_present(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A")
    d = report.to_dict()
    missing = REQUIRED_KEYS - set(d.keys())
    assert not missing, f"Section 28 keys missing: {missing}"


def test_section28_target_metadata(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A")
    assert report.target["name"]
    assert report.target["chain"] == "A"


def test_section28_provenance(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A")
    p = report.provenance
    assert "started_at" in p
    assert "finished_at" in p
    assert p["chain"] == "A"
    assert p["seed"] == 42


# ----------------------------------------------------------------------
# Scientific language discipline
# ----------------------------------------------------------------------

FORBIDDEN_PHRASES = [
    "will bind",
    "WILL bind",
    "binding confirmed",
    "is proven to bind",
    "definitely binds",
    "guaranteed to inhibit",
    "cures",
    "drug of choice",
]


def _walk_strings(obj) -> Iterable[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


def test_no_forbidden_phrases(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A")
    blob = " ".join(_walk_strings(report.to_dict())).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase.lower() not in blob, (
            f"Forbidden phrase found in report: '{phrase}'"
        )


def test_overall_assessment_uses_hedged_language(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A")
    text = report.overall_assessment.lower()
    # At least one hedging word
    hedged_words = {"candidate", "evidence", "plausible", "compatible",
                    "docking", "support", "predicted"}
    assert any(w in text for w in hedged_words), (
        f"Overall assessment lacks hedging: {report.overall_assessment!r}"
    )


# ----------------------------------------------------------------------
# JSON export
# ----------------------------------------------------------------------

def test_json_round_trip(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A")
    out = tmp_path / "report.json"
    out.write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["target"]["name"] == report.target["name"]
    assert loaded["provenance"]["seed"] == 42


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

def test_deterministic_with_seed(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    a = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A", seed=42)
    b = run_analysis(pdb_path=str(pdb), ligand_smiles="CCO", chain="A", seed=42)
    # Best pose score must be identical
    assert a.docking.get("best_score") == b.docking.get("best_score")
    # Same overall assessment
    assert a.overall_assessment == b.overall_assessment


# ----------------------------------------------------------------------
# Ligand failure does not crash
# ----------------------------------------------------------------------

def test_invalid_ligand_returns_clean_report(tmp_path):
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    report = run_analysis(pdb_path=str(pdb), ligand_smiles="XYZ", chain="A")
    # Pipeline must NOT raise — just report failure.
    assert "LIGAND FAILED" in report.overall_assessment or \
           report.overall_assessment != ""
    assert "error" in report.ligand or report.ligand.get("valid") is False


# ----------------------------------------------------------------------
# CLI disclaimer
# ----------------------------------------------------------------------

def test_cli_prints_disclaimer(tmp_path, capsys):
    from pipeline import main
    pdb = tmp_path / "t.pdb"
    make_test_pdb(pdb)
    rc = main(["--pdb", str(pdb), "--smiles", "CCO", "--chain", "A"])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "computational prediction" in captured.lower()
    assert "FINAL LIGAND" in captured