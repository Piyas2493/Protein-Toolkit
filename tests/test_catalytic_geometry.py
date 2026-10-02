import tempfile
from pathlib import Path

from structure.parser import parse_pdb
from structure.catalytic_geometry import SpatialCatalyticDetector


def _atom_line(serial, name, resname, chain, resnum, x, y, z, element):
    return (
        f"ATOM  {serial:>5d} {name:<4s} {resname:<3s} "
        f"{chain:1s}{resnum:>4d}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"{1.00:>6.2f}{20.00:>6.2f}          {element:>2s}"
    )


def _parse(lines):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "t.pdb"
        path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
        return parse_pdb(str(path))


def test_clustered_triad_detected():
    # SER, HIS, ASP all within a few Angstrom of each other.
    lines = [
        _atom_line(1, "CA", "SER", "A", 1, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "HIS", "A", 2, 2.0, 0.0, 0.0, "C"),
        _atom_line(3, "CA", "ASP", "A", 3, 4.0, 0.0, 0.0, "C"),
    ]
    structure = _parse(lines)
    hits = SpatialCatalyticDetector().predict(structure)
    names = {h["name"] for h in hits}
    assert "Serine Protease Triad" in names


def test_scattered_residues_not_detected():
    # Same residue types, but far apart — should NOT be flagged, unlike
    # the sequence-only composition check this is meant to improve on.
    lines = [
        _atom_line(1, "CA", "SER", "A", 1, 0.0, 0.0, 0.0, "C"),
        _atom_line(2, "CA", "HIS", "A", 2, 100.0, 0.0, 0.0, "C"),
        _atom_line(3, "CA", "ASP", "A", 3, 0.0, 100.0, 0.0, "C"),
    ]
    structure = _parse(lines)
    hits = SpatialCatalyticDetector().predict(structure)
    names = {h["name"] for h in hits}
    assert "Serine Protease Triad" not in names


def test_requires_distinct_instances_for_repeated_residue():
    # Aspartic Protease needs TWO distinct ASPs near each other, not one
    # ASP counted twice.
    lines = [
        _atom_line(1, "CA", "ASP", "A", 1, 0.0, 0.0, 0.0, "C"),
    ]
    structure = _parse(lines)
    hits = SpatialCatalyticDetector().predict(structure)
    names = {h["name"] for h in hits}
    assert "Aspartic Protease" not in names

    lines.append(_atom_line(2, "CA", "ASP", "A", 2, 2.0, 0.0, 0.0, "C"))
    structure = _parse(lines)
    hits = SpatialCatalyticDetector().predict(structure)
    names = {h["name"] for h in hits}
    assert "Aspartic Protease" in names
