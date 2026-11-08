import tempfile
from pathlib import Path

from structure.parser import parse_pdb
from structure.catalytic_geometry import SpatialCatalyticDetector

# The tests below are about the geometry alone, on bare CA-atom residues with no sequence
# motif, so they use the legacy mode. The default (strict) mode is tested further down.


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
    hits = SpatialCatalyticDetector(strict=False).predict(structure)
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
    hits = SpatialCatalyticDetector(strict=False).predict(structure)
    names = {h["name"] for h in hits}
    assert "Serine Protease Triad" not in names


def test_requires_distinct_instances_for_repeated_residue():
    # Aspartic Protease needs TWO distinct ASPs near each other, not one
    # ASP counted twice.
    lines = [
        _atom_line(1, "CA", "ASP", "A", 1, 0.0, 0.0, 0.0, "C"),
    ]
    structure = _parse(lines)
    hits = SpatialCatalyticDetector(strict=False).predict(structure)
    names = {h["name"] for h in hits}
    assert "Aspartic Protease" not in names

    lines.append(_atom_line(2, "CA", "ASP", "A", 2, 2.0, 0.0, 0.0, "C"))
    structure = _parse(lines)
    hits = SpatialCatalyticDetector(strict=False).predict(structure)
    names = {h["name"] for h in hits}
    assert "Aspartic Protease" in names


# ---- strict mode: motif evidence, motif-marked residues, side-chain atoms, MODRES ----

def _residue(serial, resnum, resname, atoms, record="ATOM  "):
    """One residue with named atoms (name, x, y, z); first atom is the CA."""
    lines = []
    for i, (name, x, y, z) in enumerate(atoms):
        lines.append(f"{record}{serial + i:>5d} {name:<4s} {resname:<3s} A{resnum:>4d}   "
                     f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {name[0]:>2s}")
    return lines


THREE = {"A": "ALA", "V": "VAL", "S": "SER", "H": "HIS", "C": "CYS", "G": "GLY", "D": "ASP",
         "W": "TRP", "Q": "GLN", "F": "PHE", "Y": "TYR"}
# trypsin-fold His loop VSAAHC ... Ser loop GDSGG, plus a decoy Ser at the far end
SEQ = "AAAVSAAHCAAAAAAAAAGDSGGAAAAAAAAAAAAAS"


def _triad_structure(names=SEQ):
    """Every residue sits far from the others (CA only, 12 A apart) except the catalytic
    His, the Ser of GDSGG, an Asp and a decoy Ser, whose side-chain atoms are placed
    within hydrogen-bond range of each other. Positions follow SEQ; `names` may change
    the residue letters (to break a motif) without moving anything."""
    his = SEQ.index("AAHC") + 2          # the His of VSAAHC
    ser = SEQ.index("GDSGG") + 2         # the catalytic Ser
    asp = SEQ.index("GDSGG") + 1         # an Asp that H-bonds the His
    decoy = len(SEQ) - 1                 # a Ser that is NOT in any motif
    place = {his: [("CA", 0, 0, 0), ("NE2", 2.0, 0, 0), ("ND1", -2.0, 0, 0)],
             ser: [("CA", 30, 30, 0), ("OG", 4.7, 0, 0)],            # 2.7 A from His NE2
             decoy: [("CA", 60, 0, 0), ("OG", 4.2, 1.0, 0)],         # nearer than the real Ser
             asp: [("CA", 90, 0, 0), ("OD1", -4.7, 0, 0)]}           # 2.7 A from His ND1
    lines, serial = [], 1
    for i, ch in enumerate(names):
        atoms = place.get(i, [("CA", 12.0 * i + 200, 200.0, 0.0)])
        lines += _residue(serial, i + 1, THREE.get(ch, "ALA"), atoms)
        serial += len(atoms)
    return _parse(lines)


def _names(structure_hits):
    return {h["name"]: h for h in structure_hits}


def test_strict_mode_needs_the_motif_not_just_the_residue_types():
    # identical residues and geometry, but both active-site loops are broken
    broken = SEQ.replace("HC", "HW").replace("GDSGG", "GDSAG")
    hits = SpatialCatalyticDetector(strict=True).predict(_triad_structure(broken))
    assert "Serine Protease Triad" not in {h["name"] for h in hits}
    # ...and with the motifs intact the same geometry is reported
    intact = SpatialCatalyticDetector(strict=True).predict(_triad_structure())
    assert "Serine Protease Triad" in {h["name"] for h in intact}


def test_strict_mode_picks_the_serine_the_motif_marks_not_a_nearer_decoy():
    hits = _names(SpatialCatalyticDetector(strict=True).predict(_triad_structure()))
    found = hits["Serine Protease Triad"]["residues_found"]
    assert f"A:SER{SEQ.index('GDSGG') + 3}" in found          # the Ser of GDSGG (1-based number)
    assert f"A:SER{len(SEQ)}" not in found                    # the decoy
    assert hits["Serine Protease Triad"]["motifs_found"]


def test_strict_mode_links_a_chain_shaped_triad_by_connectivity():
    # Ser and Asp are 9.4 A apart, each 2.7 A from the His: a chain, not a clique
    hits = SpatialCatalyticDetector(strict=True).predict(_triad_structure())
    assert "Serine Protease Triad" in {h["name"] for h in hits}


def test_modres_records_are_parsed_and_a_modified_cys_counts_as_cysteine():
    seq = "AAAQAAAGACWAASSAAAAAAHAAAA"            # Q.{3}G.C[YW].{2}[STAGC][STAGC]; catalytic Cys at index 9
    cys = seq.index("GACW") + 2
    his = seq.index("AHAAAA") + 1
    lines, serial = [], 1
    for i, ch in enumerate(seq):
        if i == cys:
            continue                                # the modified Cys is a HETATM, added below
        atoms = [("CA", 0, 0, 0), ("ND1", 0, 0, 0)] if i == his else [("CA", 12.0 * i + 200, 200.0, 0.0)]
        lines += _residue(serial, i + 1, THREE.get(ch, "ALA"), atoms)
        serial += len(atoms)
    lines += _residue(serial, cys + 1, "CSD", [("CA", 50, 0, 0), ("SG", 3.5, 0, 0)], record="HETATM")
    lines.insert(0, f"MODRES 1ABC CSD A {cys + 1:>4d}  CYS  S-OXY CYSTEINE")
    st = _parse(lines)
    assert st.modified_residues == {("A", cys + 1): "CYS"}
    hits = SpatialCatalyticDetector(strict=True).predict(st)
    assert "Cysteine Protease Dyad" in {h["name"] for h in hits}
