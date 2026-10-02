"""
AutoDock Vina adapter.

Two backends, tried in order:
    1) The `vina` Python package (pip install vina) — not buildable on
       native Windows without Boost + a C++ toolchain.
    2) The Vina CLI binary — a precompiled `vina.exe`/`vina` on PATH,
       or bundled at `<project_root>/bin/vina(.exe)`. This is how a
       Windows checkout gets real docking without a from-source build:
           https://github.com/ccsb-scripps/AutoDock-Vina/releases

The adapter is registered with the manager via
`docking.engines.registry.KNOWN_ENGINES`.

If neither is available, `is_available()` returns False and the
manager will raise a clear error.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from ..manager import DockingAdapter, DockingRequest, DockingResult, DockedPose

# `<project_root>/bin/`, checked as a fallback when the binary isn't on PATH.
_BUNDLED_BIN_DIR = Path(__file__).resolve().parents[2] / "bin"

_RESULT_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$"
)


class VinaAdapter(DockingAdapter):
    name = "vina"

    def __init__(self, vina_binary: str = "vina"):
        self.vina_binary = vina_binary

    def _resolve_cli_binary(self) -> Optional[str]:
        """PATH first, then the project-local bin/ folder."""
        found = shutil.which(self.vina_binary)
        if found:
            return found
        for candidate in (_BUNDLED_BIN_DIR / "vina.exe",
                          _BUNDLED_BIN_DIR / "vina"):
            if candidate.exists():
                return str(candidate)
        return None

    def is_available(self) -> bool:
        # Prefer the Python package; fall back to the CLI.
        try:
            import vina  # noqa: F401
            return True
        except ImportError:
            pass
        return self._resolve_cli_binary() is not None

    def dock(self, request: DockingRequest) -> DockingResult:
        try:
            import vina  # type: ignore
        except ImportError:
            # Subprocess fallback to the CLI binary
            binary = self._resolve_cli_binary()
            if binary is None:
                raise RuntimeError(
                    "AutoDock Vina is not installed. `pip install vina` "
                    "or place a vina binary on PATH / in bin/."
                )
            return self._dock_cli(request, binary)

        # Pure-Python `vina` package flow
        v = vina.Vina(sf_name="vina")
        v.set_receptor(request.receptor_path)
        # Ligand must be an SDF or PDBQT string/path; we write to temp.
        from pathlib import Path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(_ligand_to_pdbqt(request.ligand).encode("utf-8"))
        v.set_ligand_from_file(tmp_path)
        v.compute_vina_maps(
            center=[
                request.box.center_x,
                request.box.center_y,
                request.box.center_z,
            ],
            box_size=[
                request.box.size_x,
                request.box.size_y,
                request.box.size_z,
            ],
        )
        v.dock(
            exhaustiveness=request.exhaustiveness,
            n_poses=request.n_poses,
        )
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as out:
            out_path = out.name
        v.write_poses(out_path, n_poses=request.n_poses, overwrite=True)

        # Parse outputs
        poses: List[DockedPose] = []
        energies = v.energies()  # (n_poses, 4) array
        for i, row in enumerate(energies):
            score = float(row[0])  # total
            poses.append(DockedPose(
                pose_id=i + 1,
                score=score,
                ligand=request.ligand,        # placeholder; coords in pdbqt
                rmsd_to_input=float(row[1]),
                provenance={"engine": "vina", "inter": float(row[2]),
                             "intra": float(row[3]), "pdbqt": out_path},
            ))

        return DockingResult(
            request=request, poses=poses, engine=self.name,
        )

    def _dock_cli(self, request: DockingRequest, binary: str) -> DockingResult:
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".pdbqt", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            ligand_path = tmp.name
            tmp.write(_ligand_to_pdbqt(request.ligand))

        out_path = tempfile.mktemp(suffix="_out.pdbqt")

        cmd = [
            binary,
            "--receptor", request.receptor_path,
            "--ligand", ligand_path,
            "--center_x", str(request.box.center_x),
            "--center_y", str(request.box.center_y),
            "--center_z", str(request.box.center_z),
            "--size_x", str(request.box.size_x),
            "--size_y", str(request.box.size_y),
            "--size_z", str(request.box.size_z),
            "--exhaustiveness", str(request.exhaustiveness),
            "--num_modes", str(request.n_poses),
            "--seed", str(request.seed),
            "--out", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Vina CLI failed (exit {proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )

        poses = _parse_vina_cli_stdout(proc.stdout, request.ligand)
        return DockingResult(
            request=request, poses=poses, engine=self.name,
            warnings=[] if poses else ["Vina CLI produced no poses."],
        )


def _parse_vina_cli_stdout(stdout: str, ligand) -> List[DockedPose]:
    """Parse Vina's results table, e.g.:

        mode |   affinity | dist from best mode
             | (kcal/mol) | rmsd l.b.| rmsd u.b.
        -----+------------+----------+----------
           1       -6.5      0.000      0.000
           2       -6.2      1.234      2.345

    `rmsd l.b.` is relative to the top pose (Vina docks into the box
    from scratch, not from a specific starting conformation), not to
    the original input ligand — noted in each pose's provenance.
    """
    poses: List[DockedPose] = []
    for line in stdout.splitlines():
        m = _RESULT_ROW_RE.match(line)
        if not m:
            continue
        mode, affinity, rmsd_lb, rmsd_ub = m.groups()
        poses.append(DockedPose(
            pose_id=int(mode),
            score=float(affinity),
            ligand=ligand,  # placeholder; real coords are in the out PDBQT
            rmsd_to_input=float(rmsd_lb),
            provenance={"engine": "vina-cli", "rmsd_lb": float(rmsd_lb),
                        "rmsd_ub": float(rmsd_ub),
                        "rmsd_reference": "best_pose, not input"},
        ))
    return poses


def _ligand_to_pdbqt_meeko(smiles: str) -> str:
    """Real PDBQT via meeko: embed + MMFF-optimize a 3D conformer, then
    let meeko derive the torsion tree and partial charges. Raises
    ImportError if rdkit/meeko aren't available (caller falls back)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not re-parse ligand SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    setups = MoleculePreparation().prepare(mol)
    pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise ValueError(f"meeko could not write PDBQT: {err}")
    return pdbqt_string


def _ligand_to_pdbqt(ligand) -> str:
    """Convert the internal Ligand object to PDBQT.

    Prefers `meeko` (the AutoDock/Vina team's own RDKit-to-PDBQT
    preparer): real rotatable-bond torsion trees and Gasteiger-like
    partial charges, not one rigid, un-optimized conformer. Docking a
    whole flexible molecule as a single rigid body routinely produces
    clashing poses or none at all — this isn't a cosmetic gap.

    Falls back to a minimal rigid-body placeholder (no torsions, zero
    charges) when meeko isn't installed or the ligand has no SMILES
    (e.g. loaded from SDF/MOL2/PDB) to build one from.
    """
    if ligand.smiles:
        try:
            return _ligand_to_pdbqt_meeko(ligand.smiles)
        except ImportError:
            pass

    lines = ["ROOT"]
    for a in ligand.atoms:
        # PDBQT atom record
        #  1-6   "ATOM  "
        #  7-11  serial
        #  13-16 atom name
        #  17    altLoc
        #  18-20 resName ("UNL")
        #  22    chain
        #  23-26 resSeq
        #  31-38 x
        #  39-46 y
        #  47-54 z
        #  55-60 occupancy
        #  61-66 tempFactor
        #  71-76 charge (%6.3f — we don't compute real Gasteiger charges)
        #  78-79 AutoDock atom type (%-2s, left-justified)
        # AutoDock atom types are case-sensitive (e.g. "Zn", not "ZN").
        atom_type = a.element.capitalize()
        lines.append(
            f"ATOM  {a.serial:>5d} {a.element:<4s} UNL X   1    "
            f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}    {0.0:>6.3f} {atom_type:<2s}"
        )
    lines.append("ENDROOT")
    lines.append("TORSDOF 0")
    return "\n".join(lines) + "\n"
