"""
Gnina adapter (stub).

Gnina is a deep-learning docking fork of AutoDock Vina that adds a
CNN scoring function. We wrap it in an adapter that shells out to the
`gnina` binary. If the binary is not on PATH, `is_available()` returns
False.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..manager import DockingAdapter, DockingRequest, DockingResult, DockedPose


class GninaAdapter(DockingAdapter):
    name = "gnina"

    def __init__(self, gnina_binary: str = "gnina"):
        self.gnina_binary = gnina_binary

    def is_available(self) -> bool:
        return shutil.which(self.gnina_binary) is not None

    def dock(self, request: DockingRequest) -> DockingResult:
        if not self.is_available():
            raise RuntimeError(
                "Gnina is not installed. Install gnina and ensure the "
                "`gnina` binary is on PATH."
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            lig_in = tmpdir / "ligand.sdf"
            lig_out = tmpdir / "ligand_out.sdf"
            lig_in.write_text(_ligand_to_sdf(request.ligand), encoding="utf-8")

            cmd = [
                self.gnina_binary,
                "--receptor", request.receptor_path,
                "--ligand", str(lig_in),
                "--out", str(lig_out),
                "--center_x", str(request.box.center_x),
                "--center_y", str(request.box.center_y),
                "--center_z", str(request.box.center_z),
                "--size_x", str(request.box.size_x),
                "--size_y", str(request.box.size_y),
                "--size_z", str(request.box.size_z),
                "--exhaustiveness", str(request.exhaustiveness),
                "--num_modes", str(request.n_poses),
                "--seed", str(request.seed),
            ]
            subprocess.run(cmd, check=True, cwd=tmpdir)

            poses = _parse_sdf_poses(lig_out, request)

        return DockingResult(
            request=request, poses=poses, engine=self.name,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ligand_to_sdf(ligand) -> str:
    """Emit a minimal SDF V2000 block."""
    from datetime import datetime
    n_atoms = len(ligand.atoms)
    n_bonds = len(ligand.bonds)
    header = [
        ligand.name,
        "  ProteinToolkit",
        datetime.now().strftime("%m%d%y%H%M"),
        "",
    ]
    counts = f"{n_atoms:>3d}{n_bonds:>3d}  0  0  0  0  0  0  0  0999 V2000"
    body = []
    for a in ligand.atoms:
        body.append(
            f"{a.x:>10.4f}{a.y:>10.4f}{a.z:>10.4f} {a.element:<3s} 0  0  0  0  0  0  0  0  0  0"
        )
    bonds = []
    for b in ligand.bonds:
        order = int(round(b.order))
        bonds.append(f"{b.a1:>3d}{b.a2:>3d}{order:>3d}  0")
    return "\n".join(header + [counts] + body + bonds + ["M  END", "$$$$", ""])


def _parse_sdf_poses(path: Path, request: DockingRequest):
    """Read multi-mol SDF into a list of DockedPose."""
    poses = []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("$$$$")
    pose_id = 1
    for block in blocks:
        if "V2000" not in block:
            continue
        # Score is conventionally the last line before M  END as comment
        score = -7.0
        rmsd = 0.0
        for line in block.splitlines():
            if line.startswith(">  <minimizedAffinity>"):
                # next non-empty line is the score
                continue
        # We don't actually parse CNN scores here without a real run;
        # leave score/rmsd as placeholders.
        poses.append(DockedPose(
            pose_id=pose_id,
            score=score,
            ligand=request.ligand,
            rmsd_to_input=rmsd,
            provenance={"engine": "gnina", "raw": True},
        ))
        pose_id += 1
        if pose_id > request.n_poses:
            break
    return poses
