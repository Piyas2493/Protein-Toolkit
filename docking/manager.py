"""
DockingManager + adapter pattern (Sprint 7).

    DockingManager
        │
        ├── VinaAdapter
        ├── GninaAdapter
        └── FutureDockingAdapter

The manager does not implement docking itself — it dispatches to a
configured adapter. Adapters are loaded lazily from `docking.engines`.
If no real docking engine is installed, the manager falls back to a
`MockDockingAdapter` that returns plausible-shaped poses for testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ligand import Ligand, LigandAtom, LigandBond

from .pocket import Pocket, BindingBox
from .preparation import PreparedReceptor


# ----------------------------------------------------------------------
# Engine interfaces
# ----------------------------------------------------------------------

@dataclass
class DockingRequest:
    receptor_path: str
    ligand: Ligand
    box: BindingBox
    n_poses: int = 10
    exhaustiveness: int = 8
    seed: int = 42
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DockedPose:
    pose_id: int
    score: float                # kcal/mol (engine-specific)
    ligand: Ligand              # translated/rotated ligand
    rmsd_to_input: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DockingResult:
    request: DockingRequest
    poses: List[DockedPose] = field(default_factory=list)
    engine: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "poses": [
                {
                    "pose_id": p.pose_id,
                    "score": round(p.score, 3),
                    "rmsd_to_input": round(p.rmsd_to_input, 3),
                }
                for p in self.poses
            ],
            "warnings": self.warnings,
        }


class DockingAdapter:
    """Abstract base for docking engine adapters."""
    name: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def dock(self, request: DockingRequest) -> DockingResult:
        raise NotImplementedError


# ----------------------------------------------------------------------
# Mock adapter (always available; used for testing without Vina)
# ----------------------------------------------------------------------

class MockDockingAdapter(DockingAdapter):
    name = "mock"

    def __init__(self, score_jitter: float = 0.8, jitter_radius: float = 0.8):
        self.score_jitter = score_jitter
        self.jitter_radius = jitter_radius

    def is_available(self) -> bool:
        return True

    def dock(self, request: DockingRequest) -> DockingResult:
        import math
        import random
        rng = random.Random(request.seed)

        # Translate ligand centroid to box center, then jitter slightly.
        cx, cy, cz = request.ligand.center()
        bx, by, bz = request.box.center_x, request.box.center_y, request.box.center_z
        dx, dy, dz = bx - cx, by - cy, bz - cz

        # Clone ligand atoms and translate.
        atoms = [
            LigandAtom(
                serial=a.serial, element=a.element,
                x=a.x + dx, y=a.y + dy, z=a.z + dz,
                formal_charge=a.formal_charge,
                aromatic=a.aromatic, in_ring=a.in_ring,
                h_count=a.h_count,
            )
            for a in request.ligand.atoms
        ]
        translated = Ligand(
            name=request.ligand.name,
            atoms=atoms,
            bonds=[
                LigandBond(
                    a1=b.a1, a2=b.a2, order=b.order,
                    aromatic=b.aromatic, in_ring=b.in_ring,
                )
                for b in request.ligand.bonds
            ],
            source_format=request.ligand.source_format,
            smiles=request.ligand.smiles,
        )

        poses: List[DockedPose] = []
        for i in range(request.n_poses):
            # Each pose jitters around the translated centroid with
            # bounded amplitude. Lower jitter produces more realistic
            # pose diversity without putting the ligand on top of
            # every pocket atom.
            jx = rng.uniform(-self.jitter_radius, self.jitter_radius)
            jy = rng.uniform(-self.jitter_radius, self.jitter_radius)
            jz = rng.uniform(-self.jitter_radius, self.jitter_radius)
            jittered = Ligand(
                name=f"{request.ligand.name}_pose{i+1}",
                atoms=[
                    LigandAtom(
                        serial=a.serial, element=a.element,
                        x=a.x + jx + rng.uniform(-0.3, 0.3),
                        y=a.y + jy + rng.uniform(-0.3, 0.3),
                        z=a.z + jz + rng.uniform(-0.3, 0.3),
                        formal_charge=a.formal_charge,
                        aromatic=a.aromatic, in_ring=a.in_ring,
                        h_count=a.h_count,
                    )
                    for a in atoms
                ],
                bonds=[
                    LigandBond(a1=b.a1, a2=b.a2, order=b.order,
                               aromatic=b.aromatic, in_ring=b.in_ring)
                    for b in request.ligand.bonds
                ],
                source_format=request.ligand.source_format,
                smiles=request.ligand.smiles,
            )
            score = -7.0 + rng.uniform(-self.score_jitter, self.score_jitter) - i * 0.3
            rmsd = rng.uniform(0.5, 4.5)
            poses.append(DockedPose(
                pose_id=i + 1,
                score=round(score, 2),
                ligand=jittered,
                rmsd_to_input=round(rmsd, 2),
                provenance={"engine": "mock", "jitter": True},
            ))

        result = DockingResult(
            request=request, poses=poses, engine=self.name,
            warnings=["MockDockingAdapter — not an actual docking engine."],
        )
        return result


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------

class DockingManager:
    """Dispatch docking work to the configured adapter.

    Usage:
        manager = DockingManager(engine="mock")  # or "vina", "gnina"
        result = manager.dock(request)
    """

    _ADAPTERS: Dict[str, type] = {}

    def __init__(self, engine: str = "auto", *, adapter: Optional[DockingAdapter] = None):
        self.engine_name = engine
        self._adapter: Optional[DockingAdapter] = adapter
        if adapter is not None:
            self.engine_name = adapter.name

    @classmethod
    def register_adapter(cls, name: str, adapter_cls: type) -> None:
        cls._ADAPTERS[name] = adapter_cls

    def _resolve_adapter(self) -> DockingAdapter:
        if self._adapter is not None:
            return self._adapter
        if self.engine_name == "auto":
            for candidate in ("vina", "gnina"):
                if candidate in self._ADAPTERS:
                    adapter = self._ADAPTERS[candidate]()
                    if adapter.is_available():
                        self._adapter = adapter
                        self.engine_name = candidate
                        return adapter
            # Fallback to mock
            self._adapter = MockDockingAdapter()
            self.engine_name = self._adapter.name
            return self._adapter

        if self.engine_name == "mock":
            self._adapter = MockDockingAdapter()
            return self._adapter

        cls = self._ADAPTERS.get(self.engine_name)
        if cls is None:
            # Lazy import: external adapters may not be present
            from .engines import load_adapter
            try:
                cls = load_adapter(self.engine_name)
                self._ADAPTERS[self.engine_name] = cls
            except KeyError:
                raise RuntimeError(
                    f"Unknown docking engine '{self.engine_name}'. "
                    f"Available: {sorted(self._ADAPTERS)}"
                )
        adapter = cls()
        if not adapter.is_available():
            raise RuntimeError(
                f"Docking engine '{self.engine_name}' is not available "
                "in this environment."
            )
        self._adapter = adapter
        return adapter

    def dock(self, request: DockingRequest) -> DockingResult:
        adapter = self._resolve_adapter()
        result = adapter.dock(request)
        # Provenance
        result.engine = adapter.name
        return result
