"""
ProteinToolkit Docking Package (Sprint 6-8)

Layout:
    docking/
        __init__.py            - this file
        ligand.py               - ligand-side helpers (geometry, hbond/contact tools)
        preparation.py          - receptor preparation (target PDB)
        pocket.py               - pocket detection, ranking, box definition
        compatibility.py        - pre-docking compatibility assessment
        manager.py              - DockingManager + adapter dispatch
        pose_analysis.py        - post-docking pose analysis (Sprint 8)
        report.py               - human-readable reports
        engines/                - docking engine adapters (Sprint 7)

Public API:
    from docking import (
        ReceptorPreparation,
        PocketDetector,
        CompatibilityEngine,
        DockingManager,
        PoseAnalyzer,
        CompatibilityReport,
        DockingReport,
    )
"""
from .preparation import ReceptorPreparation, write_receptor_pdbqt
from .pocket import PocketDetector, Pocket, BindingBox
from .compatibility import (
    CompatibilityEngine,
    CompatibilityReport,
    PocketCompatibility,
)
from .manager import DockingManager, DockingAdapter, DockingRequest, DockingResult, DockedPose, MockDockingAdapter
from .pose_analysis import PoseAnalyzer, PoseAnalysis, PoseRanker, PoseAnalysisReport
from .report import DockingReportWriter

__all__ = [
    "ReceptorPreparation",
    "write_receptor_pdbqt",
    "PocketDetector",
    "Pocket",
    "BindingBox",
    "CompatibilityEngine",
    "CompatibilityReport",
    "PocketCompatibility",
    "DockingManager",
    "DockingAdapter",
    "DockingRequest",
    "DockingResult",
    "DockedPose",
    "MockDockingAdapter",
    "PoseAnalyzer",
    "PoseAnalysis",
    "PoseRanker",
    "PoseAnalysisReport",
    "DockingReportWriter",
]
