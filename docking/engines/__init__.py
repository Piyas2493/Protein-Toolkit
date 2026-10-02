"""
Docking engine adapters (Sprint 7).

Each adapter is a thin wrapper over an external docking tool. They are
loaded lazily — only when the user explicitly requests a specific
engine. If the engine is not installed, `is_available()` returns False
and the manager raises a clear error.

Currently registered:
    - vina     (requires the `vina` Python package)
    - gnina    (requires the `gnina` binary on PATH)

When neither is available, the manager falls back to the mock adapter
in `docking.manager`.
"""
from .registry import load_adapter, KNOWN_ENGINES

__all__ = ["load_adapter", "KNOWN_ENGINES"]
