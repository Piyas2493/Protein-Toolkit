"""
Adapter registry.

Adding a new engine: implement a class `MyAdapter(DockingAdapter)` with
`is_available` and `dock`, then add it to `KNOWN_ENGINES` below.
"""
from __future__ import annotations

from typing import Dict, Type

from .vina_adapter import VinaAdapter
from .gnina_adapter import GninaAdapter


KNOWN_ENGINES: Dict[str, Type] = {
    "vina": VinaAdapter,
    "gnina": GninaAdapter,
}


def load_adapter(name: str) -> Type:
    """Return the adapter class for `name` or raise KeyError."""
    if name not in KNOWN_ENGINES:
        raise KeyError(name)
    return KNOWN_ENGINES[name]
