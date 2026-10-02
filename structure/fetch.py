"""
Fetch a PDB structure by ID from RCSB.
"""
import urllib.error
import urllib.request
from pathlib import Path

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "pdb_cache"


def looks_like_pdb_id(value: str) -> bool:
    """Classic RCSB ID shape: 4 chars, alphanumeric, starts with a digit."""
    value = value.strip()
    return len(value) == 4 and value[0].isdigit() and value.isalnum()


def fetch_pdb(pdb_id: str, cache_dir: Path = CACHE_DIR) -> Path:
    """Download a PDB file by ID from RCSB, caching it locally.

    Returns the local file path. Raises ValueError if the ID doesn't
    resolve to a real entry.
    """
    pdb_id = pdb_id.strip().upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{pdb_id}.pdb"

    if out_path.exists():
        return out_path

    url = RCSB_URL.format(pdb_id=pdb_id)
    req = urllib.request.Request(
        url, headers={"User-Agent": "ProteinToolkit/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(
            f"PDB ID '{pdb_id}' not found on RCSB (HTTP {exc.code})."
        ) from exc

    out_path.write_bytes(data)
    return out_path
