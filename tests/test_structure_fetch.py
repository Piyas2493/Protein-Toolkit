import urllib.error

import pytest

from structure.fetch import fetch_pdb, looks_like_pdb_id


def test_looks_like_pdb_id():
    assert looks_like_pdb_id("1LYZ") is True
    assert looks_like_pdb_id("1lyz") is True
    assert looks_like_pdb_id("lysozyme.pdb") is False
    assert looks_like_pdb_id("ABCD") is False  # must start with a digit
    assert looks_like_pdb_id("1AB") is False  # wrong length


def test_fetch_pdb_uses_cache(tmp_path):
    cached = tmp_path / "1LYZ.pdb"
    cached.write_text("HEADER  fake cached entry\n", encoding="utf-8")

    # Cache hit must not touch the network at all.
    result = fetch_pdb("1lyz", cache_dir=tmp_path)
    assert result == cached


def test_fetch_pdb_invalid_id_raises(tmp_path):
    try:
        fetch_pdb("9ZZZ", cache_dir=tmp_path)
    except ValueError as error:
        assert "9ZZZ" in str(error)
    except (urllib.error.URLError, OSError):
        pytest.skip("no network access in this environment")
