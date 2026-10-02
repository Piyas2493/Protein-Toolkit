from pathlib import Path

import pytest

from ml.dataset_builder import DatasetBuilder


@pytest.fixture
def example_fasta(tmp_path: Path) -> str:
    p = tmp_path / "example.fasta"
    p.write_text(
        ">toy\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAN\n",
        encoding="utf-8",
    )
    return str(p)


def test_dataset_builder(example_fasta: str) -> None:
    df = DatasetBuilder().build(example_fasta, label="TestProtein")
    assert "Label" in df.columns
    assert "Length" in df.columns
    assert len(df) > 0
    assert (df["Label"] == "TestProtein").all()
