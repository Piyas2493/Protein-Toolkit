from pathlib import Path

import pandas as pd
import pytest

from ml.feature_extractor import FeatureExtractor
from ml.trainer import ModelTrainer


# Reusable sequences so each generated row has the same schema.
_SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAN",
    "MKKAVINGEQIRSISDLHQTLKKELALPEYYGEN",
    "GPRPLPIFRTPRWPLAGGGGGSHHHHHH",
    "ACDGEFGHIKLMNPQRSTVWY",
    "MGEKRAQKILQKILQKILQKILQK",
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGE",
    "MAGGEAPRPPQAPAPAPAPAPAPAPAAAPPA",
    "MTMDKSELVQKAKLAEQAERYDDMAAAMKSV",
    "MASLGTPGGSPSPDFTSAAAERLGYVYQGT",
    "MAKKKKKKKMVKKKKKKKKKKKKKK",
    "MDDDDDRRRRRRLLLLLVVVVV",
    "MHHHHHHHDDDDDEEEE",
]


@pytest.fixture
def example_dataset(tmp_path: Path) -> str:
    """Build a tiny balanced classification CSV from FeatureExtractor output.

    The trainer uses stratify=y, so we need at least 2 samples per class.
    """
    rows = []
    for i, seq in enumerate(_SEQS):
        feats = FeatureExtractor().extract(seq)
        feats["ID"] = f"toy{i}"
        feats["Label"] = i % 2  # perfectly balanced 6/6
        rows.append(feats)
    df = pd.DataFrame(rows)
    p = tmp_path / "example_dataset.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_trainer(example_dataset: str) -> None:
    result = ModelTrainer().train(example_dataset)
    assert "accuracy" in result
    assert result["accuracy"] >= 0
    assert hasattr(result["model"], "predict")
