from core.catalytic import CatalyticPredictor


def test_single_residue_does_not_satisfy_duplicate_requirement():
    # One ASP and one HIS must not satisfy rules needing two of a kind
    # (Aspartic Protease needs 2x ASP, Metalloprotease needs 2x HIS).
    predictor = CatalyticPredictor()

    hits = predictor.predict("AAAAADAAAAAHAAAAA")

    names = {hit["name"] for hit in hits}

    assert "Aspartic Protease" not in names
    assert "Metalloprotease" not in names


def test_duplicate_residues_satisfy_requirement():
    predictor = CatalyticPredictor()

    hits = predictor.predict("AAADAAADAAA")

    names = {hit["name"] for hit in hits}

    assert "Aspartic Protease" in names
