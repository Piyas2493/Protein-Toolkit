from ml.feature_extractor import FeatureExtractor


def test_feature_extractor():

    sequence = (
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAN"
    )

    extractor = FeatureExtractor()

    features = extractor.extract(sequence)

    assert "Length" in features
    assert "MolecularWeight" in features
    assert "AA_A" in features
    assert features["Length"] == len(sequence)