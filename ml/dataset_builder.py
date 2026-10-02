"""
ProteinToolkit Dataset Builder
"""

import pandas as pd

from core.fasta import FASTAParser
from ml.feature_extractor import FeatureExtractor


class DatasetBuilder:

    def __init__(self):

        self.extractor = FeatureExtractor()

    def build(self, fasta_file, label):

        records = FASTAParser.load(fasta_file)

        sequences = [
            {"id": r.header, "sequence": r.sequence}
            for r in records
        ]

        rows = []

        for item in sequences:

            features = self.extractor.extract(
                item["sequence"]
            )

            features["ID"] = item["id"]

            features["Label"] = label

            rows.append(features)

        return pd.DataFrame(rows)

    def save(self, dataframe, output):

        dataframe.to_csv(
            output,
            index=False
        )