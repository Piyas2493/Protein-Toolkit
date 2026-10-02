"""
ProteinToolkit Machine Learning Model Loader
"""

from pathlib import Path
import joblib


class ModelLoader:

    @staticmethod
    def load(model_path):
        """
        Load a trained machine learning model.
        """

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        return joblib.load(model_path)