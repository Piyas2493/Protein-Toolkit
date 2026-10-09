"""
ProteinToolkit Random Forest Trainer
"""

import pandas as pd

from .splits import cv_splitter, held_out_split

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix
    )

except ImportError as exc:
    RandomForestClassifier = None
    cross_val_score = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None

    sklearn_import_error = exc

class ModelTrainer:

    def __init__(
        self,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42
    ):

        if RandomForestClassifier is None:
            raise ImportError(
                "scikit-learn is required for ModelTrainer. "
                "Install scikit-learn to use this module."
            )

        # Unbounded depth/leaf size on hundreds of (often sparse
        # dipeptide) features overfits badly and produces multi-
        # gigabyte pickles at tens of thousands of rows — depth cap
        # alone wasn't enough (tree count x unconstrained leaves was
        # the bigger driver), so min_samples_leaf and a lower default
        # n_estimators do the rest.
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, csv_file):

        if cross_val_score is None or accuracy_score is None:
            raise ImportError(
                "scikit-learn is required to train the model. "
                "Install scikit-learn to use training functionality."
            )

        dataframe = pd.read_csv(csv_file)

        X = dataframe.drop(
            columns=["ID", "Label"]
        )

        y = dataframe["Label"]

        train_idx, test_idx, groups = held_out_split(dataframe)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups[train_idx] if groups is not None else None

        self.model.fit(
            X_train,
            y_train
        )

        # ==============================
        # 5-FOLD CROSS-VALIDATION
        # ==============================

        cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train,
            cv=cv_splitter(g_train),
            groups=g_train,
            scoring="accuracy",
            n_jobs=-1
        )

        print("\n" + "=" * 50)
        print("5-FOLD CROSS-VALIDATION")
        print("=" * 50)

        print("Scores:", cv_scores)
        print(f"Mean accuracy: {cv_scores.mean():.4f}")
        print(f"Std deviation: {cv_scores.std():.4f}")

        # ==============================
        # TEST SET PREDICTION
        # ==============================

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)

        print(f"\nValidation accuracy: {accuracy:.4f}")

        # ==============================
        # CLASSIFICATION REPORT
        # ==============================

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        # ==============================
        # CONFUSION MATRIX
        # ==============================

        print("\nConfusion Matrix:\n")

        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        # ==============================
        # FEATURE IMPORTANCE
        # ==============================

        print("\nFeature Importance:\n")

        importance = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(
            ascending=False
        )

        print(importance)

        return {
            "model": self.model,
            "accuracy": accuracy,
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "feature_importance": importance
        }

    def save(self, output):

        if joblib is None:
            raise ImportError(
                "joblib is required to save the model. "
                "Install joblib or use a different export path."
            )

        joblib.dump(
            self.model,
            output
        )