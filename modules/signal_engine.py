"""Machine learning signal generation engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Container for model training output."""

    model_name: str
    model: object
    accuracy: float
    confusion: list[list[int]]
    X_test: pd.DataFrame
    y_test: pd.Series


class SignalEngine:
    """Train, evaluate, and infer signals using classification models."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self, model_name: str, random_state: int = 42):
        """Create model instance by name."""

        model_name = model_name.lower()
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=12,
                min_samples_leaf=6,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        if model_name == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            )
        raise ValueError(f"Unsupported model_name: {model_name}")

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        features: list[str],
        model_name: str = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainResult:
        """Train model with 80/20 split and compute evaluation metrics."""

        model = self._build_model(model_name=model_name, random_state=random_state)
        X = df[features].copy()
        y = df["Target"].astype(int).copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=[-1, 0, 1]).tolist()
        LOGGER.info("Model=%s accuracy=%.4f", model_name, acc)
        LOGGER.info("Confusion matrix labels=[SELL(-1), HOLD(0), BUY(1)]: %s", cm)

        return TrainResult(
            model_name=model_name,
            model=model,
            accuracy=acc,
            confusion=cm,
            X_test=X_test,
            y_test=y_test,
        )

    def save_model(self, model: object, filename: str) -> Path:
        """Persist trained model with joblib."""

        path = self.model_dir / filename
        joblib.dump(model, path)
        LOGGER.info("Saved model to %s", path)
        return path

    @staticmethod
    def generate_signals(df: pd.DataFrame, model: object, features: list[str]) -> pd.DataFrame:
        """Generate model predictions for all rows and return enriched dataframe."""

        data = df.copy()
        data["Signal"] = model.predict(data[features])
        return data

    def train_models(
        self,
        df: pd.DataFrame,
        features: list[str],
        random_state: int = 42,
    ) -> Tuple[TrainResult, Dict[str, TrainResult]]:
        """Train both required models and return best-by-accuracy plus all results."""

        names = ["random_forest", "gradient_boosting"]
        results: Dict[str, TrainResult] = {}
        for name in names:
            result = self.train_and_evaluate(
                df=df,
                features=features,
                model_name=name,
                test_size=0.2,
                random_state=random_state,
            )
            results[name] = result

        best = max(results.values(), key=lambda x: x.accuracy)
        LOGGER.info("Selected best model: %s (accuracy=%.4f)", best.model_name, best.accuracy)
        return best, results

