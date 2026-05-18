"""
model.py
--------
Model definition and persistence utilities.

Provides a thin wrapper so that swapping the underlying estimator
(e.g. from LogisticRegression to XGBoost) requires changes in only
one place.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils import get_logger

logger = get_logger(__name__)

# Default model and its hyper-parameters
_DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",   # important for imbalanced churn datasets
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(params: dict[str, Any] | None = None) -> LogisticRegression:
    """Instantiate the churn classifier.

    Args:
        params: Hyper-parameters passed to :class:`~sklearn.linear_model.LogisticRegression`.
                Merged on top of ``_DEFAULT_MODEL_PARAMS``; caller values win.

    Returns:
        Untrained :class:`~sklearn.linear_model.LogisticRegression` instance.
    """
    merged = {**_DEFAULT_MODEL_PARAMS, **(params or {})}
    model = LogisticRegression(**merged)
    logger.info("Model built: %s(%s).", type(model).__name__, merged)
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str | Path) -> None:
    """Persist a fitted model (or scaler) to disk with pickle.

    Args:
        model: Any picklable object (e.g. fitted sklearn estimator).
        path:  Destination file path; parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("Saved: %s", path)


def load_model(path: str | Path) -> Any:
    """Load a pickled model from disk.

    Args:
        path: Path to the pickled file.

    Returns:
        Deserialised object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded: %s", path)
    return obj


# ---------------------------------------------------------------------------
# Feature/target split
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    target_col: str = "ChurnStatus",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a model-ready DataFrame into X and y.

    Args:
        df:         Model-ready DataFrame (output of feature engineering).
        target_col: Name of the target column.

    Returns:
        Tuple ``(X, y)`` where X is a DataFrame of features and y is a Series.

    Raises:
        KeyError: If ``target_col`` is not present in ``df``.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info("Split complete – X: %s, y: %s.", X.shape, y.shape)
    return X, y
