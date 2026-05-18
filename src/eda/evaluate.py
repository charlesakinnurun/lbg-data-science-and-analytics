"""
evaluate.py
-----------
Model evaluation utilities: classification report, confusion matrix,
and feature importance logging.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Print and return evaluation metrics for a fitted classifier.

    Args:
        model:  Any fitted sklearn-compatible classifier.
        X_test: Feature matrix for the hold-out set.
        y_test: True labels for the hold-out set.

    Returns:
        Dict with keys ``'report'`` (classification report dict) and
        ``'confusion_matrix'`` (2-D list).
    """
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    logger.info(
        "Evaluation – Accuracy: %.3f | Precision (churn): %.3f | Recall (churn): %.3f",
        report["accuracy"],
        report.get("1", {}).get("precision", float("nan")),
        report.get("1", {}).get("recall", float("nan")),
    )

    return {"report": report, "confusion_matrix": cm}


def log_feature_importance(model, feature_names: list[str], top_n: int = 15) -> None:
    """Log the top-N most important features for a linear classifier.

    Works for any estimator with a ``coef_`` attribute
    (LogisticRegression, LinearSVC, etc.).

    Args:
        model:         Fitted linear classifier.
        feature_names: Ordered list of feature names matching the training X.
        top_n:         Number of top features to display.
    """
    if not hasattr(model, "coef_"):
        logger.warning("Model does not expose 'coef_'; skipping importance logging.")
        return

    importance = pd.Series(
        abs(model.coef_[0]), index=feature_names
    ).sort_values(ascending=False)

    print(f"\n=== Top {top_n} Features by |Coefficient| ===")
    print(importance.head(top_n).to_string())
    logger.info("Feature importance logged for top %d features.", top_n)
