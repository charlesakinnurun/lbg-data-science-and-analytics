"""
train.py
--------
Training pipeline: loads data, preprocesses, engineers features, trains a
classifier, and saves the model + scaler to ``models/``.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_loader import load_sheets
from src.evaluate import evaluate_model
from src.feature_engineering import run_feature_engineering
from src.model import build_model, save_model, split_features_target
from src.preprocessing import run_preprocessing
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def train(config_path: str | Path = "configs/config.yaml") -> None:
    """Execute the full training pipeline.

    Steps
    -----
    1. Load Excel sheets
    2. Preprocess & merge
    3. Feature engineering (encoding + scaling)
    4. Train / test split
    5. Fit classifier
    6. Evaluate on hold-out set
    7. Save model and scaler

    Args:
        config_path: Path to YAML configuration file.
    """
    cfg = load_config(config_path)

    # ---- 1. Load ----------------------------------------------------------
    sheets = load_sheets(config_path)

    # ---- 2. Preprocess ----------------------------------------------------
    df_clean = run_preprocessing(sheets, config_path=str(config_path))

    # ---- 3. Feature engineering -------------------------------------------
    df_final, scaler = run_feature_engineering(df_clean, config_path=str(config_path))

    # ---- 4. Export model-ready data ---------------------------------------
    output_path = Path(cfg["data"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(output_path, index=False)
    logger.info("Exported model-ready data: %s (%s rows).", output_path, len(df_final))

    # ---- 5. Train / test split --------------------------------------------
    X, y = split_features_target(df_final)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        "Train/test split: %d train, %d test rows.", len(X_train), len(X_test)
    )

    # ---- 6. Fit -----------------------------------------------------------
    model = build_model()
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    # ---- 7. Evaluate ------------------------------------------------------
    evaluate_model(model, X_test, y_test)

    # ---- 8. Save ----------------------------------------------------------
    models_dir = Path("models")
    save_model(model, models_dir / "churn_model.pkl")
    save_model(scaler, models_dir / "scaler.pkl")
    logger.info("Artefacts saved to '%s/'.", models_dir)
