"""
feature_engineering.py
-----------------------
Encoding, derived feature creation, and numeric scaling.

Pipeline
--------
1. Binary encoding   – Gender
2. Ordinal encoding  – IncomeLevel  (Low < Medium < High)
3. One-hot encoding  – MaritalStatus, ServiceUsage, FavCategory, AgeGroup
4. Derived features  – SpendPerTx, ComplaintRate, IsHighRisk
5. Standard scaling  – all numeric columns except the target (ChurnStatus)
6. Drop raw / helper columns that are no longer needed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, load_config

logger = get_logger(__name__)

# Columns to drop after encoding (raw string columns + helpers)
_DROP_COLS = [
    "CustomerID",
    "Gender",
    "MaritalStatus",
    "IncomeLevel",
    "ServiceUsage",
    "FavCategory",
    "AgeGroup",
    "LastLoginDate",
]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_binary(df: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    """Apply a binary label mapping to a categorical column.

    Args:
        df:      Input DataFrame (not modified in place).
        col:     Source column name.
        mapping: Dict mapping raw values to integer codes, e.g. ``{'M':1,'F':0}``.

    Returns:
        DataFrame with a new column named ``{col}_enc``.
    """
    df = df.copy()
    enc_col = f"{col}_enc"
    df[enc_col] = df[col].map(mapping)
    logger.debug("Binary-encoded '%s' → '%s'.", col, enc_col)
    return df


def encode_ordinal(df: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    """Apply an ordinal integer mapping to a categorical column.

    Args:
        df:      Input DataFrame.
        col:     Source column name.
        mapping: Ordered mapping, e.g. ``{'Low':0,'Medium':1,'High':2}``.

    Returns:
        DataFrame with a new column named ``{col}_enc``.
    """
    df = df.copy()
    enc_col = f"{col}_enc"
    df[enc_col] = df[col].map(mapping)
    logger.debug("Ordinal-encoded '%s' → '%s'.", col, enc_col)
    return df


def encode_onehot(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """One-hot encode a categorical column (first dummy dropped).

    Args:
        df:     Input DataFrame.
        col:    Column to encode.
        prefix: Prefix for the generated dummy column names.

    Returns:
        DataFrame with one-hot columns appended and the source column
        *still present* (caller decides when to drop it).
    """
    dummies = pd.get_dummies(df[col], prefix=prefix, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    logger.debug("One-hot-encoded '%s' → %d dummy columns.", col, len(dummies.columns))
    return df


def encode_age_group(df: pd.DataFrame, age_bins: list[int], age_codes: list[int]) -> pd.DataFrame:
    """Encode ``AgeGroup`` as an ordinal integer via ``pd.cut``.

    Args:
        df:        Input DataFrame containing an ``Age`` column.
        age_bins:  Bin edges, e.g. ``[17, 35, 55, 70]``.
        age_codes: Integer codes for each bin, e.g. ``[0, 1, 2]``.

    Returns:
        DataFrame with ``AgeGroup_enc`` column added.
    """
    df = df.copy()
    df["AgeGroup_enc"] = pd.cut(df["Age"], bins=age_bins, labels=age_codes).astype(int)
    logger.debug("Age group encoded with bins %s → codes %s.", age_bins, age_codes)
    return df


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def create_derived_features(df: pd.DataFrame, high_risk_login_threshold: int = 10) -> pd.DataFrame:
    """Create business-logic-derived features.

    * ``SpendPerTx``    – average spend per individual transaction
    * ``ComplaintRate`` – fraction of interactions that are complaints
    * ``IsHighRisk``    – flag for low-engagement + complaint history

    Args:
        df:                          Input DataFrame (post-encoding).
        high_risk_login_threshold:   ``LoginFrequency`` below this value
                                     counts as low engagement.

    Returns:
        DataFrame with three new columns appended.
    """
    df = df.copy()
    df["SpendPerTx"] = df["TotalSpend"] / df["NumTransactions"]
    df["ComplaintRate"] = (
        df["NumComplaints"] / df["NumInteractions"].replace(0, np.nan)
    ).fillna(0)
    df["IsHighRisk"] = (
        (df["LoginFrequency"] < high_risk_login_threshold) & (df["NumComplaints"] > 0)
    ).astype(int)

    logger.info("Derived features added: SpendPerTx, ComplaintRate, IsHighRisk.")
    return df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_numeric_features(
    df: pd.DataFrame,
    target_col: str = "ChurnStatus",
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale all numeric columns except the target.

    Args:
        df:         Input DataFrame.
        target_col: Name of the target column to exclude from scaling.

    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler).
        The scaler can be persisted and reused for inference.
    """
    df = df.copy()
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info("Scaled %d numeric columns.", len(numeric_cols))
    return df, scaler


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_feature_engineering(
    df: pd.DataFrame,
    config_path: str = "configs/config.yaml",
) -> tuple[pd.DataFrame, StandardScaler]:
    """Execute the complete encoding + feature + scaling pipeline.

    Args:
        df:          Master DataFrame from :func:`src.preprocessing.run_preprocessing`.
        config_path: Path to YAML config.

    Returns:
        Tuple of (model-ready DataFrame, fitted StandardScaler).
    """
    cfg = load_config(config_path)
    enc_cfg = cfg["encoding"]
    pre_cfg = cfg["preprocessing"]
    feat_cfg = cfg["features"]

    # 1. Binary encoding
    df = encode_binary(df, "Gender", enc_cfg["gender_map"])

    # 2. Ordinal encoding
    df = encode_ordinal(df, "IncomeLevel", enc_cfg["income_map"])

    # 3. One-hot encoding
    df = encode_onehot(df, "MaritalStatus", "MS")
    df = encode_onehot(df, "ServiceUsage", "SU")
    df = encode_onehot(df, "FavCategory", "FC")

    # 4. Age group encoding
    df = encode_age_group(df, pre_cfg["age_bins"], pre_cfg["age_label_codes"])

    # 5. Remove duplicate columns (safety guard)
    df = df.loc[:, ~df.columns.duplicated()]

    # 6. Drop raw columns
    drop_existing = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)

    # 7. Convert any boolean dummies to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 8. Derived features
    df = create_derived_features(df, feat_cfg["high_risk_login_threshold"])

    # 9. Standard scaling
    df, scaler = scale_numeric_features(df, target_col="ChurnStatus")

    logger.info("Feature engineering complete. Final shape: %s.", df.shape)
    return df, scaler
