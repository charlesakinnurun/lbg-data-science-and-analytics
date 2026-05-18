"""
preprocessing.py
----------------
All data-cleaning and aggregation steps needed before feature engineering:

1. Aggregate ``Transaction_History``   (many rows per customer → one row)
2. Aggregate ``Customer_Service``      (many rows per customer → one row)
3. Derive ``DaysSinceLastLogin`` from  ``Online_Activity``
4. Merge all five sheets into a single master DataFrame
5. Handle missing service records (332 customers with no interactions)
6. Detect and clip outliers in ``AvgSpend``
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – Transaction aggregation
# ---------------------------------------------------------------------------

def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-transaction rows into one summary row per customer.

    The original sheet has ~5 transactions per customer.  We aggregate to:

    * ``TotalSpend``       – sum of ``AmountSpent``
    * ``AvgSpend``         – mean of ``AmountSpent``
    * ``NumTransactions``  – count of transaction IDs
    * ``FavCategory``      – modal ``ProductCategory``
    * ``DaysSinceLastTx``  – days from the most recent transaction to the
                             latest transaction date in the dataset (reference)

    Args:
        transactions: Raw ``Transaction_History`` DataFrame.

    Returns:
        Aggregated DataFrame with one row per ``CustomerID``.
    """
    logger.info("Aggregating transactions (%d rows → per-customer).", len(transactions))

    transactions = transactions.copy()
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    reference_date = transactions["TransactionDate"].max()

    aggregated = (
        transactions.groupby("CustomerID")
        .agg(
            TotalSpend=("AmountSpent", "sum"),
            AvgSpend=("AmountSpent", "mean"),
            NumTransactions=("TransactionID", "count"),
            FavCategory=("ProductCategory", lambda x: x.mode()[0]),
            DaysSinceLastTx=(
                "TransactionDate",
                lambda x: (reference_date - x.max()).days,
            ),
        )
        .reset_index()
    )

    logger.info("Transaction aggregation done: %d customer rows.", len(aggregated))
    return aggregated


# ---------------------------------------------------------------------------
# Step 2 – Customer service aggregation
# ---------------------------------------------------------------------------

def aggregate_service(service: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-interaction rows into one summary row per customer.

    Produces:

    * ``NumInteractions`` – total interactions
    * ``NumComplaints``   – interactions of type ``'Complaint'``
    * ``NumUnresolved``   – interactions with ``ResolutionStatus == 'Unresolved'``

    Args:
        service: Raw ``Customer_Service`` DataFrame.

    Returns:
        Aggregated DataFrame with one row per ``CustomerID`` that has at
        least one service record.  Customers with no record are handled
        later during the merge/fill step.
    """
    logger.info("Aggregating customer service records (%d rows).", len(service))

    aggregated = (
        service.groupby("CustomerID")
        .agg(
            NumInteractions=("InteractionID", "count"),
            NumComplaints=("InteractionType", lambda x: (x == "Complaint").sum()),
            NumUnresolved=("ResolutionStatus", lambda x: (x == "Unresolved").sum()),
        )
        .reset_index()
    )

    logger.info("Service aggregation done: %d customer rows.", len(aggregated))
    return aggregated


# ---------------------------------------------------------------------------
# Step 3 – Online activity feature
# ---------------------------------------------------------------------------

def engineer_online_features(
    online: pd.DataFrame,
    reference_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Add ``DaysSinceLastLogin`` to the online-activity sheet.

    Args:
        online:         Raw ``Online_Activity`` DataFrame.
        reference_date: ISO date string used as the "today" anchor.

    Returns:
        Copy of ``online`` with an extra ``DaysSinceLastLogin`` column.
    """
    online = online.copy()
    ref = pd.Timestamp(reference_date)
    online["LastLoginDate"] = pd.to_datetime(online["LastLoginDate"])
    online["DaysSinceLastLogin"] = (ref - online["LastLoginDate"]).dt.days
    return online


# ---------------------------------------------------------------------------
# Step 4 – Master merge
# ---------------------------------------------------------------------------

def merge_all(
    demographics: pd.DataFrame,
    transactions_agg: pd.DataFrame,
    service_agg: pd.DataFrame,
    online: pd.DataFrame,
    churn: pd.DataFrame,
    age_bins: list[int] | None = None,
    age_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Left-join all five sheets on ``CustomerID`` and add ``AgeGroup``.

    Left joins preserve every customer in demographics even when no
    matching service or transaction record exists.

    Args:
        demographics:    Cleaned demographics sheet (1 row per customer).
        transactions_agg: Aggregated transactions (1 row per customer).
        service_agg:     Aggregated service interactions (1 row per customer).
        online:          Online activity sheet with ``DaysSinceLastLogin``.
        churn:           Churn status sheet.
        age_bins:        Bin edges for ``pd.cut``; defaults to ``[17,35,55,70]``.
        age_labels:      Labels for each bin; defaults to ``['Young','Middle','Senior']``.

    Returns:
        Merged master DataFrame.
    """
    age_bins = age_bins or [17, 35, 55, 70]
    age_labels = age_labels or ["Young", "Middle", "Senior"]

    online_cols = [
        "CustomerID",
        "DaysSinceLastLogin",
        "LoginFrequency",
        "LastLoginDate",
        "ServiceUsage",
    ]

    logger.info("Merging all sheets on CustomerID.")
    df = (
        demographics
        .merge(transactions_agg, on="CustomerID", how="left")
        .merge(service_agg, on="CustomerID", how="left")
        .merge(online[online_cols], on="CustomerID", how="left")
        .merge(churn, on="CustomerID", how="left")
    )

    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    logger.info("Merge complete: %s rows × %s cols.", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 5 – Fill missing service records
# ---------------------------------------------------------------------------

def fill_missing_service(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN service columns for customers with no service record.

    Adds a binary ``HasServiceRecord`` flag, then fills
    ``NumInteractions``, ``NumComplaints``, and ``NumUnresolved`` with 0
    for customers who never contacted support.

    Args:
        df: Master merged DataFrame (output of :func:`merge_all`).

    Returns:
        DataFrame with service NaNs replaced and flag column added.
    """
    df = df.copy()
    df["HasServiceRecord"] = (df["NumInteractions"] > 0).fillna(False).astype(int)

    service_cols = ["NumInteractions", "NumComplaints", "NumUnresolved"]
    missing_before = df[service_cols].isnull().sum().sum()
    df[service_cols] = df[service_cols].fillna(0)

    logger.info(
        "Filled %d missing service cells with 0. Customers with no record: %d.",
        missing_before,
        (df["HasServiceRecord"] == 0).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 6 – Outlier detection & clipping
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, numeric_cols: list[str], iqr_factor: float = 1.5) -> None:
    """Log an IQR-based outlier report (does not modify the DataFrame).

    Args:
        df:           DataFrame to inspect.
        numeric_cols: Column names to check.
        iqr_factor:   Multiplier applied to IQR; standard is 1.5.
    """
    logger.info("=== Outlier Report (IQR × %.1f) ===", iqr_factor)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        n_low = (df[col] < lower).sum()
        n_high = (df[col] > upper).sum()
        status = "✓" if (n_low + n_high) == 0 else "!"
        logger.info(
            "%s %s | range=[%.1f–%.1f] IQR=[Q1=%.1f, Q3=%.1f] "
            "outliers: %d low, %d high",
            status, col, df[col].min(), df[col].max(), q1, q3, n_low, n_high,
        )


def clip_avg_spend(
    df: pd.DataFrame,
    low_pct: float = 0.05,
    high_pct: float = 0.95,
) -> pd.DataFrame:
    """Winsorise ``AvgSpend`` at the given percentiles.

    Creates a new column ``AvgSpend_clean`` instead of modifying the
    original, preserving the raw values for audit purposes.

    Args:
        df:       DataFrame containing an ``AvgSpend`` column.
        low_pct:  Lower percentile clip bound (default 5 %).
        high_pct: Upper percentile clip bound (default 95 %).

    Returns:
        DataFrame with ``AvgSpend_clean`` column added.
    """
    df = df.copy()
    low = df["AvgSpend"].quantile(low_pct)
    high = df["AvgSpend"].quantile(high_pct)
    df["AvgSpend_clean"] = df["AvgSpend"].clip(lower=low, upper=high)
    logger.info(
        "AvgSpend clipped: raw=[%.1f–%.1f] → clean=[%.1f–%.1f].",
        df["AvgSpend"].min(), df["AvgSpend"].max(),
        df["AvgSpend_clean"].min(), df["AvgSpend_clean"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_preprocessing(
    sheets: dict[str, pd.DataFrame],
    config_path: str = "configs/config.yaml",
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline in one call.

    Args:
        sheets:      Dict returned by :func:`src.data_loader.load_sheets`.
        config_path: Path to YAML config.

    Returns:
        Cleaned and merged master DataFrame ready for feature engineering.
    """
    cfg = load_config(config_path)
    pre_cfg = cfg["preprocessing"]
    out_cfg = cfg["outlier_check"]

    transactions_agg = aggregate_transactions(sheets["transactions"])
    service_agg = aggregate_service(sheets["service"])
    online = engineer_online_features(
        sheets["online"], reference_date=pre_cfg["reference_date"]
    )

    df = merge_all(
        demographics=sheets["demographics"],
        transactions_agg=transactions_agg,
        service_agg=service_agg,
        online=online,
        churn=sheets["churn"],
        age_bins=pre_cfg["age_bins"],
        age_labels=pre_cfg["age_labels"],
    )

    df = fill_missing_service(df)

    detect_outliers(df, numeric_cols=out_cfg["numeric_cols"], iqr_factor=out_cfg["iqr_factor"])
    df = clip_avg_spend(
        df,
        low_pct=pre_cfg["avg_spend_clip_low"],
        high_pct=pre_cfg["avg_spend_clip_high"],
    )

    return df
