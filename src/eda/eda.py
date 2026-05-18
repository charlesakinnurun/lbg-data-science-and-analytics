"""
eda.py
------
Exploratory data analysis: descriptive statistics, visualisations, and
correlation analysis.  All functions accept a DataFrame and write plots
to disk (or display them inline) rather than embedding magic in notebooks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def print_statistics(df: pd.DataFrame) -> None:
    """Print numeric and categorical summary statistics to stdout.

    Args:
        df: Master merged DataFrame.
    """
    print("=== Numeric Statistics ===")
    print(df.describe().to_string())
    print("\n=== Categorical Statistics ===")
    print(df.describe(include="object").to_string())

    cat_cols = ["Gender", "MaritalStatus", "IncomeLevel", "ServiceUsage", "FavCategory"]
    print("\n=== Categorical Value Counts ===")
    for col in cat_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts().to_string())


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    """Plot histograms and churn-rate bar charts.

    Creates a 2 × 3 grid:

    * Row 1 – histograms: Age, TotalSpend, LoginFrequency
    * Row 2 – Churn Rate by AgeGroup, TotalSpend boxplot by churn,
               Churn Rate by IncomeLevel

    Args:
        df:        Master merged DataFrame (must contain ``ChurnStatus``).
        save_path: If provided, saves the figure to this path instead of
                   calling ``plt.show()``.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # --- Row 1: distributions ---
    df["Age"].hist(ax=axes[0, 0], edgecolor="white")
    axes[0, 0].set_title("Age Distribution")

    df["TotalSpend"].hist(ax=axes[0, 1], color="#1D9E75", edgecolor="white")
    axes[0, 1].set_title("Total Spend Distribution")

    df["LoginFrequency"].hist(ax=axes[0, 2], color="#7F77DD", edgecolor="white")
    axes[0, 2].set_title("Login Frequency Distribution")

    # --- Row 2: churn insights ---
    df.groupby("AgeGroup")["ChurnStatus"].mean().plot(
        kind="bar", ax=axes[1, 0], color="#E24B4A"
    )
    axes[1, 0].set_title("Churn Rate by Age Group")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

    df.boxplot(column="TotalSpend", by="ChurnStatus", ax=axes[1, 1])
    axes[1, 1].set_title("Total Spend: Churned vs Retained")

    df.groupby("IncomeLevel")["ChurnStatus"].mean().plot(
        kind="bar", ax=axes[1, 2], color="#BA7517"
    )
    axes[1, 2].set_title("Churn Rate by Income Level")
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    """Plot a full numeric correlation heatmap.

    Args:
        df:        DataFrame (numeric columns only are used).
        save_path: Optional path to save the figure.
    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def churn_correlations(df: pd.DataFrame) -> pd.Series:
    """Return Pearson correlations of all numeric columns with ``ChurnStatus``.

    Args:
        df: DataFrame containing a ``ChurnStatus`` column.

    Returns:
        Series of correlation coefficients, sorted by absolute value.
    """
    corr = df.corrwith(df["ChurnStatus"], numeric_only=True).sort_values(
        key=abs, ascending=False
    )
    logger.info("Correlations with ChurnStatus computed (%d features).", len(corr))
    return corr


def login_frequency_by_churn(df: pd.DataFrame) -> dict[str, float]:
    """Compare mean ``LoginFrequency`` between churned and retained customers.

    Args:
        df: DataFrame with ``ChurnStatus`` and ``LoginFrequency`` columns.

    Returns:
        Dict with keys ``'churned'`` and ``'retained'``.
    """
    churned = df[df["ChurnStatus"] == 1]["LoginFrequency"].mean()
    retained = df[df["ChurnStatus"] == 0]["LoginFrequency"].mean()
    logger.info(
        "Avg LoginFrequency – churned: %.1f | retained: %.1f", churned, retained
    )
    return {"churned": churned, "retained": retained}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    else:
        plt.show()
    plt.close(fig)
