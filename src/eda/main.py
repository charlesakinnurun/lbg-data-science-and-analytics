"""
main.py
-------
Command-line entry point for the customer-churn project.

Usage
-----
Run the full training pipeline::

    python main.py

Run only exploratory data analysis::

    python main.py --eda

Use a custom config::

    python main.py --config configs/my_config.yaml
"""

from __future__ import annotations

import argparse
import sys

from data_loader import load_sheets
from eda import (
    churn_correlations,
    login_frequency_by_churn,
    plot_correlation_heatmap,
    plot_distributions,
    print_statistics,
)
from preprocessing import run_preprocessing
from train import train
from utils import get_logger, load_config

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Churn Pipeline")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run EDA only (no model training).",
    )
    return parser.parse_args(argv)


def run_eda(config_path: str) -> None:
    """Load, preprocess, and generate EDA outputs."""
    sheets = load_sheets(config_path)
    df = run_preprocessing(sheets, config_path=config_path)

    print_statistics(df)
    plot_distributions(df, save_path="data/plots/distributions.png")
    plot_correlation_heatmap(df, save_path="data/plots/correlation_heatmap.png")

    corr = churn_correlations(df)
    print("\n=== Correlations with ChurnStatus ===")
    print(corr.to_string())

    login_stats = login_frequency_by_churn(df)
    print(
        f"\nAvg logins – churned: {login_stats['churned']:.1f}, "
        f"retained: {login_stats['retained']:.1f}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger.info("Starting pipeline with config: %s", args.config)

    if args.eda:
        logger.info("EDA mode selected.")
        run_eda(args.config)
    else:
        logger.info("Training mode selected.")
        train(config_path=args.config)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
