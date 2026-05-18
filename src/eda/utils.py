"""
utils.py
--------
Shared utilities: logging setup, config loading, and DataFrame inspection.
"""

import logging
import yaml
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO", fmt: str | None = None) -> logging.Logger:
    """Create (or retrieve) a named logger with a stream handler.

    Args:
        name:  Logger name, typically ``__name__`` of the calling module.
        level: Logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
        fmt:   Optional format string; falls back to a sensible default.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:          # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        fmt = fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# DataFrame inspection
# ---------------------------------------------------------------------------

def overview(df: pd.DataFrame, name: str = "") -> None:
    """Print a concise structural summary of a DataFrame.

    Displays shape, dtypes, first five rows, missing-value counts, and
    unique-value counts for every column.

    Args:
        df:   DataFrame to inspect.
        name: Optional label printed in the header line.
    """
    header = f"=== Overview: {name} ===" if name else "=== Overview ==="
    print(header)
    print(f"Shape:  {df.shape}")
    print("\nDtypes:\n", df.dtypes.to_string())
    print("\nFirst 5 rows:\n", df.head().to_string())
    print("\nMissing values:\n", df.isnull().sum().to_string())
    print("\nUnique values:\n", df.nunique().to_string())
    print()
