"""
data_loader.py
--------------
Loads the customer-churn Excel workbook and returns each sheet as a
named DataFrame.  All path and sheet-name configuration is read from
``configs/config.yaml`` so no values are hardcoded here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sheets(config_path: str | Path = "configs/config.yaml") -> dict[str, pd.DataFrame]:
    """Load every configured sheet from the Excel workbook.

    Reads the workbook once with ``sheet_name=None`` (returns all sheets)
    and then selects only the sheets declared in the config.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Mapping of logical name → DataFrame, e.g.::

            {
                "demographics": pd.DataFrame(...),
                "transactions": pd.DataFrame(...),
                "service":      pd.DataFrame(...),
                "online":       pd.DataFrame(...),
                "churn":        pd.DataFrame(...),
            }

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        KeyError: If an expected sheet is absent from the workbook.
    """
    cfg = load_config(config_path)
    input_path = Path(cfg["data"]["input_path"])
    sheet_map: dict[str, str] = cfg["data"]["sheets"]  # logical → workbook name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {input_path}. "
            "Place the Excel workbook in the 'data/' directory."
        )

    logger.info("Reading workbook: %s", input_path)
    all_sheets: dict[str, pd.DataFrame] = pd.read_excel(input_path, sheet_name=None)

    logger.info("Available sheets: %s", list(all_sheets.keys()))

    sheets: dict[str, pd.DataFrame] = {}
    for logical_name, workbook_name in sheet_map.items():
        if workbook_name not in all_sheets:
            raise KeyError(
                f"Sheet '{workbook_name}' not found in workbook. "
                f"Available: {list(all_sheets.keys())}"
            )
        sheets[logical_name] = all_sheets[workbook_name]
        logger.info("Loaded '%s' (%s rows)", logical_name, len(sheets[logical_name]))

    return sheets
