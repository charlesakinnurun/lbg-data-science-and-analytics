import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the churn dataset from an Excel file.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataframe into features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y