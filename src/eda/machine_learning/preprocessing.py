import logging
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def prepare_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple:
    """
    Performs stratified train-test split.
    """
    logger.info("Splitting data into train and test sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE to handle class imbalance in the training set.
    """
    logger.info("Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res