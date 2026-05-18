import logging
import jobpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import xgboost as xgb

logger = logging.getLogger(__name__)

def run_hyperparameter_search(X_train, y_train, n_iter, cv, random_state):
    """
    Finds the best hyperparameters for Random Forest.
    """
    logger.info("Starting Hyperparameter Search...")
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20)
    }
    
    rand_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced'),
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1
    )
    rand_search.fit(X_train, y_train)
    logger.info(f"Best Params found: {rand_search.best_params_}")
    return rand_search.best_estimator_

def train_best_rf(X_train, y_train, params, random_state):
    """
    Trains the Random Forest model with specific parameters.
    """
    logger.info("Training Best Random Forest model...")
    model = RandomForestClassifier(**params, random_state=random_state)
    model.fit(X_train, y_train)
    return model