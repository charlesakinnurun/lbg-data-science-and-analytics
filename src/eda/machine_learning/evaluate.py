import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve)
import shap
import logging

logger = logging.getLogger(__name__)

def evaluate_performance(model, X_test, y_test, target_names=['Retained', 'Churned']):
    """
    Prints classification report and AUC-ROC.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.3f}")
    return y_prob

def plot_visuals(y_test, y_prob, model, X_test):
    """
    Generates ROC, PR curves, and SHAP summary.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label='Model')
    axes[0].set_title('ROC Curve')
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, color='green')
    axes[1].set_title('Precision-Recall Curve')
    
    plt.tight_layout()
    plt.savefig('evaluation_curves.png')
    
    # SHAP
    logger.info("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Handle SHAP dimensionality for classification
    sv = shap_values[:, :, 1] if shap_values.ndim == 3 else (shap_values[1] if isinstance(shap_values, list) else shap_values)
    
    plt.figure()
    shap.summary_plot(sv, X_test, plot_type='bar', show=False)
    plt.savefig('shap_summary.png')