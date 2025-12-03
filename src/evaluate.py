from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,  
    recall_score,     
    r2_score,
)

import torch

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R^2 for regression."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    # Some older versions of scikit-learn do not support the 'squared' keyword.
    # We therefore compute RMSE manually from the MSE.
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1-score (weighted), and QWK for classification."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    
    # <--- Added Precision and Recall (weighted for imbalance)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    
    f1 = f1_score(y_true, y_pred, average="weighted")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return {
        "accuracy": acc, 
        "precision_weighted": precision, # <--- Return this
        "recall_weighted": recall,       # <--- Return this
        "f1_weighted": f1, 
        "qwk": qwk
    }


def confusion_matrix_values(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the confusion matrix for further visualization."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return confusion_matrix(y_true, y_pred)


def gender_bias_regression(
    y_true: np.ndarray, y_pred: np.ndarray, genders: np.ndarray
) -> Dict[str, float]:
    """Compute gender-wise MAE and their difference.

    Parameters
    ----------
    y_true : array-like
        True ages in years.
    y_pred : array-like
        Predicted ages in years.
    genders : array-like
        Binary gender indicator (1 = male, 0 = female).
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    genders = np.asarray(genders, dtype=int)

    male_mask = genders == 1
    female_mask = genders == 0

    mae_male = (
        mean_absolute_error(y_true[male_mask], y_pred[male_mask]) if male_mask.any() else float("nan")
    )
    mae_female = (
        mean_absolute_error(y_true[female_mask], y_pred[female_mask]) if female_mask.any() else float("nan")
    )

    bias_diff = (
        mae_male - mae_female if not (np.isnan(mae_male) or np.isnan(mae_female)) else float("nan")
    )

    return {
        "mae_male": mae_male,
        "mae_female": mae_female,
        "mae_diff_male_minus_female": bias_diff,
    }


def gender_bias_classification(
    y_true: np.ndarray, y_pred: np.ndarray, genders: np.ndarray
) -> Dict[str, float]:
    """Compute gender-wise accuracy for the classification task."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    genders = np.asarray(genders, dtype=int)

    male_mask = genders == 1
    female_mask = genders == 0

    acc_male = accuracy_score(y_true[male_mask], y_pred[male_mask]) if male_mask.any() else float("nan")
    acc_female = (
        accuracy_score(y_true[female_mask], y_pred[female_mask]) if female_mask.any() else float("nan")
    )
    bias_diff = (
        acc_male - acc_female if not (np.isnan(acc_male) or np.isnan(acc_female)) else float("nan")
    )

    return {
        "acc_male": acc_male,
        "acc_female": acc_female,
        "acc_diff_male_minus_female": bias_diff,
    }

def compare_approaches(model, dataloader, device):
    """
    Compare two approaches for classification:
    1. Direct Classification Head (Learned)
    2. Regression Head -> Discretized (Threshold based)
    """
    model.eval()
    true_stages = []
    
    # Approach 1: Direct predictions from the Classification Head
    preds_cls_head = []
    
    # Approach 2: Predictions derived from the Regression Head
    preds_reg_head = []

    print("\n--- Running Comparison of Approaches ---")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            # We don't need 'age_years' for prediction, but we need 'stage' for ground truth
            stages = batch['stage'].cpu().numpy()
            
            # Forward pass
            age_pred, stage_logits = model(images)
            
            # --- Approach 1: Direct Classification ---
            # Get the class with the highest score
            _, cls_preds = torch.max(stage_logits, 1)
            preds_cls_head.extend(cls_preds.cpu().numpy())
            
            # --- Approach 2: Regression -> Discretization ---
            # Convert continuous age to stage buckets
            ages = age_pred.cpu().numpy().flatten()
            for age in ages:
                if age < 12:
                    pred_stage = 0 # Child
                elif age <= 18:
                    pred_stage = 1 # Adolescent
                else:
                    pred_stage = 2 # Adult
                preds_reg_head.extend([pred_stage])
            
            true_stages.extend(stages)

    # Calculate Accuracies
    acc_cls = accuracy_score(true_stages, preds_cls_head)
    acc_reg = accuracy_score(true_stages, preds_reg_head)

    print(f"{'Metric':<30} | {'Direct Classification':<20} | {'Regression Thresholding':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<30} | {acc_cls:.4f}{'':<14} | {acc_reg:.4f}")
    
    return {
        "acc_direct_classification": acc_cls,
        "acc_regression_threshold": acc_reg
    }