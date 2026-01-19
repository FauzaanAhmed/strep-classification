"""Utility functions for training and evaluation."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)

    # Binary predictions
    y_pred_binary = (y_proba >= 0.5).astype(int)

    # Other metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # Specificity (TN / (TN + FP))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
    }


def aggregate_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across folds with mean and std."""
    keys = list(metrics_list[0].keys())
    aggregated = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return aggregated


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
