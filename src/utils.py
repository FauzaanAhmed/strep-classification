"""
Utility functions for training, evaluation, and configuration management.

This module provides:
- set_seed: Reproducibility utilities
- FocalLoss: Class imbalance handling
- compute_metrics: Classification metrics computation
- aggregate_metrics: Cross-validation metrics aggregation
- save_results: JSON serialization
- load_config: YAML configuration loading
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Ensures deterministic behavior for:
    - Python's random module
    - NumPy random number generator
    - PyTorch random number generators (CPU and CUDA)
    - cuDNN deterministic operations
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in binary classification.
    
    Focal loss downweights easy examples and focuses training on hard examples:
    FL = α * (1 - p_t)^γ * BCE
    
    Where:
    - α: Weighting factor for rare class (default 0.25)
    - γ: Focusing parameter (default 2.0)
    - p_t: Model's predicted probability for true class
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model logits [B, 1]
            targets: Ground truth labels [B, 1]
            
        Returns:
            Scalar focal loss value
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)  # Predicted probability for true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (unused, computed from y_proba)
        y_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary with metrics: auc, accuracy, precision, recall_sensitivity,
        specificity, f1
    """
    from sklearn.metrics import (
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
    )

    # Compute ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)

    # Binary predictions using 0.5 threshold
    y_pred_binary = (y_proba >= 0.5).astype(int)

    # Standard classification metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # Specificity: True Negative Rate (TN / (TN + FP))
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
    """
    Aggregate metrics across cross-validation folds.
    
    Computes mean and standard deviation for each metric across all folds.
    
    Args:
        metrics_list: List of metric dictionaries, one per fold
        
    Returns:
        Dictionary with same keys as input, values are dicts with 'mean' and 'std'
    """
    if not metrics_list:
        return {}
    
    keys = list(metrics_list[0].keys())
    aggregated = {}

    for key in keys:
        values = [metrics[key] for metrics in metrics_list]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return aggregated


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results dictionary to JSON file.
    
    Creates output directory if it doesn't exist.
    
    Args:
        results: Results dictionary to save
        output_path: Path to output JSON file
    """
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
