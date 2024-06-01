"""Framework-agnostic evaluation metrics.

Metric computation should be decoupled from framework-specific code to
ensure consistent evaluation across NumPy, PyTorch, and TensorFlow
implementations. This module accepts NumPy arrays and returns scalar
values, enabling fair cross-framework comparison.

All metrics here are standard and well-defined. We deliberately avoid
framework-specific metric APIs (e.g., torchmetrics, tf.keras.metrics) to
ensure identical computation logic across all implementations.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels, shape (n_samples,).
        y_pred: Predicted labels (not probabilities), shape (n_samples,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> float:
    """Compute F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy ('micro', 'macro', 'weighted', 'binary').

    Returns:
        F1 score as a float in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error for regression tasks.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a non-negative float.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination) for regression.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R² score (can be negative for poor models).
    """
    return float(r2_score(y_true, y_pred))


def compute_task_metrics(
    task: str, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute all relevant metrics for a given task type.

    This is the primary interface for evaluation. It dispatches to the
    appropriate metric functions based on task type.

    Args:
        task: One of 'cifar10', 'sentiment', 'regression'.
        y_true: Ground truth values.
        y_pred: Predicted values (labels for classification, values for regression).

    Returns:
        Dictionary mapping metric names to scalar values.

    Raises:
        ValueError: If task is not recognized.
    """
    if task in ("cifar10", "sentiment"):
        return {
            "accuracy": compute_accuracy(y_true, y_pred),
            "f1_weighted": compute_f1(y_true, y_pred, average="weighted"),
            "f1_macro": compute_f1(y_true, y_pred, average="macro"),
        }
    elif task == "regression":
        return {
            "rmse": compute_rmse(y_true, y_pred),
            "r2": compute_r2(y_true, y_pred),
        }
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'cifar10', 'sentiment', or 'regression'.")
