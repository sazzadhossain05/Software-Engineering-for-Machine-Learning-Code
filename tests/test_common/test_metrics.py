"""Tests for framework-agnostic evaluation metrics."""

import numpy as np
import pytest

from src.common.metrics import (
    compute_accuracy,
    compute_f1,
    compute_r2,
    compute_rmse,
    compute_task_metrics,
)


class TestClassificationMetrics:
    def test_accuracy_perfect(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_accuracy_half(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert compute_accuracy(y_true, y_pred) == 0.5

    def test_f1_perfect(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert compute_f1(y_true, y_pred, average="weighted") == 1.0

    def test_f1_zero_division(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        result = compute_f1(y_true, y_pred, average="weighted")
        assert isinstance(result, float)


class TestRegressionMetrics:
    def test_rmse_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1 / 3)
        assert abs(compute_rmse(y_true, y_pred) - expected) < 1e-6

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_r2(y, y) == 1.0

    def test_r2_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])
        assert compute_r2(y_true, y_pred) < 0


class TestTaskMetrics:
    def test_cifar10_task(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        result = compute_task_metrics("cifar10", y_true, y_pred)
        assert "accuracy" in result
        assert "f1_weighted" in result

    def test_regression_task(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 1.9])
        result = compute_task_metrics("regression", y_true, y_pred)
        assert "rmse" in result
        assert "r2" in result

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            compute_task_metrics("unknown", np.array([0]), np.array([0]))
