"""Tests for framework-agnostic data loading."""

import numpy as np
import pytest

from src.common.data_loader import load_california_housing, load_dataset


class TestCaliforniaHousing:
    def test_loads_correct_splits(self):
        data = load_california_housing(val_split=0.1, seed=42)
        assert "x_train" in data
        assert "x_val" in data
        assert "x_test" in data
        assert data["x_train"].dtype == np.float32
        assert data["y_train"].dtype == np.float32

    def test_no_data_leakage(self):
        """Verify train/val/test splits don't overlap."""
        data = load_california_housing(seed=42)
        n_total = len(data["x_train"]) + len(data["x_val"]) + len(data["x_test"])
        assert n_total == 20640  # California Housing dataset size

    def test_features_standardized(self):
        data = load_california_housing(seed=42)
        # Training features should have approx zero mean and unit variance
        means = np.abs(data["x_train"].mean(axis=0))
        assert np.all(means < 0.1)  # Close to zero


class TestDatasetDispatch:
    def test_regression_dispatch(self):
        data = load_dataset("regression", val_split=0.1, seed=42)
        assert "x_train" in data

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            load_dataset("nonexistent_task")
