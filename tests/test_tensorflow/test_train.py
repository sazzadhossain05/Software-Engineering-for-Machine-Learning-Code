"""Tests for TensorFlow dataset utilities."""

import pytest

try:
    import numpy as np
    import tensorflow as tf
    from src.tensorflow_impl.datasets import create_tf_datasets
    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")


class TestTFDatasets:
    def test_create_datasets(self):
        data = {
            "x_train": np.random.rand(32, 8).astype(np.float32),
            "y_train": np.random.rand(32).astype(np.float32),
            "x_val": np.random.rand(8, 8).astype(np.float32),
            "y_val": np.random.rand(8).astype(np.float32),
            "x_test": np.random.rand(8, 8).astype(np.float32),
            "y_test": np.random.rand(8).astype(np.float32),
        }
        datasets = create_tf_datasets(data, "regression", batch_size=16)
        assert "train" in datasets
        for batch in datasets["train"].take(1):
            x, y = batch
            assert x.shape[1] == 8
