"""Tests for PyTorch training utilities."""

import pytest

try:
    import torch
    from src.pytorch_impl.datasets import numpy_to_tensor_dataset, create_dataloaders
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestDatasets:
    def test_cifar10_tensor_dataset(self):
        x = np.random.rand(16, 32, 32, 3).astype(np.float32)
        y = np.random.randint(0, 10, 16).astype(np.int64)
        ds = numpy_to_tensor_dataset(x, y, task="cifar10")
        assert len(ds) == 16
        x_t, y_t = ds[0]
        assert x_t.shape == (3, 32, 32)  # NCHW

    def test_sentiment_tensor_dataset(self):
        x = np.random.randint(0, 100, (16, 64)).astype(np.int32)
        y = np.random.randint(0, 2, 16).astype(np.int64)
        ds = numpy_to_tensor_dataset(x, y, task="sentiment")
        x_t, y_t = ds[0]
        assert x_t.dtype == torch.long

    def test_create_dataloaders(self):
        data = {
            "x_train": np.random.rand(32, 8).astype(np.float32),
            "y_train": np.random.rand(32).astype(np.float32),
            "x_val": np.random.rand(8, 8).astype(np.float32),
            "y_val": np.random.rand(8).astype(np.float32),
            "x_test": np.random.rand(8, 8).astype(np.float32),
            "y_test": np.random.rand(8).astype(np.float32),
        }
        loaders = create_dataloaders(data, "regression", batch_size=16)
        assert "train" in loaders and "val" in loaders and "test" in loaders
