"""Tests for NumPy model architectures — shape and forward pass validation."""

import numpy as np
import pytest

from src.numpy_impl.models.cnn_cifar10 import NumpyCNNCifar10, cross_entropy_loss
from src.numpy_impl.models.mlp_regression import NumpyMLPRegression, mse_loss
from src.numpy_impl.models.lstm_sentiment import NumpyLSTMSentiment, binary_cross_entropy_loss


class TestCNNCifar10:
    def test_output_shape(self):
        model = NumpyCNNCifar10(seed=42)
        x = np.random.rand(4, 3, 32, 32).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (4, 10)

    def test_cross_entropy_loss(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.array([0, 3, 5, 9])
        loss, grad = cross_entropy_loss(logits, targets)
        assert isinstance(loss, float) and loss > 0
        assert grad.shape == logits.shape

    def test_has_trainable_parameters(self):
        model = NumpyCNNCifar10(seed=42)
        assert len(model.parameters()) > 0


class TestMLPRegression:
    def test_output_shape(self):
        model = NumpyMLPRegression(n_features=8, seed=42)
        x = np.random.rand(4, 8).astype(np.float32)
        out = model.forward(x)
        assert out.shape == (4, 1)

    def test_mse_loss(self):
        preds = np.array([[2.0], [3.0]], dtype=np.float32)
        targets = np.array([1.0, 4.0])
        loss, grad = mse_loss(preds, targets)
        assert loss == pytest.approx(1.0, abs=1e-5)
        assert grad.shape == preds.shape


class TestLSTMSentiment:
    def test_output_shape(self):
        model = NumpyLSTMSentiment(vocab_size=100, embedding_dim=16, lstm_hidden=16, seed=42)
        x = np.random.randint(0, 100, size=(4, 32)).astype(np.int32)
        out = model.forward(x)
        assert out.shape == (4, 1)

    def test_output_range(self):
        model = NumpyLSTMSentiment(vocab_size=100, embedding_dim=16, lstm_hidden=16, seed=42)
        x = np.random.randint(0, 100, size=(4, 32)).astype(np.int32)
        out = model.forward(x)
        assert np.all(out >= 0) and np.all(out <= 1)
