"""Tests for PyTorch model architectures."""

import pytest

try:
    import torch
    from src.pytorch_impl.models.cnn_cifar10 import PyTorchCNNCifar10
    from src.pytorch_impl.models.lstm_sentiment import PyTorchLSTMSentiment
    from src.pytorch_impl.models.mlp_regression import PyTorchMLPRegression
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestPyTorchCNN:
    def test_output_shape(self):
        model = PyTorchCNNCifar10()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_gradient_flow(self):
        model = PyTorchCNNCifar10()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestPyTorchLSTM:
    def test_output_shape(self):
        model = PyTorchLSTMSentiment(vocab_size=100, embedding_dim=16, lstm_hidden=16)
        x = torch.randint(0, 100, (4, 32))
        out = model(x)
        assert out.shape == (4,)

    def test_output_range(self):
        model = PyTorchLSTMSentiment(vocab_size=100)
        x = torch.randint(0, 100, (4, 32))
        out = model(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)


class TestPyTorchMLP:
    def test_output_shape(self):
        model = PyTorchMLPRegression(n_features=8)
        x = torch.randn(4, 8)
        out = model(x)
        assert out.shape == (4,)
