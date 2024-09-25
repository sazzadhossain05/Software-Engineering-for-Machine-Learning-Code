"""Integration test: PyTorch end-to-end pipeline on tiny data."""

import pytest

try:
    import torch
    import torch.nn as nn
    from src.pytorch_impl.models.mlp_regression import PyTorchMLPRegression
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestPyTorchPipeline:
    def test_mlp_trains_and_loss_decreases(self):
        torch.manual_seed(42)
        model = PyTorchMLPRegression(n_features=8)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x = torch.randn(32, 8)
        y = torch.randn(32)

        initial_loss = None
        final_loss = None

        for _ in range(50):
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss
