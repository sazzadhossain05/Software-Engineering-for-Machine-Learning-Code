"""Integration test: NumPy end-to-end pipeline on tiny data."""

import numpy as np
import pytest

from src.numpy_impl.models.mlp_regression import NumpyMLPRegression, mse_loss
from src.numpy_impl.optimizers import Adam


class TestNumpyPipeline:
    def test_mlp_trains_and_loss_decreases(self):
        """Verify the NumPy MLP can overfit a tiny dataset."""
        np.random.seed(42)
        x = np.random.rand(32, 8).astype(np.float32)
        y = np.random.rand(32).astype(np.float32)

        model = NumpyMLPRegression(n_features=8, seed=42)
        opt = Adam(model.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for epoch in range(50):
            out = model.forward(x)
            loss, grad = mse_loss(out, y)
            model.backward(grad)
            opt.parameters = model.parameters()
            opt.step()
            opt.zero_grad()

            if initial_loss is None:
                initial_loss = loss
            final_loss = loss

        assert final_loss < initial_loss, "Loss should decrease during training"
