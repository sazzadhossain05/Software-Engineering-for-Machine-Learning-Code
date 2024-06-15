"""NumPy from-scratch MLP for California Housing regression.

Architecture: Input(8) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1).
The simplest architecture in our benchmark suite, where NumPy implementations
remain competitive with framework-based ones in both performance and code size.

SE observation: Even for this simple model, the NumPy version requires explicit
gradient computation and manual training loop management — exposing the full
complexity that model.fit() abstracts in Keras.
"""

from __future__ import annotations

import numpy as np

from src.numpy_impl.layers import Layer, Linear, ReLU


class NumpyMLPRegression:
    """3-layer MLP for tabular regression.

    Input: (batch, n_features) float32 standardized features.
    Output: (batch, 1) predicted values.
    """

    def __init__(self, n_features: int = 8, hidden_sizes: list[int] = None, seed: int = 42):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.layers: list[Layer] = [
            Linear(n_features, hidden_sizes[0], seed=seed),
            ReLU(),
            Linear(hidden_sizes[0], hidden_sizes[1], seed=seed + 1),
            ReLU(),
            Linear(hidden_sizes[1], 1, seed=seed + 2),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> None:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> list[dict]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute MSE loss and gradient.

    Args:
        predictions: Model output, shape (batch, 1).
        targets: Ground truth, shape (batch,) or (batch, 1).

    Returns:
        Tuple of (scalar loss, gradient w.r.t. predictions).
    """
    targets = targets.reshape(-1, 1)
    batch_size = predictions.shape[0]
    diff = predictions - targets
    loss = float((diff ** 2).mean())
    grad = 2.0 * diff / batch_size
    return loss, grad
