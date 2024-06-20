"""NumPy from-scratch CNN for CIFAR-10 image classification.

Architecture: 3 convolutional blocks (Conv2D -> ReLU -> MaxPool) -> Flatten -> FC -> ReLU -> FC.
This is deliberately simple to remain tractable in pure NumPy while exercising
all core deep learning operations that frameworks abstract away.

SE observation: This file requires ~120 lines to define what PyTorch does in ~30
and Keras in ~15. The verbosity is a direct, measurable SE quality trade-off.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.numpy_impl.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Layer


class NumpyCNNCifar10:
    """Simple CNN for CIFAR-10 classification, built entirely in NumPy.

    Input: (batch, 3, 32, 32) float32 images normalized to [0, 1].
    Output: (batch, 10) logits (unnormalized class scores).
    """

    def __init__(self, seed: int = 42):
        self.layers: List[Layer] = [
            # Block 1: 3 -> 16 channels
            Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1, seed=seed),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 32x32 -> 16x16
            # Block 2: 16 -> 32 channels
            Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1, seed=seed + 1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 16x16 -> 8x8
            # Block 3: 32 -> 64 channels
            Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1, seed=seed + 2),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 8x8 -> 4x4
            # Classifier
            Flatten(),
            Linear(in_features=64 * 4 * 4, out_features=128, seed=seed + 3),
            ReLU(),
            Linear(in_features=128, out_features=10, seed=seed + 4),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers.

        Args:
            x: Input images, shape (batch, 3, 32, 32). Note: NumPy impl
               uses NCHW format to match PyTorch convention.

        Returns:
            Logits of shape (batch, 10).
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> None:
        """Backward pass through all layers in reverse order.

        Args:
            grad_output: Gradient of loss w.r.t. output logits.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> list[dict]:
        """Collect all trainable parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute cross-entropy loss and its gradient w.r.t. logits.

    Args:
        logits: Raw model output, shape (batch, num_classes).
        targets: Integer class labels, shape (batch,).

    Returns:
        Tuple of (scalar loss, gradient w.r.t. logits).
    """
    batch_size = logits.shape[0]

    # Numerically stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Cross-entropy loss
    correct_log_probs = -np.log(probs[np.arange(batch_size), targets] + 1e-12)
    loss = correct_log_probs.mean()

    # Gradient: softmax output - one-hot targets
    grad = probs.copy()
    grad[np.arange(batch_size), targets] -= 1.0
    grad /= batch_size

    return float(loss), grad
