"""From-scratch optimizers in pure NumPy.

Implementing optimizers from scratch highlights the significant engineering
effort that frameworks abstract away. The Adam optimizer alone requires
maintaining two momentum buffers per parameter, bias correction, and
epsilon-safe division — all of which are single-line calls in
PyTorch/TensorFlow but ~40 lines of careful NumPy code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SGD:
    """Stochastic Gradient Descent with optional momentum.

    Args:
        parameters: List of {'param': ndarray, 'grad': ndarray} dicts.
        lr: Learning rate.
        momentum: Momentum factor (0 for vanilla SGD).
        weight_decay: L2 regularization factor.
    """

    def __init__(
        self,
        parameters: list[dict],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocities: list[Optional[np.ndarray]] = [None] * len(parameters)

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, p in enumerate(self.parameters):
            if p["grad"] is None:
                continue

            grad = p["grad"].copy()

            # L2 regularization
            if self.weight_decay > 0:
                grad += self.weight_decay * p["param"]

            # Momentum
            if self.momentum > 0:
                if self._velocities[i] is None:
                    self._velocities[i] = np.zeros_like(grad)
                self._velocities[i] = self.momentum * self._velocities[i] + grad
                update = self._velocities[i]
            else:
                update = grad

            p["param"] -= self.lr * update

    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        for p in self.parameters:
            p["grad"] = None


class Adam:
    """Adam optimizer (Kingma & Ba, 2015).

    Args:
        parameters: List of {'param': ndarray, 'grad': ndarray} dicts.
        lr: Learning rate.
        beta1: Exponential decay rate for first moment estimates.
        beta2: Exponential decay rate for second moment estimates.
        eps: Small constant for numerical stability.
        weight_decay: L2 regularization factor.
    """

    def __init__(
        self,
        parameters: list[dict],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # First and second moment estimates
        self._m: list[np.ndarray] = [np.zeros_like(p["param"]) for p in parameters]
        self._v: list[np.ndarray] = [np.zeros_like(p["param"]) for p in parameters]
        self._t: int = 0

    def step(self) -> None:
        """Perform a single optimization step with bias correction."""
        self._t += 1

        for i, p in enumerate(self.parameters):
            if p["grad"] is None:
                continue

            grad = p["grad"].copy()

            if self.weight_decay > 0:
                grad += self.weight_decay * p["param"]

            # Update biased first moment estimate
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias-corrected estimates
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)

            # Parameter update
            p["param"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        for p in self.parameters:
            p["grad"] = None


def create_optimizer(
    name: str, parameters: list[dict], lr: float = 0.001, **kwargs
) -> SGD | Adam:
    """Factory function to create an optimizer by name.

    Args:
        name: 'sgd' or 'adam'.
        parameters: Parameter dicts from model layers.
        lr: Learning rate.

    Returns:
        Optimizer instance.
    """
    optimizers = {"sgd": SGD, "adam": Adam}
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    return optimizers[name.lower()](parameters, lr=lr, **kwargs)
