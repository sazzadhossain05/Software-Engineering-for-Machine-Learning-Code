"""PyTorch MLP for California Housing regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class PyTorchMLPRegression(nn.Module):
    """3-layer MLP for tabular regression."""

    def __init__(self, n_features: int = 8, hidden_sizes: list[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
