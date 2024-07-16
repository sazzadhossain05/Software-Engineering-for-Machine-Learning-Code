"""PyTorch CNN for CIFAR-10 image classification.

Architecture mirrors the NumPy version exactly: 3 conv blocks + 2 FC layers.
The dramatic reduction in code (~30 lines vs ~120) is itself a key SE finding.

SE observation: PyTorch's nn.Module provides automatic parameter registration,
gradient tracking, and device management — all manually coded in NumPy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PyTorchCNNCifar10(nn.Module):
    """Simple CNN for CIFAR-10, equivalent to the NumPy implementation."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (batch, 3, 32, 32). Output: (batch, 10) logits."""
        x = self.features(x)
        x = self.classifier(x)
        return x
