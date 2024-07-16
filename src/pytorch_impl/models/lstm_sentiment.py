"""PyTorch LSTM for IMDb sentiment analysis.

SE observation: nn.LSTM encapsulates the entire LSTM cell computation
(4 gates, state management) that required ~60 lines of manual NumPy code
into a single constructor call. This is the clearest example of how
framework abstraction reduces code complexity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PyTorchLSTMSentiment(nn.Module):
    """LSTM-based binary sentiment classifier."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        lstm_hidden: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: (batch, seq_len) long. Output: (batch,) probs."""
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.fc(hidden.squeeze(0))
        return torch.sigmoid(logits).squeeze(-1)
