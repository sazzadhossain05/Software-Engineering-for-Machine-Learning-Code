"""TensorFlow/Keras LSTM for IMDb sentiment analysis."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def create_lstm_sentiment(
    vocab_size: int = 10000, embedding_dim: int = 128, lstm_hidden: int = 128
) -> keras.Model:
    """Create LSTM sentiment classifier matching NumPy/PyTorch architecture."""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        layers.LSTM(lstm_hidden),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model
