"""TensorFlow/Keras MLP for California Housing regression."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def create_mlp_regression(n_features: int = 8, hidden_sizes: list[int] = None) -> keras.Model:
    """Create MLP for tabular regression matching NumPy/PyTorch architecture."""
    if hidden_sizes is None:
        hidden_sizes = [64, 32]
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(hidden_sizes[0], activation="relu"),
        layers.Dense(hidden_sizes[1], activation="relu"),
        layers.Dense(1),
    ])
    return model
