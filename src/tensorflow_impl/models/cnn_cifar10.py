"""TensorFlow/Keras CNN for CIFAR-10.

SE observation: Keras Sequential API reduces the CNN definition to ~10 lines.
This is the most compact implementation across all three frameworks, but
customization beyond standard patterns requires switching to the functional
API or subclassing — a well-documented SE friction point.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_cifar10() -> keras.Model:
    """Create CNN for CIFAR-10 matching the NumPy/PyTorch architecture."""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2),
        # Block 2
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        # Block 3
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ])
    return model
