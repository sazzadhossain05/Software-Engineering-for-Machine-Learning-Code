"""Tests for TensorFlow/Keras model architectures."""

import pytest

try:
    import numpy as np
    import tensorflow as tf
    from src.tensorflow_impl.models.cnn_cifar10 import create_cnn_cifar10
    from src.tensorflow_impl.models.lstm_sentiment import create_lstm_sentiment
    from src.tensorflow_impl.models.mlp_regression import create_mlp_regression
    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")


class TestTFCNN:
    def test_output_shape(self):
        model = create_cnn_cifar10()
        x = np.random.rand(4, 32, 32, 3).astype(np.float32)
        out = model(x, training=False)
        assert out.shape == (4, 10)


class TestTFLSTM:
    def test_output_shape(self):
        model = create_lstm_sentiment(vocab_size=100, embedding_dim=16, lstm_hidden=16)
        x = np.random.randint(0, 100, (4, 32)).astype(np.int32)
        out = model(x, training=False)
        assert out.shape == (4, 1)


class TestTFMLP:
    def test_output_shape(self):
        model = create_mlp_regression(n_features=8)
        x = np.random.rand(4, 8).astype(np.float32)
        out = model(x, training=False)
        assert out.shape == (4, 1)
