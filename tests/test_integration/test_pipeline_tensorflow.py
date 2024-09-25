"""Integration test: TensorFlow end-to-end pipeline on tiny data."""

import pytest

try:
    import numpy as np
    import tensorflow as tf
    from src.tensorflow_impl.models.mlp_regression import create_mlp_regression
    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")


class TestTFPipeline:
    def test_mlp_trains_and_loss_decreases(self):
        tf.random.set_seed(42)
        model = create_mlp_regression(n_features=8)
        model.compile(optimizer="adam", loss="mse")

        x = np.random.rand(32, 8).astype(np.float32)
        y = np.random.rand(32).astype(np.float32)

        history = model.fit(x, y, epochs=50, verbose=0)
        assert history.history["loss"][-1] < history.history["loss"][0]
