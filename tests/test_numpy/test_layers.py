"""Tests for NumPy from-scratch layer implementations."""

import numpy as np
import pytest

from src.numpy_impl.layers import Embedding, Flatten, Linear, ReLU, Sigmoid


class TestLinear:
    def test_output_shape(self):
        layer = Linear(8, 16, seed=42)
        x = np.random.rand(4, 8).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 16)

    def test_backward_produces_gradients(self):
        layer = Linear(8, 16, seed=42)
        x = np.random.rand(4, 8).astype(np.float32)
        layer.forward(x)
        grad = np.ones((4, 16), dtype=np.float32)
        dx = layer.backward(grad)
        assert dx.shape == (4, 8)
        assert layer.grad_weight is not None
        assert layer.grad_bias is not None

    def test_parameters_count(self):
        layer = Linear(8, 16)
        params = layer.parameters()
        assert len(params) == 2  # weight + bias


class TestReLU:
    def test_positive_pass_through(self):
        layer = ReLU()
        x = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
        out = layer.forward(x)
        np.testing.assert_array_equal(out, [[1.0, 0.0, 0.5]])

    def test_gradient_mask(self):
        layer = ReLU()
        x = np.array([[1.0, -1.0, 0.5]], dtype=np.float32)
        layer.forward(x)
        grad = np.ones_like(x)
        dx = layer.backward(grad)
        np.testing.assert_array_equal(dx, [[1.0, 0.0, 1.0]])


class TestSigmoid:
    def test_output_range(self):
        layer = Sigmoid()
        x = np.array([[-100, 0, 100]], dtype=np.float32)
        out = layer.forward(x)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_zero_gives_half(self):
        layer = Sigmoid()
        out = layer.forward(np.array([[0.0]], dtype=np.float32))
        assert abs(out[0, 0] - 0.5) < 1e-6


class TestFlatten:
    def test_flatten_shape(self):
        layer = Flatten()
        x = np.random.rand(4, 3, 8, 8).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 192)


class TestEmbedding:
    def test_output_shape(self):
        layer = Embedding(100, 32, seed=42)
        x = np.array([[1, 5, 10], [2, 3, 4]], dtype=np.int32)
        out = layer.forward(x)
        assert out.shape == (2, 3, 32)
