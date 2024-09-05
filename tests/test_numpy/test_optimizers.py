"""Tests for NumPy from-scratch optimizers."""

import numpy as np

from src.numpy_impl.optimizers import SGD, Adam, create_optimizer


class TestSGD:
    def test_parameter_update(self):
        param = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        params = [{"param": param, "grad": grad}]
        opt = SGD(params, lr=1.0)
        opt.step()
        np.testing.assert_allclose(param, [0.9, 1.9, 2.9])

    def test_zero_grad(self):
        params = [{"param": np.zeros(3), "grad": np.ones(3)}]
        opt = SGD(params, lr=0.1)
        opt.zero_grad()
        assert params[0]["grad"] is None


class TestAdam:
    def test_converges_on_quadratic(self):
        """Adam should minimize f(x) = x^2 starting from x=5."""
        x = np.array([5.0], dtype=np.float32)
        params = [{"param": x, "grad": None}]
        opt = Adam(params, lr=0.1)
        for _ in range(200):
            params[0]["grad"] = 2.0 * params[0]["param"]
            opt.step()
        assert abs(params[0]["param"][0]) < 0.1


class TestCreateOptimizer:
    def test_create_sgd(self):
        params = [{"param": np.zeros(3), "grad": None}]
        opt = create_optimizer("sgd", params, lr=0.01)
        assert isinstance(opt, SGD)

    def test_create_adam(self):
        params = [{"param": np.zeros(3), "grad": None}]
        opt = create_optimizer("adam", params, lr=0.001)
        assert isinstance(opt, Adam)
