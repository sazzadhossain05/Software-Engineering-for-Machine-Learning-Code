"""Tests for reproducibility utilities."""

import numpy as np

from src.common.reproducibility import get_seed_info, set_global_seed


class TestSeedSetting:
    def test_numpy_deterministic(self):
        set_global_seed(42)
        a = np.random.rand(10)
        set_global_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_global_seed(42)
        a = np.random.rand(10)
        set_global_seed(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_seed_info_returns_dict(self):
        info = get_seed_info()
        assert isinstance(info, dict)
        assert "numpy_available" in info
