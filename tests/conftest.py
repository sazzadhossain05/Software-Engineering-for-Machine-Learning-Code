"""Shared test fixtures and utilities for the test suite."""

import numpy as np
import pytest


@pytest.fixture
def seed():
    """Standard seed for reproducible tests."""
    return 42


@pytest.fixture
def small_batch_cifar10():
    """Small synthetic CIFAR-10-like batch for fast unit tests."""
    rng = np.random.RandomState(42)
    x = rng.rand(8, 32, 32, 3).astype(np.float32)
    y = rng.randint(0, 10, size=8).astype(np.int64)
    return x, y


@pytest.fixture
def small_batch_sentiment():
    """Small synthetic sentiment batch."""
    rng = np.random.RandomState(42)
    x = rng.randint(1, 5000, size=(8, 64)).astype(np.int32)
    y = rng.randint(0, 2, size=8).astype(np.int64)
    return x, y


@pytest.fixture
def small_batch_regression():
    """Small synthetic regression batch."""
    rng = np.random.RandomState(42)
    x = rng.rand(16, 8).astype(np.float32)
    y = rng.rand(16).astype(np.float32) * 5
    return x, y
