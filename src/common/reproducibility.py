"""Reproducibility utilities for deterministic ML experiments.

Reproducibility is a well-known challenge in ML research. Non-deterministic
behavior arises from random weight initialization, data shuffling, GPU
floating-point operations, and framework-specific sources. This module
centralizes seed management to ensure experiments are as deterministic
as possible.

Note: even with identical seeds, GPU operations may introduce small
numerical differences due to non-associative floating-point arithmetic
in parallel reductions. This module documents these limitations rather
than hiding them.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """Set random seeds across all relevant libraries for reproducibility.

    This function sets seeds for Python's random module, NumPy, and optionally
    PyTorch and TensorFlow if they are available. It also configures
    deterministic behavior for CUDA operations where supported.

    Args:
        seed: Integer seed value. Default 42 (convention in ML research).
        deterministic_cuda: If True, attempt to enable deterministic CUDA
            operations. Note: this may reduce performance by 10-20%.

    Note:
        Even with all seeds set and deterministic mode enabled, some GPU
        operations (e.g., atomicAdd in scatter/gather) may produce slightly
        different results across runs. This is a known limitation documented
        in both PyTorch and TensorFlow.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable used by some frameworks
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (conditional import — may not be installed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic_cuda:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # TensorFlow (conditional import — may not be installed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic_cuda:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


def get_seed_info() -> dict:
    """Return a dictionary describing the current seed configuration.

    Useful for logging seed state to MLflow or experiment records.

    Returns:
        Dictionary with framework availability and seed configuration status.
    """
    info = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not set"),
        "numpy_available": True,
    }

    try:
        import torch

        info["pytorch_available"] = True
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    except ImportError:
        info["pytorch_available"] = False

    try:
        import tensorflow as tf

        info["tensorflow_available"] = True
        info["tensorflow_version"] = tf.__version__
        info["tf_deterministic_ops"] = os.environ.get("TF_DETERMINISTIC_OPS", "not set")
    except ImportError:
        info["tensorflow_available"] = False

    return info
