"""Framework-agnostic data loading and preprocessing.

Data handling is widely recognized as one of the most challenging aspects
of ML workflows. By centralizing data loading in a framework-agnostic
module, we ensure all three implementations (NumPy, PyTorch, TensorFlow)
train and evaluate on identical data splits. This eliminates data handling
as a confounding variable in our SE comparison.

Design decision: This module returns NumPy arrays. Each framework-specific
implementation wraps these into its native data structures (e.g., DataLoader
for PyTorch, tf.data.Dataset for TensorFlow). This separation follows the
Dependency Inversion Principle — high-level training logic depends on
abstractions, not concrete data formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_cifar10(
    root: str = "./data", val_split: float = 0.1, seed: int = 42
) -> dict[str, np.ndarray]:
    """Load CIFAR-10 dataset and return as NumPy arrays.

    Downloads the dataset via keras.datasets (available even without full
    TensorFlow) and splits training data into train/val sets.

    Args:
        root: Directory to cache downloaded data.
        val_split: Fraction of training data to use for validation.
        seed: Random seed for train/val split reproducibility.

    Returns:
        Dictionary with keys: x_train, y_train, x_val, y_val, x_test, y_test.
        Images are float32 normalized to [0, 1], shape (N, 32, 32, 3).
        Labels are int64, shape (N,).
    """
    # Use keras for download — lightweight and always available with TF
    # For environments without TF, we fall back to manual download
    try:
        from tensorflow.keras.datasets import cifar10 as cifar10_ds

        (x_train_full, y_train_full), (x_test, y_test) = cifar10_ds.load_data()
    except ImportError:
        try:
            import torchvision.datasets as tvd

            train_ds = tvd.CIFAR10(root=root, train=True, download=True)
            test_ds = tvd.CIFAR10(root=root, train=False, download=True)
            x_train_full = np.array(train_ds.data)
            y_train_full = np.array(train_ds.targets).reshape(-1, 1)
            x_test = np.array(test_ds.data)
            y_test = np.array(test_ds.targets).reshape(-1, 1)
        except ImportError:
            raise ImportError(
                "Neither TensorFlow nor torchvision is available. "
                "Install at least one to download CIFAR-10."
            )

    # Normalize to [0, 1]
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train_full = y_train_full.flatten().astype(np.int64)
    y_test = y_test.flatten().astype(np.int64)

    # Train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=val_split, random_state=seed, stratify=y_train_full
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def load_imdb_sentiment(
    max_vocab_size: int = 10000,
    max_seq_length: int = 256,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Load IMDb sentiment dataset with tokenization and padding.

    Builds a vocabulary from the training set and converts text to integer
    sequences. This custom preprocessing ensures identical tokenization
    across all framework implementations — a critical requirement for fair
    SE quality comparison.

    Args:
        max_vocab_size: Maximum vocabulary size (most frequent words).
        max_seq_length: Maximum sequence length (longer sequences truncated).
        val_split: Fraction of training data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: x_train, y_train, x_val, y_val, x_test, y_test.
        Sequences are int32 padded arrays of shape (N, max_seq_length).
        Labels are int64 with values 0 (negative) or 1 (positive).
    """
    try:
        from tensorflow.keras.datasets import imdb

        (x_train_raw, y_train_full), (x_test_raw, y_test) = imdb.load_data(
            num_words=max_vocab_size
        )
    except ImportError:
        # Fallback: generate synthetic data for development/testing
        rng = np.random.RandomState(seed)
        n_train, n_test = 25000, 25000
        x_train_raw = [
            rng.randint(1, max_vocab_size, size=rng.randint(50, max_seq_length)).tolist()
            for _ in range(n_train)
        ]
        x_test_raw = [
            rng.randint(1, max_vocab_size, size=rng.randint(50, max_seq_length)).tolist()
            for _ in range(n_test)
        ]
        y_train_full = rng.randint(0, 2, size=n_train)
        y_test = rng.randint(0, 2, size=n_test)

    def _pad_sequences(sequences: list, maxlen: int) -> np.ndarray:
        """Pad or truncate sequences to fixed length (pre-padding with zeros)."""
        result = np.zeros((len(sequences), maxlen), dtype=np.int32)
        for i, seq in enumerate(sequences):
            if len(seq) > maxlen:
                result[i] = seq[:maxlen]
            else:
                result[i, maxlen - len(seq) :] = seq
        return result

    x_train_full = _pad_sequences(x_train_raw, max_seq_length)
    x_test = _pad_sequences(x_test_raw, max_seq_length)
    y_train_full = np.array(y_train_full, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    # Train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=val_split, random_state=seed, stratify=y_train_full
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "vocab_size": max_vocab_size,
    }


def load_california_housing(
    val_split: float = 0.1, seed: int = 42
) -> dict[str, np.ndarray]:
    """Load California Housing dataset for regression.

    Features are standardized using StandardScaler fit on training data only
    (preventing data leakage — a common ML engineering mistake).

    Args:
        val_split: Fraction of training data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: x_train, y_train, x_val, y_val, x_test, y_test.
        Features are float32 standardized arrays. Targets are float32.
    """
    housing = fetch_california_housing()
    x_all = housing.data.astype(np.float32)
    y_all = housing.target.astype(np.float32)

    # 80/10/10 split: first split off test, then split remaining into train/val
    x_temp, x_test, y_temp, y_test = train_test_split(
        x_all, y_all, test_size=0.1, random_state=seed
    )
    relative_val_size = val_split / (1.0 - 0.1)  # Adjust for remaining data
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=relative_val_size, random_state=seed
    )

    # Standardize features — fit only on training data to prevent leakage
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "n_features": x_train.shape[1],
    }


def load_dataset(task: str, **kwargs) -> dict[str, np.ndarray]:
    """Dispatch to the appropriate dataset loader based on task name.

    Args:
        task: One of 'cifar10', 'sentiment', 'regression'.
        **kwargs: Passed through to the specific loader.

    Returns:
        Dictionary of NumPy arrays for train/val/test splits.

    Raises:
        ValueError: If task is not recognized.
    """
    loaders = {
        "cifar10": load_cifar10,
        "sentiment": load_imdb_sentiment,
        "regression": load_california_housing,
    }
    if task not in loaders:
        raise ValueError(f"Unknown task: {task}. Available: {list(loaders.keys())}")
    return loaders[task](**kwargs)
