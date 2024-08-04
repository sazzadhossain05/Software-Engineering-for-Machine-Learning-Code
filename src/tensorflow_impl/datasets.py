"""TensorFlow tf.data pipeline wrappers.

SE observation: tf.data pipelines offer performance optimizations (prefetch,
parallel map) but add API complexity compared to simple NumPy iteration
or PyTorch DataLoader. The trade-off: better throughput at the cost of
more opaque data handling code.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def create_tf_datasets(
    data: dict[str, np.ndarray], task: str, batch_size: int = 128
) -> dict[str, tf.data.Dataset]:
    """Create tf.data.Dataset pipelines from NumPy arrays.

    Args:
        data: Dictionary from load_dataset() with x_train, y_train, etc.
        task: Task name for dtype handling.
        batch_size: Mini-batch size.

    Returns:
        Dictionary with 'train', 'val', 'test' tf.data.Dataset instances.
    """
    def _make_dataset(x, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(10000, len(x)))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return {
        "train": _make_dataset(data["x_train"], data["y_train"], shuffle=True),
        "val": _make_dataset(data["x_val"], data["y_val"]),
        "test": _make_dataset(data["x_test"], data["y_test"]),
    }
