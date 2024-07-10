"""PyTorch Dataset wrappers for framework-agnostic NumPy data.

Design decision: We wrap the NumPy arrays from src.common.data_loader into
PyTorch Dataset/DataLoader objects. This keeps data loading centralized
while using PyTorch's native batching, shuffling, and multiprocessing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def numpy_to_tensor_dataset(
    x: np.ndarray, y: np.ndarray, task: str
) -> TensorDataset:
    """Convert NumPy arrays to a PyTorch TensorDataset.

    Args:
        x: Input features as NumPy array.
        y: Labels/targets as NumPy array.
        task: Task name — determines tensor dtype and layout.

    Returns:
        TensorDataset ready for DataLoader.
    """
    if task == "cifar10":
        # Convert NHWC to NCHW for PyTorch convention
        x_tensor = torch.from_numpy(x.transpose(0, 3, 1, 2)).float()
        y_tensor = torch.from_numpy(y).long()
    elif task == "sentiment":
        x_tensor = torch.from_numpy(x).long()
        y_tensor = torch.from_numpy(y).float()
    elif task == "regression":
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()
    else:
        raise ValueError(f"Unknown task: {task}")

    return TensorDataset(x_tensor, y_tensor)


def create_dataloaders(
    data: dict[str, np.ndarray],
    task: str,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders from data dictionary.

    Args:
        data: Dictionary from load_dataset() with x_train, y_train, etc.
        task: Task name for dtype handling.
        batch_size: Mini-batch size.
        num_workers: Number of data loading workers.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader instances.
    """
    train_ds = numpy_to_tensor_dataset(data["x_train"], data["y_train"], task)
    val_ds = numpy_to_tensor_dataset(data["x_val"], data["y_val"], task)
    test_ds = numpy_to_tensor_dataset(data["x_test"], data["y_test"], task)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
