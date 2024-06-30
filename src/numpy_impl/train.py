"""Training loop for NumPy implementations.

This is the most verbose training loop in the project — every operation
that PyTorch/TF automate (batching, gradient computation, metric logging)
must be handled manually. This verbosity is itself a key SE finding.
"""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from src.common.config import ExperimentConfig, parse_args
from src.common.data_loader import load_dataset
from src.common.logging_utils import end_run, log_epoch_metrics, log_final_results, setup_experiment
from src.common.metrics import compute_task_metrics
from src.common.reproducibility import set_global_seed
from src.numpy_impl.models.cnn_cifar10 import NumpyCNNCifar10, cross_entropy_loss
from src.numpy_impl.models.lstm_sentiment import NumpyLSTMSentiment, binary_cross_entropy_loss
from src.numpy_impl.models.mlp_regression import NumpyMLPRegression, mse_loss
from src.numpy_impl.optimizers import create_optimizer


def _create_model(config: ExperimentConfig, data: dict) -> tuple:
    """Instantiate model and loss function based on task config."""
    seed = config.training.seed

    if config.task == "cifar10":
        model = NumpyCNNCifar10(seed=seed)
        loss_fn = cross_entropy_loss
    elif config.task == "sentiment":
        vocab_size = data.get("vocab_size", config.data.max_vocab_size)
        model = NumpyLSTMSentiment(
            vocab_size=vocab_size,
            embedding_dim=config.model.embedding_dim,
            lstm_hidden=config.model.lstm_hidden,
            seed=seed,
        )
        loss_fn = binary_cross_entropy_loss
    elif config.task == "regression":
        n_features = data.get("n_features", 8)
        model = NumpyMLPRegression(
            n_features=n_features,
            hidden_sizes=config.model.hidden_sizes,
            seed=seed,
        )
        loss_fn = mse_loss
    else:
        raise ValueError(f"Unknown task: {config.task}")

    return model, loss_fn


def _batch_iterator(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Yield mini-batches from data arrays."""
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]


def _prepare_input(x_batch: np.ndarray, task: str) -> np.ndarray:
    """Convert data to the format expected by the NumPy model."""
    if task == "cifar10":
        # Convert NHWC (from data loader) to NCHW (for our Conv2D)
        return x_batch.transpose(0, 3, 1, 2)
    return x_batch


def _get_predictions(output: np.ndarray, task: str) -> np.ndarray:
    """Convert model output to predictions for metric computation."""
    if task == "cifar10":
        return output.argmax(axis=1)
    elif task == "sentiment":
        return (output.flatten() > 0.5).astype(np.int64)
    else:  # regression
        return output.flatten()


def train(config: ExperimentConfig) -> dict:
    """Execute the full training pipeline for a NumPy implementation.

    Args:
        config: Experiment configuration.

    Returns:
        Dictionary of final test metrics.
    """
    # Setup
    set_global_seed(config.training.seed)
    run_id = setup_experiment(config)

    # Load data
    data = load_dataset(
        config.task,
        val_split=config.data.val_split,
        seed=config.training.seed,
    )

    # Create model and optimizer
    model, loss_fn = _create_model(config, data)
    optimizer = create_optimizer(
        config.training.optimizer,
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Training loop
    for epoch in range(config.training.epochs):
        epoch_start = time.time()
        epoch_losses = []

        for x_batch, y_batch in _batch_iterator(
            data["x_train"], data["y_train"], config.training.batch_size
        ):
            x_batch = _prepare_input(x_batch, config.task)

            # Forward pass
            output = model.forward(x_batch)
            loss, grad = loss_fn(output, y_batch)
            epoch_losses.append(loss)

            # Backward pass
            model.backward(grad)

            # Parameter update — must refresh references after each step
            optimizer.parameters = model.parameters()
            optimizer.step()
            optimizer.zero_grad()

        # Validation evaluation
        x_val = _prepare_input(data["x_val"], config.task)
        val_output = model.forward(x_val)
        val_loss, _ = loss_fn(val_output, data["y_val"])
        val_preds = _get_predictions(val_output, config.task)
        val_targets = data["y_val"]
        val_metrics = compute_task_metrics(config.task, val_targets, val_preds)

        epoch_time = time.time() - epoch_start
        metrics = {
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val_loss,
            "epoch_time_s": epoch_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        log_epoch_metrics(epoch, metrics)

        primary_metric = list(val_metrics.values())[0]
        print(
            f"Epoch {epoch + 1}/{config.training.epochs} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {primary_metric:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    # Test evaluation
    x_test = _prepare_input(data["x_test"], config.task)
    test_output = model.forward(x_test)
    test_preds = _get_predictions(test_output, config.task)
    test_metrics = compute_task_metrics(config.task, data["y_test"], test_preds)
    log_final_results(test_metrics)

    print(f"\nTest Results: {test_metrics}")

    end_run()
    return test_metrics


if __name__ == "__main__":
    cfg = parse_args()
    cfg.framework = "numpy"
    train(cfg)
