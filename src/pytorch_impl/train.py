"""Training loop for PyTorch implementations.

SE observation: PyTorch's training loop is more concise than NumPy's but
requires explicit gradient zeroing, loss.backward(), and optimizer.step()
calls — a pattern that is both flexible and error-prone (forgetting
zero_grad is a documented common bug in PyTorch codebases).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.common.config import ExperimentConfig, parse_args
from src.common.data_loader import load_dataset
from src.common.logging_utils import end_run, log_epoch_metrics, log_final_results, setup_experiment
from src.common.metrics import compute_task_metrics
from src.common.reproducibility import set_global_seed
from src.pytorch_impl.datasets import create_dataloaders
from src.pytorch_impl.models.cnn_cifar10 import PyTorchCNNCifar10
from src.pytorch_impl.models.lstm_sentiment import PyTorchLSTMSentiment
from src.pytorch_impl.models.mlp_regression import PyTorchMLPRegression


def _create_model_and_criterion(
    config: ExperimentConfig, data: dict
) -> tuple[nn.Module, nn.Module]:
    """Instantiate model and loss function."""
    if config.task == "cifar10":
        model = PyTorchCNNCifar10()
        criterion = nn.CrossEntropyLoss()
    elif config.task == "sentiment":
        vocab_size = data.get("vocab_size", config.data.max_vocab_size)
        model = PyTorchLSTMSentiment(
            vocab_size=vocab_size,
            embedding_dim=config.model.embedding_dim,
            lstm_hidden=config.model.lstm_hidden,
        )
        criterion = nn.BCELoss()
    elif config.task == "regression":
        n_features = data.get("n_features", 8)
        model = PyTorchMLPRegression(n_features=n_features, hidden_sizes=config.model.hidden_sizes)
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {config.task}")

    return model, criterion


def _get_predictions(output: torch.Tensor, task: str) -> np.ndarray:
    """Convert model output to numpy predictions."""
    with torch.no_grad():
        if task == "cifar10":
            return output.argmax(dim=1).cpu().numpy()
        elif task == "sentiment":
            return (output > 0.5).long().cpu().numpy()
        else:
            return output.cpu().numpy()


def train(config: ExperimentConfig) -> dict:
    """Execute full PyTorch training pipeline."""
    set_global_seed(config.training.seed)
    run_id = setup_experiment(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and create DataLoaders
    data = load_dataset(config.task, val_split=config.data.val_split, seed=config.training.seed)
    loaders = create_dataloaders(data, config.task, batch_size=config.training.batch_size)

    # Create model, criterion, optimizer
    model, criterion = _create_model_and_criterion(config, data)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Training loop
    for epoch in range(config.training.epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses = []

        for x_batch, y_batch in loaders["train"]:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds_list, val_targets_list = [], []
        val_losses = []

        with torch.no_grad():
            for x_batch, y_batch in loaders["val"]:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                val_losses.append(criterion(output, y_batch).item())
                val_preds_list.append(_get_predictions(output, config.task))
                val_targets_list.append(y_batch.cpu().numpy())

        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)
        val_metrics = compute_task_metrics(config.task, val_targets, val_preds)

        epoch_time = time.time() - epoch_start
        metrics = {
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": float(np.mean(val_losses)),
            "epoch_time_s": epoch_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        log_epoch_metrics(epoch, metrics)

        primary = list(val_metrics.values())[0]
        print(
            f"Epoch {epoch + 1}/{config.training.epochs} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val Metric: {primary:.4f} | Time: {epoch_time:.1f}s"
        )

    # Test evaluation
    model.eval()
    test_preds_list, test_targets_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in loaders["test"]:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            test_preds_list.append(_get_predictions(output, config.task))
            test_targets_list.append(y_batch.numpy())

    test_preds = np.concatenate(test_preds_list)
    test_targets = np.concatenate(test_targets_list)
    test_metrics = compute_task_metrics(config.task, test_targets, test_preds)
    log_final_results(test_metrics)

    # Save model
    if config.logging.save_model:
        save_path = f"{config.logging.model_dir}/{config.task}_pytorch.pt"
        torch.save(model.state_dict(), save_path)

    print(f"\nTest Results: {test_metrics}")
    end_run()
    return test_metrics


if __name__ == "__main__":
    cfg = parse_args()
    cfg.framework = "pytorch"
    train(cfg)
