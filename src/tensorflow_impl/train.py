"""Training loop for TensorFlow/Keras implementations.

SE observation: Keras model.fit() reduces the entire training loop to a single
call with callbacks — the most concise approach across all three frameworks.
However, debugging custom behavior requires overriding train_step(), which
reintroduces much of the complexity that fit() abstracts away.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.common.config import ExperimentConfig, parse_args
from src.common.data_loader import load_dataset
from src.common.logging_utils import end_run, log_epoch_metrics, log_final_results, setup_experiment
from src.common.metrics import compute_task_metrics
from src.common.reproducibility import set_global_seed
from src.tensorflow_impl.datasets import create_tf_datasets
from src.tensorflow_impl.models.cnn_cifar10 import create_cnn_cifar10
from src.tensorflow_impl.models.lstm_sentiment import create_lstm_sentiment
from src.tensorflow_impl.models.mlp_regression import create_mlp_regression


class MLflowCallback(keras.callbacks.Callback):
    """Custom Keras callback for MLflow metric logging."""

    def __init__(self, task: str):
        super().__init__()
        self.task = task

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            metrics = {k: float(v) for k, v in logs.items()}
            log_epoch_metrics(epoch, metrics)


def _create_model(config: ExperimentConfig, data: dict) -> keras.Model:
    """Instantiate and compile a Keras model."""
    if config.task == "cifar10":
        model = create_cnn_cifar10()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.training.learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
    elif config.task == "sentiment":
        vocab_size = data.get("vocab_size", config.data.max_vocab_size)
        model = create_lstm_sentiment(
            vocab_size=vocab_size,
            embedding_dim=config.model.embedding_dim,
            lstm_hidden=config.model.lstm_hidden,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.training.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    elif config.task == "regression":
        n_features = data.get("n_features", 8)
        model = create_mlp_regression(n_features=n_features, hidden_sizes=config.model.hidden_sizes)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.training.learning_rate),
            loss="mse",
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")
    return model


def train(config: ExperimentConfig) -> dict:
    """Execute full TensorFlow/Keras training pipeline."""
    set_global_seed(config.training.seed)
    run_id = setup_experiment(config)

    # Load data
    data = load_dataset(config.task, val_split=config.data.val_split, seed=config.training.seed)
    datasets = create_tf_datasets(data, config.task, batch_size=config.training.batch_size)

    # Create and compile model
    model = _create_model(config, data)

    # Train with model.fit() — the Keras-native approach
    callbacks = [MLflowCallback(config.task)]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=config.training.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Test evaluation
    if config.task in ("cifar10", "sentiment"):
        test_output = model.predict(datasets["test"])
        if config.task == "cifar10":
            test_preds = test_output.argmax(axis=1)
        else:
            test_preds = (test_output.flatten() > 0.5).astype(np.int64)
    else:
        test_preds = model.predict(datasets["test"]).flatten()

    test_metrics = compute_task_metrics(config.task, data["y_test"], test_preds)
    log_final_results(test_metrics)

    # Save model
    if config.logging.save_model:
        save_path = f"{config.logging.model_dir}/{config.task}_tensorflow"
        model.save(save_path)

    print(f"\nTest Results: {test_metrics}")
    end_run()
    return test_metrics


if __name__ == "__main__":
    cfg = parse_args()
    cfg.framework = "tensorflow"
    train(cfg)
