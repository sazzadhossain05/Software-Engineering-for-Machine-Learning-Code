"""MLflow integration helpers for experiment tracking.

Experiment tracking is a core pillar of MLOps maturity. MLflow is chosen
as the tracking backend because it is open-source, framework-agnostic,
and provides the most comprehensive lifecycle management among free
tools (Tracking, Projects, Models, Registry).

This module wraps MLflow calls to provide a consistent logging interface
across all three framework implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import mlflow

from src.common.config import ExperimentConfig
from src.common.reproducibility import get_seed_info


def setup_experiment(config: ExperimentConfig) -> str:
    """Initialize MLflow experiment and start a new run.

    Creates the experiment if it doesn't exist, then starts a run
    with all configuration parameters logged.

    Args:
        config: Full experiment configuration.

    Returns:
        The MLflow run ID.
    """
    mlflow.set_experiment(config.logging.experiment_name)

    run = mlflow.start_run(run_name=config.logging.run_name)

    # Log all configuration as parameters
    mlflow.log_param("task", config.task)
    mlflow.log_param("framework", config.framework)
    mlflow.log_param("batch_size", config.training.batch_size)
    mlflow.log_param("epochs", config.training.epochs)
    mlflow.log_param("learning_rate", config.training.learning_rate)
    mlflow.log_param("optimizer", config.training.optimizer)
    mlflow.log_param("seed", config.training.seed)
    mlflow.log_param("model_name", config.model.name)

    # Log reproducibility info
    seed_info = get_seed_info()
    mlflow.log_param("reproducibility_info", json.dumps(seed_info))

    return run.info.run_id


def log_epoch_metrics(epoch: int, metrics: dict[str, float]) -> None:
    """Log metrics for a single training epoch.

    Args:
        epoch: Current epoch number (0-indexed).
        metrics: Dictionary of metric name -> value pairs.
    """
    for name, value in metrics.items():
        mlflow.log_metric(name, value, step=epoch)


def log_final_results(metrics: dict[str, float]) -> None:
    """Log final evaluation metrics (on test set).

    Args:
        metrics: Dictionary of final metric name -> value pairs.
    """
    for name, value in metrics.items():
        mlflow.log_metric(f"test_{name}", value)


def log_artifact_file(filepath: str | Path) -> None:
    """Log a file as an MLflow artifact.

    Args:
        filepath: Path to the file to log.
    """
    mlflow.log_artifact(str(filepath))


def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()
