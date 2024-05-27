"""Configuration management for experiments.

This module provides a unified interface for loading YAML-based experiment
configurations. Centralizing config handling is an SE best practice that
reduces hardcoded values and improves reproducibility — a key concern
in ML systems where hyperparameter choices directly affect results.

Configuration management is identified as a major source of technical
debt in ML systems. This module addresses that by enforcing typed,
validated configurations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Dataset configuration."""

    name: str = "cifar10"
    root: str = "./data"
    val_split: float = 0.1
    max_vocab_size: int = 10000  # For text tasks
    max_seq_length: int = 256  # For text tasks


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "cnn"
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.0
    embedding_dim: int = 128  # For text tasks
    lstm_hidden: int = 128  # For text tasks


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0
    seed: int = 42


@dataclass
class LoggingConfig:
    """MLflow and logging configuration."""

    experiment_name: str = "default"
    run_name: str | None = None
    log_every_n_steps: int = 100
    save_model: bool = True
    model_dir: str = "./models"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""

    task: str = "cifar10"
    framework: str = "pytorch"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dataclass(dc_instance: Any, overrides: dict) -> None:
    """Recursively merge a dict of overrides into a dataclass instance."""
    for key, value in overrides.items():
        if hasattr(dc_instance, key):
            current = getattr(dc_instance, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                _merge_dataclass(current, value)
            else:
                setattr(dc_instance, key, value)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Fully populated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    config = ExperimentConfig()
    _merge_dataclass(config, raw)
    return config


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments to load a config file.

    Returns:
        ExperimentConfig loaded from the specified YAML file.
    """
    parser = argparse.ArgumentParser(description="Run ML experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file",
    )
    args = parser.parse_args()
    return load_config(args.config)
