# Software Engineering for Machine Learning: A Comparative Analysis of Implementation Practices Across NumPy, PyTorch, and TensorFlow in the Era of MLOps

> **MSc Project Thesis - Sazzad Hossain** | Computer Science and Engineering

## Overview

This repository contains all code, configurations, and experimental artifacts for an MSc thesis investigating **software engineering quality attributes** across three major ML frameworks. The work implements equivalent machine learning tasks in NumPy, PyTorch, and TensorFlow, then measures and compares SE metrics including cyclomatic complexity, maintainability index, test coverage, and API surface area.

The project also provides a systematic analysis of **MLOps practices**, mapping CI/CD pipeline patterns, experiment tracking workflows, and deployment strategies against published maturity models from Google and Microsoft.

## Research Questions

- **RQ1:** How do NumPy, PyTorch, and TensorFlow differ in SE quality attributes (maintainability, testability, modularity, reproducibility) when implementing equivalent ML tasks?
- **RQ2:** What are current MLOps best practices and how do they map to framework-specific implementation patterns?
- **RQ3:** What SE challenges are specific to ML systems, and how do modern tools and practices address them?
- **RQ4:** What are the emerging trends, ethical considerations, and sustainability implications shaping the future of SE4ML?

## Benchmark Tasks

| Task | Dataset | Architecture | Modality |
|------|---------|-------------|----------|
| Image Classification | CIFAR-10 | CNN (3 conv blocks + 2 FC) | Vision |
| Sentiment Analysis | IMDb Reviews | Embedding + LSTM + FC | NLP |
| Tabular Regression | California Housing | 3-layer MLP | Structured |

Each task is implemented in **three frameworks** (NumPy from scratch, PyTorch, TensorFlow/Keras) with identical architectures, hyperparameters, and data splits. This yields **9 implementations** whose SE properties are then measured and compared.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sazzadhossain05/Software-Engineering-for-Machine-Learning-Code.git
cd Software-Engineering-for-Machine-Learning-Code

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install base + one framework
pip install -r requirements/base.txt
pip install -r requirements/pytorch.txt      # or tensorflow.txt

# Install development tools
pip install -r requirements/dev.txt

# Run all tests
make test

# Train a model (example: PyTorch CIFAR-10)
python -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml

# Run SE metric analysis on all implementations
make analyze

# Start MLflow tracking UI
make mlflow
```

## Repository Structure

```
se4ml-thesis/
├── src/
│   ├── common/              # Shared utilities (data loading, metrics, config, reproducibility)
│   ├── numpy_impl/          # NumPy from-scratch implementations
│   ├── pytorch_impl/        # PyTorch implementations
│   ├── tensorflow_impl/     # TensorFlow/Keras implementations
│   └── analysis/            # SE metric collection, visualization, comparison tools
├── tests/                   # Unit, integration, and pipeline tests
├── configs/                 # YAML experiment configurations
├── mlops/                   # Docker, DVC pipeline, docker-compose
├── notebooks/               # Exploratory analysis (non-production)
├── .github/workflows/       # CI/CD pipeline definitions
├── data/                    # Datasets (managed by DVC)
├── models/                  # Saved model artifacts (managed by DVC)
├── results/                 # Experiment outputs and SE metrics
├── docs/                    # Thesis source, figures, tables
├── requirements/            # Dependency files per framework
└── Makefile                 # Common automation commands
```

## SE Metrics Collected

| Metric | Tool | Purpose |
|--------|------|---------|
| Cyclomatic Complexity | Radon | Decision complexity per function |
| Maintainability Index | Radon | Composite maintainability score |
| Logical Lines of Code | Radon | Implementation verbosity |
| pylint Score | pylint | Overall code quality |
| Test Coverage | pytest-cov | Code exercised by tests |
| API Surface Area | Custom | Framework API dependency |
| Cognitive Complexity | Custom | Code comprehension difficulty |

## MLOps Stack

| Component | Tool |
|-----------|------|
| Experiment Tracking | MLflow |
| Data Versioning | DVC (remote: Google Drive) |
| CI/CD | GitHub Actions |
| Code Quality | Black + isort + flake8 + pylint |
| Testing | pytest + pytest-cov |
| Containerization | Docker |
| Static Analysis | Radon + custom scripts |

## Reproducibility

All experiments use fixed random seeds and deterministic configurations.

```bash
# Pull versioned data
dvc pull

# Reproduce the full pipeline
dvc repro

# Or run a specific experiment with MLflow
python -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml
```

Every experiment logs parameters, metrics, and artifacts to MLflow. View results:

```bash
mlflow ui --port 5000
```

## Key Makefile Commands

```bash
make install          # Install all dependencies
make test             # Run full test suite with coverage
make lint             # Run linting (black, isort, flake8, pylint)
make analyze          # Run SE metric analysis on all implementations
make train-all        # Train all 9 implementations
make mlflow           # Launch MLflow UI
make clean            # Remove generated artifacts
```

## License

MIT License — see [LICENSE](LICENSE) for details.
