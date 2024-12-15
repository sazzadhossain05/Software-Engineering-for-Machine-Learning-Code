# Software Engineering for Machine Learning: A Comparative Analysis of Implementation Practices Across NumPy, PyTorch, and TensorFlow in the Era of MLOps

> **MSc Project Thesis - Sazzad Hossain** | Computer Science and Engineering | North South University

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

## Experimental Results

All 9 experiments were run and results are committed to the `results/` folder.

| Task | NumPy | PyTorch | TensorFlow |
|------|-------|---------|------------|
| Regression (R²) | 0.781 | 0.784 | **0.795** |
| CIFAR-10 (Accuracy) | 56.75% | **70.38%** | 68.57% |
| Sentiment (Accuracy) | 50.00% ⚠️ | 50.28% ⚠️ | **83.72%** |

> ⚠️ NumPy LSTM suffers from vanishing gradients; PyTorch LSTM overfits. TensorFlow's built-in LSTM handles long sequences significantly better.

### SE Metric Results

| Metric | Value |
|--------|-------|
| Average Cyclomatic Complexity | 2.21 |
| Average Maintainability Index | 83.55 |
| Total Source Lines of Code | 1,455 |
| Total Files Analyzed | 34 |
| Test Suite | 55 tests, 41% coverage |

## Repository Structure

```
SE4ML-Code/
├── src/
│   ├── common/              # Shared utilities (data loading, metrics, config, reproducibility)
│   ├── numpy_impl/          # NumPy from-scratch implementations
│   ├── pytorch_impl/        # PyTorch implementations
│   ├── tensorflow_impl/     # TensorFlow/Keras implementations
│   └── analysis/            # SE metric collection, visualization, comparison tools
├── tests/                   # Unit, integration, and pipeline tests
├── configs/                 # YAML experiment configurations
├── mlops/                   # Docker, DVC pipeline, docker-compose
├── .github/workflows/       # CI/CD pipeline definitions
├── data/                    # Datasets (auto-downloaded on first run)
├── models/                  # Saved model artifacts
├── results/                 # Experiment outputs and SE metrics
├── requirements/            # Dependency files per framework
└── Makefile                 # Common automation commands (Linux/macOS)
```

## Quick Start (Linux / macOS)

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

# Run SE metric analysis
make analyze

# Start MLflow tracking UI
make mlflow
```

## Quick Start (Windows — Git Bash)

`make` is not available on Windows. Use the following equivalent commands instead.

### Step 1 — Clone and navigate

```bash
git clone https://github.com/sazzadhossain05/Software-Engineering-for-Machine-Learning-Code.git
cd "Software-Engineering-for-Machine-Learning-Code"
```

### Step 2 — Install dependencies

```bash
# Base dependencies (required for all frameworks)
pip install -r requirements/base.txt

# Choose one or more frameworks
pip install -r requirements/pytorch.txt
pip install -r requirements/tensorflow.txt

# Development and analysis tools
pip install -r requirements/dev.txt
pip install radon pytest-cov
```

### Step 3 — Download datasets (Windows SSL fix)

On Windows, scikit-learn may fail to download datasets due to SSL certificate issues. Run this once to pre-download all datasets:

```bash
# Fix SSL and download California Housing (regression)
python -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_california_housing
fetch_california_housing()
print('California Housing downloaded successfully')
"
```

CIFAR-10 and IMDb are downloaded automatically on first run via torchvision/keras.

### Step 4 — Run experiments

**NumPy experiments:**
```bash
python -m src.numpy_impl.train --config configs/regression_numpy.yaml
python -m src.numpy_impl.train --config configs/cifar10_numpy.yaml
python -m src.numpy_impl.train --config configs/sentiment_numpy.yaml
```

**PyTorch experiments:**
```bash
python -m src.pytorch_impl.train --config configs/regression_pytorch.yaml
python -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml
python -m src.pytorch_impl.train --config configs/sentiment_pytorch.yaml
```

**TensorFlow experiments:**
```bash
python -m src.tensorflow_impl.train --config configs/regression_tensorflow.yaml
python -m src.tensorflow_impl.train --config configs/cifar10_tensorflow.yaml
python -m src.tensorflow_impl.train --config configs/sentiment_tensorflow.yaml
```

> **Note for TensorFlow on Windows:** TensorFlow >= 2.11 does not support GPU on native Windows. CPU-only training will be used automatically. GPU support is available via WSL2.

### Step 5 — Run tests

```bash
pytest tests/ -v --cov=src --cov-report=json:results/coverage_report.json --cov-report=term-missing -m "not slow"
```

Expected output: **55 tests passed**, 41% coverage.

### Step 6 — Run SE analysis

```bash
# Code metrics (lines of code, functions, classes)
python -m src.analysis.code_metrics --target src --output results/code_metrics.json

# Cyclomatic complexity and maintainability index
python -m src.analysis.complexity_analysis --target src --output results/complexity_report.json
```

### Step 7 — Generate visualizations

```bash
python -c "
import matplotlib.pyplot as plt
import numpy as np
import json

results = {
    'numpy':      {'regression_r2': 0.781, 'cifar10_acc': 0.568, 'sentiment_acc': 0.500},
    'pytorch':    {'regression_r2': 0.784, 'cifar10_acc': 0.704, 'sentiment_acc': 0.503},
    'tensorflow': {'regression_r2': 0.795, 'cifar10_acc': 0.686, 'sentiment_acc': 0.837},
}

frameworks = list(results.keys())
metrics = ['regression_r2', 'cifar10_acc', 'sentiment_acc']
labels = ['Regression (R2)', 'CIFAR-10 (Accuracy)', 'Sentiment (Accuracy)']
x = np.arange(len(metrics))
width = 0.25
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(12, 6))
for i, (fw, color) in enumerate(zip(frameworks, colors)):
    values = [results[fw][m] for m in metrics]
    ax.bar(x + i * width, values, width, label=fw.capitalize(), color=color)

ax.set_xlabel('Task / Metric')
ax.set_ylabel('Score')
ax.set_title('SE4ML: Framework Comparison Across Tasks')
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig('results/framework_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/framework_comparison.png')
"
```

### Step 8 — View MLflow results

```bash
mlflow ui --port 5000
```

Then open your browser at `http://localhost:5000` to view all experiment runs, metrics, and parameters.

## SE Metrics Collected

| Metric | Tool | Purpose |
|--------|------|---------|
| Cyclomatic Complexity | Radon | Decision complexity per function |
| Maintainability Index | Radon | Composite maintainability score |
| Logical Lines of Code | Radon | Implementation verbosity |
| Test Coverage | pytest-cov | Code exercised by tests |
| API Surface Area | Custom | Framework API dependency |
| Cognitive Complexity | Custom | Code comprehension difficulty |

## MLOps Stack

| Component | Tool |
|-----------|------|
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Code Quality | Black + isort + flake8 |
| Testing | pytest + pytest-cov |
| Containerization | Docker |
| Static Analysis | Radon + custom scripts |

## Reproducibility

All experiments use fixed random seeds (seed=42) and deterministic configurations. Every experiment logs parameters, metrics, and artifacts to MLflow automatically.

```bash
# Run a specific experiment
python -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml

# View results in MLflow UI
mlflow ui --port 5000
```

## Key Findings

1. **TensorFlow dominates NLP** — 83.72% sentiment accuracy vs 50% for NumPy and PyTorch, due to superior built-in LSTM implementation handling vanishing gradients
2. **PyTorch leads image classification** — 70.38% CIFAR-10 accuracy, slightly ahead of TensorFlow (68.57%) and significantly ahead of NumPy (56.75%)
3. **All frameworks converge for regression** — R² scores within 1.4% of each other (0.781–0.795), suggesting framework choice matters less for simple tabular tasks
4. **NumPy is prohibitively slow for complex tasks** — 100s per epoch for CIFAR-10 vs 25s (PyTorch) and 12s (TensorFlow)
5. **Code maintainability is high** — Average maintainability index of 83.55/100 across all implementations

## License

Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)

Copyright (c) 2024 Sazzad Hossain

You are free to use, share, modify, and build upon this work for non-commercial purposes with attribution. Commercial use is prohibited without explicit written permission from the author.

See [LICENSE](LICENSE) for full details.
