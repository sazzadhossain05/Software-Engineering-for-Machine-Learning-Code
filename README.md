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
| Regression (R2) | 0.781 | 0.784 | 0.795 (best) |
| CIFAR-10 (Accuracy) | 56.75% | 70.38% (best) | 68.57% |
| Sentiment (Accuracy) | 50.00% (failed - vanishing gradient) | 50.28% (failed - overfitting) | 83.72% (best) |

Note on Sentiment results: The NumPy LSTM implementation fails to learn due to vanishing gradients during backpropagation through time. The PyTorch LSTM memorizes training data but completely fails on unseen data due to severe overfitting. TensorFlow's built-in LSTM implementation handles long sequences significantly better, achieving 83.72% accuracy.

### SE Metric Results

| Metric | Value |
|--------|-------|
| Average Cyclomatic Complexity | 2.21 |
| Average Maintainability Index | 83.55 |
| Total Source Lines of Code | 1,455 |
| Total Files Analyzed | 34 |
| Test Suite | 55 tests passed, 41% coverage |

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

The `make` command is not available on Windows. Use the following equivalent commands instead.

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

On Windows, scikit-learn may fail to download datasets due to SSL certificate issues. Run this once to pre-download the California Housing dataset:

```bash
python -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_california_housing
fetch_california_housing()
print('California Housing downloaded successfully')
"
```

CIFAR-10 and IMDb datasets are downloaded automatically on first run via torchvision and keras respectively.

### Step 4 — Run experiments

See the full **Reproducing Experimental Results** section below for all 9 experiment commands with expected outputs.

### Step 5 — Run tests

```bash
pytest tests/ -v --cov=src --cov-report=json:results/coverage_report.json --cov-report=term-missing -m "not slow"
```

Expected output: 55 tests passed, 41% overall coverage.

### Step 6 — Run SE analysis

```bash
# Code metrics (lines of code, functions, classes per file)
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

## Reproducing Experimental Results

All 9 experiments can be reproduced using the commands below. Datasets are downloaded automatically on first run. All results are logged to MLflow automatically.

Note: TensorFlow model saving requires Keras 3 compatibility. The fix is already applied in `src/tensorflow_impl/train.py` (model saved with `.keras` extension).

---

### NumPy Experiments

**Regression — California Housing**
```bash
python -m src.numpy_impl.train --config configs/regression_numpy.yaml
```
Expected result: RMSE = 0.540, R2 = 0.781. Training runs for 100 epochs at approximately 0.1 seconds per epoch. The California Housing dataset is downloaded automatically via scikit-learn on first run. On Windows, run the SSL fix in Step 3 first before running this command.

**Image Classification — CIFAR-10**
```bash
python -m src.numpy_impl.train --config configs/cifar10_numpy.yaml
```
Expected result: Accuracy = 56.75%, F1 = 0.561. Training runs for 50 epochs at approximately 100 seconds per epoch. Note that this experiment takes approximately 90 minutes to complete. CIFAR-10 (170MB) is downloaded automatically via torchvision on first run, which requires PyTorch to be installed even when running the NumPy implementation.

**Sentiment Analysis — IMDb**
```bash
python -m src.numpy_impl.train --config configs/sentiment_numpy.yaml
```
Expected result: Accuracy = 49.99%, which is effectively random guessing. Training runs for 10 epochs at approximately 75 seconds per epoch. The NumPy LSTM implementation suffers from vanishing gradients and fails to learn. The loss remains stuck at 0.6931 (ln(2)) throughout all epochs, meaning the model outputs 50/50 probability for every prediction regardless of input.

---

### PyTorch Experiments

**Regression — California Housing**
```bash
python -m src.pytorch_impl.train --config configs/regression_pytorch.yaml
```
Expected result: RMSE = 0.537, R2 = 0.784. Training runs for 100 epochs at approximately 0.6 seconds per epoch.

**Image Classification — CIFAR-10**
```bash
python -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml
```
Expected result: Accuracy = 70.38%, F1 = 0.705. Training runs for 50 epochs at approximately 25 seconds per epoch. This is the best CIFAR-10 result across all three frameworks.

**Sentiment Analysis — IMDb**
```bash
python -m src.pytorch_impl.train --config configs/sentiment_pytorch.yaml
```
Expected result: Accuracy = 50.28%, which is effectively random guessing. Training runs for 10 epochs. Warning: each epoch takes approximately 10 to 13 minutes, so this experiment takes approximately 2 hours to complete. The PyTorch LSTM severely overfits — training loss drops to near zero (0.0008) while validation loss explodes to 3.40, indicating the model memorizes training data but cannot generalize to unseen reviews.

---

### TensorFlow Experiments

**Regression — California Housing**
```bash
python -m src.tensorflow_impl.train --config configs/regression_tensorflow.yaml
```
Expected result: RMSE = 0.523, R2 = 0.795. Training runs for 100 epochs at approximately 1 second per epoch. This is the best regression result across all three frameworks.

**Image Classification — CIFAR-10**
```bash
python -m src.tensorflow_impl.train --config configs/cifar10_tensorflow.yaml
```
Expected result: Accuracy = 68.57%, F1 = 0.683. Training runs for 50 epochs at approximately 12 seconds per epoch. CIFAR-10 (170MB) is downloaded automatically via keras on first run.

**Sentiment Analysis — IMDb**
```bash
python -m src.tensorflow_impl.train --config configs/sentiment_tensorflow.yaml
```
Expected result: Accuracy = 83.72%, F1 = 0.837. Training runs for 10 epochs at approximately 2 minutes per epoch, so approximately 20 minutes total. This is by far the best sentiment result across all three frameworks, demonstrating TensorFlow's superior built-in LSTM implementation for handling long text sequences without vanishing gradient issues.

---

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
# View all experiment results in MLflow UI
mlflow ui --port 5000
```

## Key Findings

1. **TensorFlow dominates NLP tasks** — 83.72% sentiment accuracy versus 50% for both NumPy and PyTorch, due to its superior built-in LSTM implementation that effectively handles vanishing gradients in long text sequences.
2. **PyTorch leads image classification** — 70.38% CIFAR-10 accuracy, slightly ahead of TensorFlow (68.57%) and significantly ahead of NumPy (56.75%).
3. **All frameworks converge for regression** — R2 scores within 1.4% of each other (0.781 to 0.795), suggesting framework choice matters less for simple tabular tasks.
4. **NumPy is prohibitively slow for complex tasks** — approximately 100 seconds per epoch for CIFAR-10 versus 25 seconds for PyTorch and 12 seconds for TensorFlow.
5. **Code maintainability is high across all implementations** — average maintainability index of 83.55 out of 100.
6. **Pure NumPy LSTM implementations are not viable for production NLP** — vanishing gradients prevent learning entirely, confirming the value of framework-level automatic differentiation.

## License

Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)

Copyright (c) 2024 Sazzad Hossain

You are free to use, share, modify, and build upon this work for non-commercial purposes with attribution. Commercial use is prohibited without explicit written permission from the author.

See [LICENSE](LICENSE) for full details.
