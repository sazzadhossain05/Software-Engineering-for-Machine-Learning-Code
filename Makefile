.PHONY: install test lint analyze train-all mlflow clean help

PYTHON := python
PYTEST := pytest
MLFLOW := mlflow

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	$(PYTHON) -m pip install -r requirements/base.txt
	$(PYTHON) -m pip install -r requirements/pytorch.txt
	$(PYTHON) -m pip install -r requirements/tensorflow.txt
	$(PYTHON) -m pip install -r requirements/dev.txt

test:  ## Run full test suite with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:  ## Run code quality checks
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	pylint src/ --rcfile=.pylintrc --fail-under=7.0

format:  ## Auto-format code
	black src/ tests/
	isort src/ tests/

analyze:  ## Run SE metric analysis on all implementations
	$(PYTHON) -m src.analysis.complexity_analysis --target src/numpy_impl/ --output results/numpy_complexity.json
	$(PYTHON) -m src.analysis.complexity_analysis --target src/pytorch_impl/ --output results/pytorch_complexity.json
	$(PYTHON) -m src.analysis.complexity_analysis --target src/tensorflow_impl/ --output results/tensorflow_complexity.json
	$(PYTHON) -m src.analysis.code_metrics --target src/ --output results/code_metrics.json
	$(PYTHON) -m src.analysis.visualization --input results/ --output docs/figures/

train-numpy-cifar10:  ## Train NumPy CIFAR-10
	$(PYTHON) -m src.numpy_impl.train --config configs/cifar10_numpy.yaml

train-pytorch-cifar10:  ## Train PyTorch CIFAR-10
	$(PYTHON) -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml

train-tf-cifar10:  ## Train TensorFlow CIFAR-10
	$(PYTHON) -m src.tensorflow_impl.train --config configs/cifar10_tensorflow.yaml

train-all:  ## Train all 9 implementations
	$(PYTHON) -m src.numpy_impl.train --config configs/cifar10_numpy.yaml
	$(PYTHON) -m src.numpy_impl.train --config configs/sentiment_numpy.yaml
	$(PYTHON) -m src.numpy_impl.train --config configs/regression_numpy.yaml
	$(PYTHON) -m src.pytorch_impl.train --config configs/cifar10_pytorch.yaml
	$(PYTHON) -m src.pytorch_impl.train --config configs/sentiment_pytorch.yaml
	$(PYTHON) -m src.pytorch_impl.train --config configs/regression_pytorch.yaml
	$(PYTHON) -m src.tensorflow_impl.train --config configs/cifar10_tensorflow.yaml
	$(PYTHON) -m src.tensorflow_impl.train --config configs/sentiment_tensorflow.yaml
	$(PYTHON) -m src.tensorflow_impl.train --config configs/regression_tensorflow.yaml

mlflow:  ## Launch MLflow tracking UI
	$(MLFLOW) ui --port 5000

clean:  ## Remove generated artifacts
	rm -rf results/*.json results/*.csv
	rm -rf docs/figures/*.png docs/figures/*.pdf
	rm -rf htmlcov/ .coverage
	rm -rf __pycache__ src/**/__pycache__ tests/**/__pycache__
	rm -rf mlruns/
	find . -name "*.pyc" -delete
