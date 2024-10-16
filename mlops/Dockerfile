FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/ requirements/
RUN pip install --no-cache-dir \
    -r requirements/base.txt \
    -r requirements/pytorch.txt \
    -r requirements/dev.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY pyproject.toml .

# Default: run tests
CMD ["pytest", "tests/", "-v", "--cov=src"]
