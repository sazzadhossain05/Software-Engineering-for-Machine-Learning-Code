"""Test coverage report aggregation across framework implementations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_coverage_for_module(module_path: str, test_path: str) -> dict:
    """Run pytest with coverage for a specific module."""
    result = subprocess.run(
        ["pytest", test_path, f"--cov={module_path}", "--cov-report=json", "-q"],
        capture_output=True, text=True
    )
    cov_file = Path("coverage.json")
    if cov_file.exists():
        with open(cov_file) as f:
            return json.load(f)
    return {"error": result.stderr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/coverage_report.json")
    args = parser.parse_args()
    print("Run 'make test' to generate coverage reports.")
