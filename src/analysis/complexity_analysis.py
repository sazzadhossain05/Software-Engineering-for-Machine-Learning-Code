"""SE quality metric extraction using Radon.

This module wraps the Radon static analysis tool to extract cyclomatic
complexity and maintainability index metrics from Python source files.
These metrics form the quantitative backbone of the cross-framework
These metrics form the quantitative backbone of the cross-framework
comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def analyze_cyclomatic_complexity(target_path: str) -> list[dict[str, Any]]:
    """Run Radon cyclomatic complexity analysis on a target directory.

    Args:
        target_path: Path to directory or file to analyze.

    Returns:
        List of dicts with file, function, complexity, and rank.
    """
    result = subprocess.run(
        ["radon", "cc", target_path, "-a", "-j"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Radon CC warning: {result.stderr}", file=sys.stderr)

    try:
        raw = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        return []

    entries = []
    for filepath, blocks in raw.items():
        for block in blocks:
            entries.append({
                "file": filepath,
                "name": block.get("name", ""),
                "type": block.get("type", ""),
                "complexity": block.get("complexity", 0),
                "rank": block.get("rank", ""),
                "lineno": block.get("lineno", 0),
            })
    return entries


def analyze_maintainability_index(target_path: str) -> list[dict[str, Any]]:
    """Run Radon maintainability index analysis.

    Args:
        target_path: Path to directory or file to analyze.

    Returns:
        List of dicts with file, mi_score, and rank.
    """
    result = subprocess.run(
        ["radon", "mi", target_path, "-j"],
        capture_output=True, text=True
    )
    try:
        raw = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        return []

    entries = []
    for filepath, data in raw.items():
        entries.append({
            "file": filepath,
            "mi_score": data.get("mi", 0),
            "rank": data.get("rank", ""),
        })
    return entries


def analyze_raw_metrics(target_path: str) -> list[dict[str, Any]]:
    """Run Radon raw metrics (LOC, LLOC, SLOC, comments, etc.).

    Args:
        target_path: Path to directory or file to analyze.

    Returns:
        List of dicts with file and raw metric counts.
    """
    result = subprocess.run(
        ["radon", "raw", target_path, "-j"],
        capture_output=True, text=True
    )
    try:
        raw = json.loads(result.stdout) if result.stdout.strip() else {}
    except json.JSONDecodeError:
        return []

    entries = []
    for filepath, data in raw.items():
        entries.append({
            "file": filepath,
            "loc": data.get("loc", 0),
            "lloc": data.get("lloc", 0),
            "sloc": data.get("sloc", 0),
            "comments": data.get("comments", 0),
            "multi": data.get("multi", 0),
            "blank": data.get("blank", 0),
        })
    return entries


def run_full_analysis(target_path: str, output_path: str | None = None) -> dict:
    """Run all Radon analyses and optionally save to JSON.

    Args:
        target_path: Path to analyze.
        output_path: Optional path to save JSON results.

    Returns:
        Combined results dictionary.
    """
    results = {
        "target": target_path,
        "cyclomatic_complexity": analyze_cyclomatic_complexity(target_path),
        "maintainability_index": analyze_maintainability_index(target_path),
        "raw_metrics": analyze_raw_metrics(target_path),
    }

    # Compute summary statistics
    cc_values = [e["complexity"] for e in results["cyclomatic_complexity"]]
    mi_values = [e["mi_score"] for e in results["maintainability_index"]]

    results["summary"] = {
        "avg_cyclomatic_complexity": sum(cc_values) / len(cc_values) if cc_values else 0,
        "max_cyclomatic_complexity": max(cc_values) if cc_values else 0,
        "avg_maintainability_index": sum(mi_values) / len(mi_values) if mi_values else 0,
        "total_sloc": sum(e["sloc"] for e in results["raw_metrics"]),
        "total_files": len(results["raw_metrics"]),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SE complexity analysis")
    parser.add_argument("--target", required=True, help="Path to analyze")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    results = run_full_analysis(args.target, args.output)
    print(json.dumps(results["summary"], indent=2))
