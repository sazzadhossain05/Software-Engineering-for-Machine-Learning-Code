"""Comparative visualization of SE metrics across frameworks.

Generates radar charts, bar plots, and comparison tables for
SE metric analysis across frameworks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_radar_chart(
    metrics: dict[str, dict[str, float]], output_path: str, title: str = "SE Quality Comparison"
) -> None:
    """Create a radar chart comparing SE metrics across frameworks.

    Args:
        metrics: {framework_name: {metric_name: value}} — values normalized to [0, 1].
        output_path: Path to save the figure.
        title: Chart title.
    """
    categories = list(next(iter(metrics.values())).keys())
    n = len(categories)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {"numpy": "#1f77b4", "pytorch": "#ff7f0e", "tensorflow": "#2ca02c"}

    for framework, vals in metrics.items():
        values = [vals[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=framework, color=colors.get(framework, None))
        ax.fill(angles, values, alpha=0.1, color=colors.get(framework, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar chart saved to {output_path}")


def create_bar_comparison(
    data: dict[str, dict[str, float]], output_path: str, title: str = "Metric Comparison"
) -> None:
    """Create grouped bar chart for metric comparison.

    Args:
        data: {framework: {metric: value}}.
        output_path: Path to save figure.
        title: Chart title.
    """
    frameworks = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fw in enumerate(frameworks):
        values = [data[fw][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=fw)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SE metric visualizations")
    parser.add_argument("--input", required=True, help="Results directory")
    parser.add_argument("--output", required=True, help="Output directory for figures")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load results and generate charts
    results_dir = Path(args.input)
    framework_summaries = {}
    for json_file in results_dir.glob("*_complexity.json"):
        with open(json_file) as f:
            data = json.load(f)
        fw_name = json_file.stem.replace("_complexity", "")
        framework_summaries[fw_name] = data.get("summary", {})

    if framework_summaries:
        print(f"Loaded results for: {list(framework_summaries.keys())}")
    else:
        print("No results found. Run 'make analyze' first.")
