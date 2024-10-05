"""Custom SE code metrics beyond Radon's built-in capabilities.

Measures API surface area, cognitive complexity estimation, and module
coupling — metrics specifically chosen for cross-framework comparison.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def count_api_calls(filepath: str, framework_modules: list[str]) -> dict[str, int]:
    """Count distinct framework API calls in a Python file.

    This measures how many unique framework functions/methods a file depends on,
    indicating the breadth of framework API coupling.

    Args:
        filepath: Path to Python source file.
        framework_modules: List of module prefixes to count (e.g., ['torch', 'torch.nn']).

    Returns:
        Dictionary mapping API names to call counts.
    """
    with open(filepath, "r") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return {}

    api_calls: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # Build dotted name
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            full_name = ".".join(reversed(parts))

            for module in framework_modules:
                if full_name.startswith(module):
                    api_calls[full_name] = api_calls.get(full_name, 0) + 1

    return api_calls


def measure_file_metrics(filepath: str) -> dict[str, Any]:
    """Compute custom SE metrics for a single file.

    Args:
        filepath: Path to Python source file.

    Returns:
        Dictionary of metric name to value.
    """
    with open(filepath, "r") as f:
        source = f.read()
        lines = source.split("\n")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"file": filepath, "parse_error": True}

    # Count functions, classes, imports
    n_functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    n_classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    n_imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))

    # Docstring coverage
    func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    n_with_docstring = 0
    for node in func_nodes:
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
            n_with_docstring += 1
    docstring_ratio = n_with_docstring / len(func_nodes) if func_nodes else 1.0

    # Framework API surface
    numpy_apis = count_api_calls(filepath, ["np", "numpy"])
    torch_apis = count_api_calls(filepath, ["torch", "nn", "F"])
    tf_apis = count_api_calls(filepath, ["tf", "keras", "layers"])

    return {
        "file": filepath,
        "total_lines": len(lines),
        "n_functions": n_functions,
        "n_classes": n_classes,
        "n_imports": n_imports,
        "docstring_coverage": round(docstring_ratio, 3),
        "numpy_api_calls": len(numpy_apis),
        "torch_api_calls": len(torch_apis),
        "tf_api_calls": len(tf_apis),
        "unique_numpy_apis": list(numpy_apis.keys()),
        "unique_torch_apis": list(torch_apis.keys()),
        "unique_tf_apis": list(tf_apis.keys()),
    }


def analyze_directory(target_path: str, output_path: str | None = None) -> list[dict]:
    """Analyze all Python files in a directory.

    Args:
        target_path: Root directory to analyze.
        output_path: Optional path to save JSON results.

    Returns:
        List of per-file metric dictionaries.
    """
    results = []
    for py_file in sorted(Path(target_path).rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        metrics = measure_file_metrics(str(py_file))
        results.append(metrics)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run custom code metrics analysis")
    parser.add_argument("--target", required=True, help="Directory to analyze")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    results = analyze_directory(args.target, args.output)
    print(f"Analyzed {len(results)} files")
