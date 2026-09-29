import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_results(result_path):
    """
    Load JSON results

    Parameters:
        result_path: Path to JSON file containing model evaluation results.

    Returns:
        A dictionary containing loaded results.
    """
    with open(result_path, "r") as file:
        return json.load(file)


def compare_models(result_paths):
    """
    Compare multiple models based on evaluation results.

    Parameters:
        result_paths: List of paths to JSON files containing model evaluation results.
    """
    results = [load_results(path) for path in result_paths]

    model_names = [result["model"] for result in results]

    metrics = {
        "Accuracy": [result["accuracy"] for result in results],
        "Precision": [result["precision"] for result in results],
        "Recall": [result["recall"] for result in results],
        "F1": [result["f1"] for result in results],
        "FAR": [result["far"] for result in results],
        "FRR": [result["frr"] for result in results],
    }

    x = np.arange(len(model_names))
    width = 0.12

    plt.figure(figsize=(12, 7))

    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(
            x + (i - 2.5) * width,
            values,
            width,
            label=metric_name,
        )

    plt.xticks(x, model_names)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    output_dir = PROJECT_ROOT / "evaluation" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_comparison.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Success! Model comparison graph saved to: {output_path}")


if __name__ == "__main__":
    results_dir = PROJECT_ROOT / "evaluation" / "results"

    result_files = list(results_dir.glob("*.json"))

    compare_models(result_files)
