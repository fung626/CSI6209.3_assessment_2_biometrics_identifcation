import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def generate_confusion_matrix(result_path):
    """
    Generate and save a confusion matrix from evaluation results.

    Parameters:
        result_path: Path to the evaluation result JSON file.
    """

    result_path = Path(result_path)

    with open(result_path, "r") as file:
        results = json.load(file)

    cm = np.array(results["confusion_matrix"])

    model_name = result_path.stem.replace("_results", "")

    output_dir = PROJECT_ROOT / "evaluation" / "confusion_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_confusion_matrix.png"

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=results["classes"],
    )

    display.plot()

    plt.title(f"Confusion Matrix - {results['model']}")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    results_dir = PROJECT_ROOT / "evaluation" / "results"

    result_files = list(results_dir.glob("*.json"))

    for result_path in result_files:
        if result_path.exists():
            generate_confusion_matrix(result_path)
        else:
            print(f"Result file not found: {result_path}")
