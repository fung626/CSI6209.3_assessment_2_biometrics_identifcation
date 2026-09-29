import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import auc, roc_curve

from config import MOBILENETV2_IMG_HEIGHT, MOBILENETV2_IMG_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

"""
Generate Receiver Operating Characteristic (ROC) curve for model evaluation result.
"""


def generate_roc_curve(result_path):
    """
    Generate the ROC curve for the given evaluation result JSON file.

    Parameters:
        result_path: Path to JSON file containing evaluation results.
    """

    result_path = Path(result_path)

    with open(result_path, "r") as file:
        results = json.load(file)

    # Initialize the test dataset and load the model
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        results["test_dir"],
        image_size=(MOBILENETV2_IMG_WIDTH, MOBILENETV2_IMG_HEIGHT),
        batch_size=32,
        label_mode="binary",
        shuffle=False,
    )
    model = tf.keras.models.load_model(results["model_path"])

    all_labels = []
    all_probs = []

    print("Gathering model prediction probabilities...")
    # Iterate through test dataset and collect model prediction probabilities
    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy().flatten())
        all_probs.extend(probabilities.flatten())

    # Compute ROC curve and AUC score
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Equivalent to FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.title(f"Receiver Operating Characteristic (ROC) - {results['model']}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(f"evaluation/roc_curve/{results['model']}_roc_curve.png", dpi=300)
    print(f"Success! ROC Curve graph saved. Computed AUC score: {roc_auc:.4f}")


if __name__ == "__main__":
    results_dir = PROJECT_ROOT / "evaluation" / "results"

    result_files = list(results_dir.glob("*.json"))

    for result_path in result_files:
        if result_path.exists():
            generate_roc_curve(result_path)
        else:
            print(f"Result file not found: {result_path}")
