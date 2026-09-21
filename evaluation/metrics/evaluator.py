import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_model(
    model_path, test_dir, image_size=(224, 224), result_path=None, batch_size=32
):
    """
    Evaluate a TensorFlow classification model.

    Parameters:
        model_path: Path to the trained model.
        test_dir: Path to the test dataset.
        image_size: Image size expected by the model.
        batch_size: Batch size used for evaluation.

    Returns:
        Dictionary containing evaluation metrics.
    """

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    model = tf.keras.models.load_model(model_path)

    all_labels = []
    all_preds = []

    for images, labels in test_dataset:
        probabilities = model.predict(images, verbose=0)
        preds = (probabilities >= 0.5).astype(int)

        all_labels.extend(labels.numpy().flatten())
        all_preds.extend(preds.flatten())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0,
    )
    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0,
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0,
    )

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": Path(model_path).stem,
        "model_path": str(Path(model_path).relative_to(PROJECT_ROOT)),
        "dataset": Path(test_dir).parent.name,
        "test_dir": str(Path(test_dir).relative_to(PROJECT_ROOT)),
        "classes": test_dataset.class_names,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "far": float(far),
        "frr": float(frr),
    }

    if result_path is not None:
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)

        with open(result_path, "w") as file:
            json.dump(results, file, indent=4)

    print(f"Evaluation results saved to: {result_path}")

    return results
