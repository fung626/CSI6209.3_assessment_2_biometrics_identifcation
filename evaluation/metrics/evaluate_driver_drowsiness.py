from pathlib import Path
from evaluator import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "deep_learning"
    / "mobilenetv2"
    / "driver_drowsiness"
    / "model"
    / "mobilenetv2_driver_drowsiness.keras"
)

TEST_DIR = PROJECT_ROOT / "data" / "Driver-Drowsiness-Dataset" / "test"

RESULT_PATH = PROJECT_ROOT / "evaluation" / "results" / "driver_drowsiness_results.json"

results = evaluate_model(
    model_path=MODEL_PATH,
    test_dir=TEST_DIR,
    result_path=RESULT_PATH,
)

print(results)
