from pathlib import Path

from evaluator import evaluate_model
from evaluator_hybrid import evaluate_hybrid_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "deep_learning"
    / "mobilenetv2"
    / "driver_drowsiness"
    / "model"
    / "mobilenetv2_driver_drowsiness.keras"
)

TEST_DIR = PROJECT_ROOT / "data" / "Driver-Drowsiness-Dataset" / "test"

standalone_results = evaluate_model(
    model_path=MODEL_PATH,
    test_dir=TEST_DIR,
    result_path=PROJECT_ROOT
    / "evaluation"
    / "results"
    / "driver_drowsiness_results.json",
)

hybrid_results = evaluate_hybrid_model(
    model_path=MODEL_PATH,
    test_dir=TEST_DIR,
    result_path=PROJECT_ROOT
    / "evaluation"
    / "results"
    / "hybrid_driver_drowsiness_results.json",
)

print(standalone_results)
print(hybrid_results)
