from pathlib import Path

from evaluator import evaluate_model
from evaluator_hybrid import evaluate_hybrid_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "deep_learning"
    / "mobilenetv2"
    / "mrl_eye"
    / "model"
    / "mobilenetv2_mrl_eye.keras"
)

TEST_DIR = PROJECT_ROOT / "data" / "MRL-Eye-Dataset" / "test"


results = evaluate_model(
    model_path=MODEL_PATH,
    test_dir=TEST_DIR,
    result_path=PROJECT_ROOT / "evaluation" / "results" / "mrl_eye_results.json",
)

hybrid_results = evaluate_hybrid_model(
    model_path=MODEL_PATH,
    test_dir=TEST_DIR,
    result_path=PROJECT_ROOT
    / "evaluation"
    / "results"
    / "hybrid_mrl_eye_resultss.json",
)

print(results)
print(hybrid_results)
