from pathlib import Path
from evaluator import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "deep_learning"
    / "mobilenetv2"
    / "mrl_eye"
    / "model"
    / "mobilenetv2_mrl_eye.keras"
)

TEST_DIR = PROJECT_ROOT / "data" / "MRL-Eye-Dataset" / "test"

RESULT_PATH = PROJECT_ROOT / "evaluation" / "results" / "mrl_eye_results.json"

results = evaluate_model(
    model_path=MODEL_PATH,
    test_dir=TEST_DIR,
    result_path=RESULT_PATH,
)

print(results)
