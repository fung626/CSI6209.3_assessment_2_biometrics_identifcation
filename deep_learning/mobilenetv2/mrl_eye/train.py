import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from deep_learning.mobilenetv2.mobilenetv2 import MobileNetV2Trainer

"""
Train MobileNetV2 model for MRL-Eye dataset.
"""

trainer = MobileNetV2Trainer(
    data_dir=PROJECT_ROOT / "data" / "MRL-Eye-Dataset",
    model_path=(
        PROJECT_ROOT
        / "deep_learning"
        / "mobilenetv2"
        / "mrl_eye"
        / "model"
        / "mobilenetv2_mrl_eye.keras"
    ),
    results_dir=(
        PROJECT_ROOT / "results" / "training_results" / "mobilenetv2" / "mrl_eye"
    ),
    epochs=10,
)

trainer.load_data()
trainer.build_model()
trainer.train()
trainer.export_training_results()
trainer.evaluate()
trainer.save()
