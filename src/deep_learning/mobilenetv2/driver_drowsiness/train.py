import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.deep_learning.mobilenetv2.mobilenetv2 import MobileNetV2Trainer

"""
Train MobileNetV2 model for Driver Drowsiness dataset.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[4]

trainer = MobileNetV2Trainer(
    data_dir=PROJECT_ROOT / "data" / "Driver-Drowsiness-Dataset",
    model_path=(
        PROJECT_ROOT
        / "src"
        / "deep_learning"
        / "mobilenetv2"
        / "driver_drowsiness"
        / "model"
        / "mobilenetv2_driver_drowsiness.keras"
    ),
    results_dir=(
        PROJECT_ROOT / "training_results" / "mobilenetv2" / "driver_drowsiness"
    ),
    epochs=10,
)

trainer.load_data()
trainer.build_model()
trainer.train()
trainer.export_training_results()
trainer.evaluate()
trainer.save()
