import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MOBILENETV2_IMG_HEIGHT, MOBILENETV2_IMG_WIDTH
from realtime.webcam.webcam import main as run_webcam


def main():
    print("Initializing Driver Drowsiness Detection System...")

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(MOBILENETV2_IMG_WIDTH, MOBILENETV2_IMG_HEIGHT, 3),
    )

    print("MobileNetV2 loaded successfully!")
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    try:
        run_webcam()
    except KeyboardInterrupt:
        print("\nApplication gracefully stopped by user.")
    except Exception as e:
        print(f"\nError occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
