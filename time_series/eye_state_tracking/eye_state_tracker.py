from pathlib import Path

import numpy as np
import tensorflow as tf

from config import MOBILENETV2_IMG_HEIGHT, MOBILENETV2_IMG_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "deep_learning"
    / "mobilenetv2"
    / "model"
    / "mobilenetv2_drowsiness.keras"
)

IMG_SIZE = (MOBILENETV2_IMG_WIDTH, MOBILENETV2_IMG_HEIGHT)

model = tf.keras.models.load_model(MODEL_PATH)


def predict_eye_state(image_path):

    image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(model.predict(image_array, verbose=0)[0][0])

    if probability >= 0.5:
        state = "sleepy"
    else:
        state = "awake"

    return state, probability


print("Eye State Tracker")
print("Model loaded successfully.")

state, probability = predict_eye_state(
    PROJECT_ROOT / "data" / "Driver-Drowsiness-Dataset" / "Non Drowsy" / "zc1680.png"
)
print(state)
print(probability)
