from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import COLOR_GREEN_BGR, DEBUG, MOBILENETV2_IMG_HEIGHT, MOBILENETV2_IMG_WIDTH


class EyePredictor:

    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]

        model_path = (
            project_root
            / "deep_learning"
            / "mobilenetv2"
            / "mrl_eye"
            / "model"
            / "mobilenetv2_mrl_eye.keras"
        )
        self.debug = DEBUG
        self.model = tf.keras.models.load_model(model_path)

        self.image_size = (MOBILENETV2_IMG_WIDTH, MOBILENETV2_IMG_HEIGHT)

        if self.debug:
            print("=" * 60)
            print("EyePredictor initialized")
            print("Classes (alphabetical order from image_dataset_from_directory):")
            print("  - awake (index 0)")
            print("  - sleepy (index 1)")
            print("Model output interpretation:")
            print("  If output < 0.5 → Awake")
            print("  If output ≥ 0.5 → Sleepy")
            print("=" * 60)

    def predict(self, eye):

        # Resize eye to model input size
        eye = cv2.resize(eye, self.image_size)

        # OpenCV BGR -> RGB
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        eye = np.expand_dims(eye, axis=0)

        # Model prediction
        prediction = self.model.predict(eye, verbose=0)[0][0]

        # 0 = Awake 1 = Sleepy
        if prediction >= 0.5:
            label = "Sleepy"
            confidence = prediction
        else:
            label = "Awake"
            confidence = 1 - prediction

        if self.debug:
            print(
                f"EyePredictor: "
                f"Raw eye drowsiness label is {label} "
                f"with confidence {confidence * 100:.1f}% "
                f"and prediction {prediction}"
            )

        return label, float(confidence)

    def draw_predictions(self, frame, eye_box):

        x_min, y_min, x_max, y_max = eye_box

        # Crop eye from webcam frame
        eye = frame[y_min:y_max, x_min:x_max]

        # Check that crop is valid
        if eye.size == 0:
            return

        if self.debug:
            # cv2.imshow("Eye Input", eye)
            cv2.imwrite(
                str(
                    Path(__file__).resolve().parents[3]
                    / "results"
                    / "debug"
                    / "debug_eye.jpg"
                ),
                eye,
            )

        # Predict
        label, confidence = self.predict(eye)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            COLOR_GREEN_BGR,
            2,
        )

        # Convert confidence to percentage
        if self.debug:
            print(
                f"EyePredictor: Drawing prediction '{label}' "
                f"with confidence {confidence * 100:.1f}% "
                f"for eye box {eye_box}"
            )
        text = f"{label}: {confidence * 100:.1f}%"

        # Draw prediction
        cv2.putText(
            frame,
            text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_GREEN_BGR,
            2,
        )

    def extract_eye(self, frame, eye_box):
        x_min, y_min, x_max, y_max = eye_box
        eye = frame[y_min:y_max, x_min:x_max]
        return eye
