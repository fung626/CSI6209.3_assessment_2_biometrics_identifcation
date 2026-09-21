from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import COLOR_GREEN_BGR, DEBUG, MOBILENETV2_IMG_HEIGHT, MOBILENETV2_IMG_WIDTH


class FacePredictor:

    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]

        model_path = (
            project_root
            / "deep_learning"
            / "mobilenetv2"
            / "driver_drowsiness"
            / "model"
            / "mobilenetv2_driver_drowsiness.keras"
        )

        self.debug = DEBUG
        self.model = tf.keras.models.load_model(model_path)

        self.image_size = (MOBILENETV2_IMG_WIDTH, MOBILENETV2_IMG_HEIGHT)

    def predict(self, face):

        # Resize face to model input size
        face = cv2.resize(face, self.image_size)

        # OpenCV BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # The model performs MobileNetV2 preprocessing internally.

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Model prediction
        prediction = self.model.predict(face, verbose=0)[0][0]

        # 0 = Drowsy
        # 1 = Non Drowsy
        if prediction >= 0.6:
            label = "Non Drowsy"
            confidence = prediction
        else:
            label = "Drowsy"
            confidence = 1 - prediction

        if self.debug:
            print(
                f"FacePredictor: "
                f"label is {label} "
                f"with confidence {confidence * 100:.1f}% "
                f"and prediction {prediction}"
            )

        return label, float(confidence)

    def draw_prediction(self, frame, face_box):

        x_min, y_min, x_max, y_max = face_box

        # Crop face from webcam frame
        face = frame[y_min:y_max, x_min:x_max]

        # Check that crop is valid
        if face.size == 0:
            return

        if self.debug:
            # cv2.imshow("Face Input", face)
            cv2.imwrite(
                str(
                    Path(__file__).resolve().parents[3]
                    / "results"
                    / "debug"
                    / "debug_face.jpg"
                ),
                face,
            )

        # Predict
        label, confidence = self.predict(face)

        # Draw face bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            COLOR_GREEN_BGR,
            2,
        )

        text = f"{label}: {confidence * 100:.1f}%"

        cv2.putText(
            frame,
            text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_GREEN_BGR,
            2,
        )

    def extract_face(self, frame, face_box):
        x_min, y_min, x_max, y_max = face_box
        face = frame[y_min:y_max, x_min:x_max]
        return face
