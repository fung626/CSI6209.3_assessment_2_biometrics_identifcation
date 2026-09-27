from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import COLOR_GREEN_BGR, DEBUG, MOBILENETV2_IMG_HEIGHT, MOBILENETV2_IMG_WIDTH

"""
Face predictor for Driver Drowsiness dataset using a trained MobileNetV2 model.
"""


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
        """
        Predict function for determining drowsiness of detected face.
        Returns:
            label (str): Predicted label ("Drowsy" or "Non Drowsy").
            confidence (float): Confidence score for the prediction.
        """

        # Resize face to model input size
        face = cv2.resize(face, self.image_size)

        # OpenCV BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        face = np.expand_dims(face, axis=0)

        # Get model prediction value for given face
        prediction = self.model.predict(face, verbose=0)[0][0]

        # 0 = Drowsy, 1 = Non Drowsy
        # Set prediction threshold for Non Drowsy vs Drowsy
        if prediction >= 0.6:
            label = "Non Drowsy"
            confidence = prediction
        else:
            label = "Drowsy"
            confidence = 1 - prediction

        # Print debug information if debug mode is enabled
        if self.debug:
            print(
                f"FacePredictor: "
                f"label is {label} "
                f"with confidence {confidence * 100:.1f}% "
                f"and prediction {prediction}"
            )

        return label, float(confidence)

    def draw_prediction(self, frame, face_box):
        """
        Draw prediction label and confidence.
        """

        # Extract coordinates from face bounding box
        x_min, y_min, x_max, y_max = face_box

        # Crop face from webcam frame using bounding box coordinates
        face = frame[y_min:y_max, x_min:x_max]

        # Check that crop is valid
        if face.size == 0:
            return

        # Print debug information if debug mode is enabled
        if self.debug:
            # Show or save cropped face for debugging purposes
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

        # Get label and confidence for the cropped face
        label, confidence = self.predict(face)

        # Draw face bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            COLOR_GREEN_BGR,
            2,
        )

        # Text for displaying prediction on the frame
        text = f"{label}: {confidence * 100:.1f}%"

        # Draw prediction text on the frame
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
        """
        Extract and return face region from frame using bounding box coordinates.
        """
        x_min, y_min, x_max, y_max = face_box
        face = frame[y_min:y_max, x_min:x_max]
        return face
