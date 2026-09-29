import cv2
import numpy as np

from config import COLOR_RED_BGR, DEBUG

"""
The class for evaluating driver drowsiness based on face, eye, and yawn inputs.
"""


class DrowsinessEvaluator:

    def __init__(
        self, face_weight=0.3, eye_weight=0.7, threshold=0.55, yawn_threshold=0.5
    ):
        self.w_face = face_weight
        self.w_eye = eye_weight
        self.threshold = threshold
        self.yawn_threshold = yawn_threshold
        self.debug = DEBUG

    def evaluate(
        self, raw_face_pred, left_eye_pred=None, right_eye_pred=None, mar_score=0.0
    ):
        """
        Evaluate function for computing final drowsiness score based on face, eye, and yawn inputs.

        Parameters:
                raw_face_pred: Raw prediction output from face drowsiness model.
                left_eye_pred: Raw prediction output from left eye drowsiness model.
                right_eye_pred: Raw prediction output from right eye drowsiness model.
                mar_score: Mouth aspect ratio score for yawn detection.

        returns:
                final_drowsy_score: Computed final drowsiness score.
                is_drowsy: Boolean indicating if the driver is considered drowsy.
        """

        # Parse face prediction to obtain drowsiness score
        face_val = (
            raw_face_pred[1]
            if isinstance(raw_face_pred, (list, tuple, np.ndarray))
            else raw_face_pred
        )
        face_drowsy_score = 1.0 - face_val

        # Check both left and right eye predictions to compute eye drowsiness score
        # Set a default eye score in case one or both eye predictions are missing
        eye_score = 0.5
        if left_eye_pred is not None and right_eye_pred is not None:
            left_val = self._parse_drowsiness_score(left_eye_pred)
            right_val = self._parse_drowsiness_score(right_eye_pred)
            eye_score = min(left_val, right_val)
        final_drowsy_score = (face_drowsy_score * self.w_face) + (
            eye_score * self.w_eye
        )

        # If driver is yawning, force a high drowsiness score
        is_yawning = mar_score > self.yawn_threshold
        if is_yawning:
            final_drowsy_score = max(final_drowsy_score, 0.95)

        # Print debug information if debug mode is enabled
        if self.debug:
            print(
                f"Fusion -> Face Drowsy: {face_drowsy_score:.2f} | Eye Score: {eye_score:.2f} | Final: {final_drowsy_score:.2f}"
            )

        # Compute final drowsiness status based on threshold
        is_drowsy = final_drowsy_score > self.threshold
        return final_drowsy_score, is_drowsy

    def draw_status(self, frame, final_score, is_drowsy, is_yawning):
        """
        Draw status overlay indicating drowsiness and yawning.

        Parameters:
            frame: Input image frame.
            final_score: Final drowsiness score.
            is_drowsy: Boolean indicating if driver is considered drowsy.
            is_yawning: Boolean indicating if driver is yawning.
        """

        # Get frame dimensions
        height, width, _ = frame.shape

        # Define bounding box coordinates for status overlay
        box_x1, box_y1 = 15, 15
        box_x2, box_y2 = 285, 120

        # Determine background and border colors based on drowsiness status
        bg_color = COLOR_RED_BGR if is_drowsy else (40, 40, 40)  # BGR
        border_color = (0, 0, 255) if is_drowsy else (0, 255, 0)  # BGR

        # Draw status overlay box with background and border
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)  # Filled
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), border_color, 2)

        # Setup all status text strings
        status_text = "STATUS: DROWSY" if is_drowsy else "STATUS: AWAKE"
        score_text = f"Score: {final_score:.2f}"
        yawn_alert_text = "YAWN DETECTED!" if is_yawning else "Normal"

        # Draw status text on overlay box
        cv2.putText(
            frame,
            "DRIVER MONITORING SYSTEM",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            status_text,
            (25, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            score_text,
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            yawn_alert_text,
            (25, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255) if is_yawning else (150, 150, 150),
            1,
        )

        # Draw warning box and text if drowsy detected
        if is_drowsy:
            warning_box_y1 = height - 80
            warning_box_y2 = height - 20
            cv2.rectangle(
                frame,
                (50, warning_box_y1),
                (width - 50, warning_box_y2),
                COLOR_RED_BGR,
                -1,
            )
            cv2.putText(
                frame,
                "*** WARNING: WAKE UP! ***",
                (width // 2 - 180, height - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    def _parse_drowsiness_score(self, raw_pred):
        """
        Converts any predictor output (tuple or raw float) into a unified 0.0 to 1.0 Drowsiness Score (Higher = Drowsy).
        """
        # Parse raw predictor output into a standardized drowsiness score
        if isinstance(raw_pred, (list, tuple, np.ndarray)):
            label = str(raw_pred[0]).lower()
            val = float(raw_pred[1])

            # If model detects Awake, invert confidence so it becomes a low drowsiness score
            if "awake" in label:
                return 1.0 - val
            else:  # "sleepy" or "drowsy"
                return val
        else:
            return float(raw_pred)
