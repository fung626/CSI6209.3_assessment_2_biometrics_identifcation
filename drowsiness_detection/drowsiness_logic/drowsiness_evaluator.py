import cv2
import numpy as np

from config import COLOR_RED_BGR, DEBUG
from utils.drawing_utils import draw_bold_text, draw_regular_text, draw_semibold_text

"""
The class for evaluating driver drowsiness based on face, eye, and yawn inputs.
"""


class DrowsinessEvaluator:

    def __init__(
        self,
        face_weight=0.3,
        eye_weight=0.7,
        threshold=0.55,
    ):
        self.w_face = face_weight
        self.w_eye = eye_weight
        self.threshold = threshold
        self.debug = DEBUG

    def evaluate(
        self,
        raw_face_pred,
        left_eye_pred=None,
        right_eye_pred=None,
        is_yawning=False,
        are_eyes_closed=False,
    ):
        """
        Evaluate function for computing final drowsiness score based on face, eye, and yawn inputs.

        Parameters:
                raw_face_pred: Raw prediction output from face drowsiness model.
                left_eye_pred: Raw prediction output from left eye drowsiness model.
                right_eye_pred: Raw prediction output from right eye drowsiness model.
                is_yawning: Boolean indicating if driver is yawning.
                are_eyes_closed: Boolean indicating if driver's eyes are closed.

        returns:
                final_drowsy_score: Computed final drowsiness score.
                is_drowsy: Boolean indicating if driver is considered drowsy.
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

        if is_yawning:
            final_drowsy_score = max(final_drowsy_score, 0.95)
        if are_eyes_closed:
            final_drowsy_score = max(final_drowsy_score, 0.98)

        # Print debug information if debug mode is enabled
        if self.debug:
            print(
                f"Fusion -> Face Drowsy: {face_drowsy_score:.2f} | Eye Score: {eye_score:.2f} | Final: {final_drowsy_score:.2f}"
            )

        # Compute final drowsiness status based on threshold
        is_drowsy = final_drowsy_score > self.threshold
        return final_drowsy_score, is_drowsy

    def draw_status(self, frame, final_score, is_drowsy, is_yawning, are_eyes_closed):
        """
        Draw status overlay indicating drowsiness and yawning.

        Parameters:
            frame: Input image frame.
            final_score: Final drowsiness score.
            is_drowsy: Boolean indicating if driver is considered drowsy.
            is_yawning: Boolean indicating if driver is yawning.
            are_eyes_closed: Boolean indicating if driver's eyes are closed.

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
        yawn_text = "YAWN DETECTED!" if is_yawning else "Normal"
        eye_text = "EYES CLOSED!" if are_eyes_closed else "Eyes Active"

        # Draw status text on overlay box
        frame = draw_bold_text(
            frame,
            "DRIVER MONITORING",
            (25, 22),
            font_size=14,
            color_bgr=(255, 255, 255),
        )
        frame = draw_semibold_text(
            frame, status_text, (25, 48), font_size=18, color_bgr=(255, 255, 255)
        )
        frame = draw_regular_text(
            frame, score_text, (25, 76), font_size=14, color_bgr=(200, 200, 200)
        )
        frame = draw_regular_text(
            frame,
            yawn_text,
            (25, 96),
            font_size=12,
            color_bgr=(0, 255, 255) if is_yawning else (180, 180, 180),
        )
        frame = draw_regular_text(
            frame,
            eye_text,
            (145, 96),
            font_size=12,
            color_bgr=(0, 0, 255) if are_eyes_closed else (180, 180, 180),
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
            frame = draw_bold_text(
                frame,
                "*** WARNING: WAKE UP! ***",
                (width // 2 - 140, height - 55),
                font_size=20,
                color_bgr=(255, 255, 255),
            )

        return frame

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
