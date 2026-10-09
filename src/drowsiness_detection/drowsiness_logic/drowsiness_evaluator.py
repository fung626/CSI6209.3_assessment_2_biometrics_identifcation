from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import (
    COLOR_BROWN_RGB,
    COLOR_DARK_GRAY_RGB,
    COLOR_DARK_RED_RGB,
    COLOR_GRAY_RGB,
    COLOR_LIGHT_GREEN_RGB,
    COLOR_LIGHT_ORANGE_RGB,
    COLOR_LIGHT_RED_RGB,
    COLOR_LIGHT_YELLOW_RGB,
    COLOR_OLIVE_RGB,
    COLOR_WHITE_RGB,
    DEBUG,
)

# from src.utils.drawing_utils import draw_bold_text, draw_regular_text, draw_semibold_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

        Returns:
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

    def draw_status(
        self,
        frame,
        is_face_detected=False,
        final_score=0.0,
        is_drowsy=False,
        is_yawning=False,
        are_eyes_closed=False,
        perclos=0.0,
    ):
        """
        Draw status card on the top-right corner
        """

        # print(
        #     f"Draw Status -> Face Detected: {is_face_detected} | Final Score: {final_score:.2f} | Drowsy: {is_drowsy} | Yawning: {is_yawning} | Eyes Closed: {are_eyes_closed} | PERCLOS: {perclos:.1f}%"
        # )

        try:
            _, width, _ = frame.shape

            w = 260
            h = 100
            x = width - w - 25
            y = 25
            corner_radius = 16

            bg_color_rgb = (24, 24, 24)
            border_line_color = (
                60,
                60,
                60,
            )
            text_muted_rgb = COLOR_GRAY_RGB
            text_white_rgb = COLOR_WHITE_RGB
            green_accent_rgb = COLOR_LIGHT_GREEN_RGB
            alert_red_rgb = COLOR_LIGHT_RED_RGB

            if is_face_detected == False:
                risk_label = "NONE"
                badge_bg = COLOR_DARK_GRAY_RGB
                badge_width = 70
                badge_text_color = COLOR_WHITE_RGB
            elif is_drowsy or final_score >= 0.7:
                risk_label = "HIGH"
                badge_bg = COLOR_DARK_RED_RGB
                badge_width = 52
                badge_text_color = COLOR_WHITE_RGB
            elif final_score >= 0.3:
                risk_label = "MED"
                badge_bg = COLOR_LIGHT_ORANGE_RGB
                badge_width = 50
                badge_text_color = COLOR_BROWN_RGB
            else:
                risk_label = "LOW"
                badge_bg = COLOR_LIGHT_YELLOW_RGB
                badge_width = 50
                badge_text_color = COLOR_OLIVE_RGB

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            draw.rounded_rectangle(
                [(x, y), (x + w, y + h)],
                radius=corner_radius,
                fill=bg_color_rgb,
            )

            font_semi_path = (
                PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Semibold.ttf"
            )
            font_bold_path = (
                PROJECT_ROOT / "assets" / "fonts" / "SF-Pro-Display-Bold.ttf"
            )

            try:
                font_header_label = ImageFont.truetype(str(font_semi_path), 14)
                font_badge = ImageFont.truetype(str(font_bold_path), 14)
                font_col_label = ImageFont.truetype(str(font_semi_path), 12)
                font_col_val = ImageFont.truetype(str(font_bold_path), 12)
            except Exception:
                font_header_label = font_badge = font_col_label = font_col_val = (
                    ImageFont.load_default()
                )

            header_height = 38
            header_bottom_y = y + header_height
            draw.line(
                [(x, header_bottom_y), (x + w, header_bottom_y)],
                fill=border_line_color,
                width=1,
            )

            draw.text(
                (x + 12, y + 12),
                "DROWSINESS LEVEL",
                font=font_header_label,
                fill=text_muted_rgb,
            )

            badge_w, badge_h = badge_width, 18
            badge_x = x + w - badge_w - 10
            badge_y = y + 10
            draw.rounded_rectangle(
                [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
                radius=8,
                fill=badge_bg,
            )
            bbox = draw.textbbox((0, 0), risk_label, font=font_badge)
            bx = badge_x + (badge_w - (bbox[2] - bbox[0])) // 2 - bbox[0]
            by = badge_y + (badge_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
            draw.text((bx, by), risk_label, font=font_badge, fill=badge_text_color)

            # ==========================================
            # 3-COLUMN GRID SECTION (Yawn | Eyes | PERCLOS)
            # ==========================================
            col_w = w // 3
            grid_top_y = header_bottom_y + 4

            draw.line(
                [(x + col_w, grid_top_y), (x + col_w, y + h - 6)],
                fill=border_line_color,
                width=1,
            )
            draw.line(
                [(x + 2 * col_w, grid_top_y), (x + 2 * col_w, y + h - 6)],
                fill=border_line_color,
                width=1,
            )

            yawn_val = "N/A" if not is_face_detected else "YES" if is_yawning else "NO"
            yawn_color = (
                alert_red_rgb
                if is_yawning
                else green_accent_rgb if is_face_detected else text_muted_rgb
            )

            eyes_val = (
                "N/A"
                if not is_face_detected
                else "CLOSED" if are_eyes_closed else "OPEN"
            )
            eyes_color = (
                alert_red_rgb
                if are_eyes_closed
                else green_accent_rgb if is_face_detected else text_muted_rgb
            )

            perclos_val = "N/A" if not is_face_detected else f"{perclos:.1f}%"
            perclos_color = (
                alert_red_rgb
                if perclos > 50
                else text_white_rgb if is_face_detected else text_muted_rgb
            )

            columns = [
                ("YAWN", yawn_val, yawn_color),
                ("EYES", eyes_val, eyes_color),
                ("PERCLOS", perclos_val, perclos_color),
            ]

            # Render each column
            for i, (col_label, col_val, val_color) in enumerate(columns):
                c_start_x = x + i * col_w
                try:
                    lbl_bbox = draw.textbbox((0, 0), col_label, font=font_col_label)
                    lbl_w = lbl_bbox[2] - lbl_bbox[0]
                    lbl_x = c_start_x + (col_w - lbl_w) // 2 - lbl_bbox[0]
                except Exception:
                    lbl_x = c_start_x + 12

                draw.text(
                    (lbl_x, grid_top_y + 6),
                    col_label,
                    font=font_col_label,
                    fill=text_muted_rgb,
                )

                try:
                    val_bbox = draw.textbbox((0, 0), col_val, font=font_col_val)
                    val_w = val_bbox[2] - val_bbox[0]
                    val_x = c_start_x + (col_w - val_w) // 2 - val_bbox[0]
                except Exception:
                    val_x = c_start_x + 12

                draw.text(
                    (val_x, grid_top_y + 22),
                    col_val,
                    font=font_col_val,
                    fill=val_color,
                )
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            if self.debug:
                print(f"Error drawing risk badge: {e}")

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
