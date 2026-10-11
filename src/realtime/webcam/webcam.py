import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.computer_vision.eye_detection.eye_detector import EyeDetector
from src.computer_vision.facial_landmarks.facial_landmark_detector import (
    FacialLandmarkDetector,
)
from src.deep_learning.mobilenetv2.driver_drowsiness.face_predictor import FacePredictor
from src.deep_learning.mobilenetv2.mrl_eye.eye_predictor import EyePredictor
from src.drowsiness_detection.drowsiness_logic.drowsiness_evaluator import (
    DrowsinessEvaluator,
)
from src.drowsiness_detection.eye_closure_detection.eye_closure_detector import (
    EyeClosureDetector,
)
from src.drowsiness_detection.yawn_detection.yawn_detector import YawnDetector
from src.realtime.alert_system.buzzer_alert import BuzzerAlert
from src.utils.drawing_utils import draw_live_badge, draw_timestamp_badge


def safe_float(val, default=0.0):
    """Dynamically scans tuples, lists, or values to find and return the first valid number."""
    if isinstance(val, (tuple, list)):
        for item in val:
            try:
                return float(item)
            except (TypeError, ValueError):
                continue
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def main():
    buzzer_alert = BuzzerAlert()

    face_detector = FacialLandmarkDetector()
    face_predictor = FacePredictor()

    eye_detector = EyeDetector()
    eye_predictor = EyePredictor()

    eye_closure_detector = EyeClosureDetector(threshold_seconds=1.5)
    yawn_detector = YawnDetector(threshold_seconds=1.0)

    evaluator = DrowsinessEvaluator(face_weight=0.2, eye_weight=0.8, threshold=0.5)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Webcam started. Press Q to quit and export session results.")

    evaluating_hybrid_model = True
    valid_face_frames = 0
    alert_frame_count = 0
    drowsy_frame_count = 0
    cnn_triggers = 0
    ear_triggers = 0
    mar_triggers = 0
    ear_values = []
    mar_values = []

    try:
        while True:
            window_name = "Driver Drowsiness Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            ret, frame = cap.read()

            if not ret:
                print("Error: Cannot read webcam frame")
                break

            is_face_detected = False
            face_landmarks = face_detector.detect(frame)
            final_score = 0.0
            is_drowsy = False
            is_yawning = False
            are_eyes_closed = False
            perclos = 0.0
            duration_threshold = 2.0

            if face_landmarks is not None:
                is_face_detected = True
                valid_face_frames += 1

                face_box = face_detector.get_face_box(frame, face_landmarks)
                raw_mar = face_detector.get_mar(face_landmarks)
                mar_score = safe_float(raw_mar)

                face_detector.draw_indicator(frame, face_landmarks)

                mouth_landmarks = [
                    face_landmarks[i]
                    for i in FacialLandmarkDetector.MOUTH_INDICES
                    if i < len(face_landmarks)
                ]

                is_yawning = mar_score > 0.5
                mar_percentage = min(100.0, (mar_score / 0.5) * 100.0)
                mouth_label = f"{mar_percentage:.1f}% is {'YAWNING' if is_yawning else 'NOT YAWNING'}"
                face_detector.draw_indicator(
                    frame,
                    mouth_landmarks,
                    label=mouth_label,
                )

                face_predictor.draw_prediction(frame, face_box)
                eye_boxes = eye_detector.detect(frame, face_landmarks)

                if eye_boxes:
                    eye_detector.draw_indicators(frame, eye_boxes)
                    for eye_box in eye_boxes:
                        eye_predictor.draw_predictions(frame, eye_box)

                raw_face_pred = face_predictor.predict(
                    face_predictor.extract_face(frame, face_box)
                )
                face_pred_val = safe_float(raw_face_pred)
                print(f"Raw Face Prediction Value: {raw_face_pred}")

                if face_pred_val >= 0.5:
                    print("CNN Triggered")
                    cnn_triggers += 1

                left_pred, right_pred = None, None
                if eye_boxes and len(eye_boxes) >= 2:
                    left_pred = eye_predictor.predict(
                        eye_predictor.extract_eye(frame, eye_boxes[0])
                    )
                    right_pred = eye_predictor.predict(
                        eye_predictor.extract_eye(frame, eye_boxes[1])
                    )

                left_ear_score = safe_float(
                    eye_detector.get_ear(eye_detector.LEFT_EYE, face_landmarks)
                )
                right_ear_score = safe_float(
                    eye_detector.get_ear(eye_detector.RIGHT_EYE, face_landmarks)
                )

                avg_ear = (left_ear_score + right_ear_score) / 2.0
                ear_values.append(avg_ear)
                mar_values.append(mar_score)

                are_eyes_closed, closure_duration, perclos = (
                    eye_closure_detector.update(
                        left_ear_score, right_ear_score, face_detected=True
                    )
                )
                is_yawning, yawn_duration, yawn_ratio = yawn_detector.update(
                    mar_score, face_detected=True
                )

                if are_eyes_closed:
                    ear_triggers += 1
                if is_yawning:
                    mar_triggers += 1

                final_score, is_drowsy = evaluator.evaluate(
                    face_pred_val, left_pred, right_pred, is_yawning, are_eyes_closed
                )

                print(
                    f"Final Score: {final_score:.2f} | Is Drowsy: {is_drowsy} | Is Yawning: {is_yawning} | Yawn Duration: {yawn_duration} | Are Eyes Closed: {are_eyes_closed} | Closure Duration: {closure_duration}"
                )
                buzzer_alert.trigger(is_drowsy)

                if is_drowsy:
                    drowsy_frame_count += 1
                else:
                    alert_frame_count += 1

            frame = evaluator.draw_status(
                frame,
                is_face_detected=is_face_detected,
                final_score=final_score,
                is_drowsy=is_drowsy,
                is_yawning=is_yawning and yawn_duration > duration_threshold,
                are_eyes_closed=are_eyes_closed
                and closure_duration > duration_threshold,
                perclos=perclos,
            )
            frame = draw_timestamp_badge(frame)
            frame = draw_live_badge(frame)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.close()

        if evaluating_hybrid_model and valid_face_frames > 0:
            results_data = {
                "model": "hybrid_mobilenetv2_ear_mar_live_session",
                "session_summary": {
                    "total_valid_face_frames": valid_face_frames,
                    "alert_frames": alert_frame_count,
                    "drowsy_detected_frames": drowsy_frame_count,
                    "cnn_trigger_count": cnn_triggers,
                    "ear_trigger_count": ear_triggers,
                    "mar_trigger_count": mar_triggers,
                    "mean_ear": float(np.mean(ear_values)) if ear_values else 0.0,
                    "mean_mar": float(np.mean(mar_values)) if mar_values else 0.0,
                },
            }

            result_path = (
                PROJECT_ROOT
                / "evaluation"
                / "results"
                / "hybrid_driver_drowsiness_results.json"
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(results_data, f, indent=4)
            print(f"\nLive session results successfully exported to: {result_path}")
        else:
            if evaluating_hybrid_model:
                print(
                    "\nNo valid face frames were detected during the session. Results not exported."
                )


if __name__ == "__main__":
    main()
