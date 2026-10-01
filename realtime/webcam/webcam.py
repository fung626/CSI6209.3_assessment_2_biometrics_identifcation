import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from computer_vision.eye.eye_detector import EyeDetector
from computer_vision.facial_landmarks.facial_landmark_detector import (
    FacialLandmarkDetector,
)
from deep_learning.mobilenetv2.driver_drowsiness.face_predictor import FacePredictor
from deep_learning.mobilenetv2.mrl_eye.eye_predictor import EyePredictor
from drowsiness_detection.drowsiness_logic.drowsiness_evaluator import (
    DrowsinessEvaluator,
)
from drowsiness_detection.eye_closure_detection.eye_closure_detector import (
    EyeClosureDetector,
)
from drowsiness_detection.yawn_detection.yawn_detector import YawnDetector
from realtime.alert_system.buzzer_alert import BuzzerAlert


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

    print("Webcam started. Press Q to quit.")

    try:
        while True:

            window_name = "Driver Drowsiness Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            ret, frame = cap.read()

            if not ret:
                print("Error: Cannot read webcam frame")
                break

            face_landmarks = face_detector.detect(frame)

            if face_landmarks is not None:
                # Draw face bounding box
                face_box = face_detector.get_face_box(frame, face_landmarks)
                mar_score = face_detector.get_mar(face_landmarks)

                face_detector.draw_indicator(frame, face_landmarks)

                # https://medium.com/@Mert.A/detect-eyes-nose-and-mouth-with-mediapipe-bbfdf7a61f21
                mouth_indices = [61, 291, 0, 17, 78, 308, 13, 14]
                mouth_landmarks = [
                    face_landmarks[i] for i in mouth_indices if i < len(face_landmarks)
                ]

                is_yawning = mar_score > 0.5
                mar_percentage = min(100.0, (mar_score / 0.5) * 100.0)
                mouth_label = f"{mar_percentage:.1f}% is {'YAWNING' if is_yawning else 'NOT YAWNING'}"
                face_detector.draw_indicator(
                    frame,
                    mouth_landmarks,
                    label=mouth_label,
                )

                # Predict face drowsiness status
                face_predictor.draw_prediction(frame, face_box)
                # Detect eyes
                eye_boxes = eye_detector.detect(frame, face_landmarks)
                # Draw eye bounding boxes
                eye_detector.draw_indicators(frame, eye_boxes)
                # Predict each eye
                if eye_boxes:
                    eye_detector.draw_indicators(frame, eye_boxes)
                    for eye_box in eye_boxes:
                        eye_predictor.draw_predictions(frame, eye_box)

                raw_face_pred = face_predictor.predict(
                    face_predictor.extract_face(frame, face_box)
                )

                left_pred, right_pred = None, None
                if eye_boxes and len(eye_boxes) >= 2:
                    left_pred = eye_predictor.predict(
                        eye_predictor.extract_eye(frame, eye_boxes[0])
                    )
                    right_pred = eye_predictor.predict(
                        eye_predictor.extract_eye(frame, eye_boxes[1])
                    )

                left_ear_score = eye_detector.get_ear(
                    eye_detector.LEFT_EYE, face_landmarks
                )
                right_ear_score = eye_detector.get_ear(
                    eye_detector.RIGHT_EYE, face_landmarks
                )

                are_eyes_closed, closure_duration, perclos = (
                    eye_closure_detector.update(
                        left_ear_score, right_ear_score, face_detected=True
                    )
                )
                is_yawning, yawn_duration, yawn_ratio = yawn_detector.update(
                    mar_score, face_detected=True
                )

                final_score, is_drowsy = evaluator.evaluate(
                    raw_face_pred, left_pred, right_pred, is_yawning, are_eyes_closed
                )

                print(
                    f"Final Score: {final_score:.2f} | Is Drowsy: {is_drowsy} | Is Yawning: {is_yawning} | Are Eyes Closed: {are_eyes_closed}"
                )

                frame = evaluator.draw_status(
                    frame, final_score, is_drowsy, is_yawning, are_eyes_closed
                )
                buzzer_alert.trigger(is_drowsy)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        face_detector.close()


if __name__ == "__main__":
    main()
