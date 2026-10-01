from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import COLOR_GREEN_BGR, COLOR_RED_BGR

"""
FacialLandmarkDetector class for detecting and drawing facial landmarks using MediaPipe.
"""

# MediaPipe
# https://google-ai-edge.github.io/mediapipe-samples-web/#/vision/face_landmarker
# Code Example Reference
# https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/
# https://medium.com/@rajarshi023/real-time-mental-fatigue-detection-using-computer-vision-and-mediapipe-21f0e1a37dd3
# https://medium.com/@Mert.A/detect-eyes-nose-and-mouth-with-mediapipe-bbfdf7a61f21


class FacialLandmarkDetector:

    # Index positions in mouth_indices:
    # 0: 61  (Left outer corner)
    # 1: 291 (Right outer corner)
    # 2: 0   (Outer upper lip)
    # 3: 17  (Outer lower lip)
    # 4: 78  (Left inner corner)
    # 5: 308 (Right inner corner)
    # 6: 13  (Inner upper lip)
    # 7: 14  (Inner lower lip)
    MOUTH_INDICES = [61, 291, 0, 17, 78, 308, 13, 14]

    def __init__(self):
        """
        Initialize FacialLandmarkDetector for detecting facial landmarks using MediaPipe.
        """
        project_root = Path(__file__).resolve().parents[2]

        model_path = (
            project_root
            / "computer_vision"
            / "facial_landmarks"
            / "models"
            / "face_landmarker.task"
        )

        base_options = mp.tasks.BaseOptions
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        options = face_landmarker_options(
            base_options=base_options(model_asset_path=str(model_path)),
            running_mode=running_mode.VIDEO,
            num_faces=1,
        )

        self.landmarker = face_landmarker.create_from_options(options)

        self.timestamp_ms = 0

    def detect(self, frame):
        """
        Detect facial landmarks in given frame.

        Parameters:
            frame: Input image frame.

        Returns:
            Detected facial landmarks or None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Increment the timestamp for video frame processing
        self.timestamp_ms += 33

        # Get results for the current video frame
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    def draw_landmarks(self, frame, landmarks):
        """
        Draw facial landmarks on given frame.

        Parameters:
            frame: Input image frame.
            landmarks: List of facial landmarks.
        """
        if landmarks is None:
            return

        height, width, _ = frame.shape

        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    def draw_indicator(self, frame, landmarks, label=None, is_alert=False):
        """
        Draw indicator around detected facial landmarks.

        Parameters:
            frame: Input image frame.
            landmarks: List of facial landmarks.
            label: Optional label to display above bounding box.
            is_alert: Boolean flag indicating if indicator should be red or green.
        """
        if landmarks is None or len(landmarks) == 0:
            return

        height, width, _ = frame.shape
        points = []

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        if not points:
            return

        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]

        x_min = min(x_coordinates)
        x_max = max(x_coordinates)
        y_min = min(y_coordinates)
        y_max = max(y_coordinates)

        # Dynamic color: Red for alert/yawn, Green for safe/awake
        color = COLOR_RED_BGR if is_alert else COLOR_GREEN_BGR  # BGR format

        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label telemetry right above the box if provided
        if label:
            cv2.putText(
                frame,
                label,
                (
                    x_min,
                    max(20, y_min - 10),
                ),  # Prevent text from cutting off at the top
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def get_mar(self, face_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) for yawning detection.

        Parameters:
            face_landmarks: List of facial landmarks.

        Returns:
            Calculated MAR value.
        """

        if face_landmarks is None:
            return 0.0

        landmarks = (
            face_landmarks.landmark
            if hasattr(face_landmarks, "landmark")
            else face_landmarks
        )

        if not landmarks or len(landmarks) <= max(self.MOUTH_INDICES):
            return 0.0

        def pt(idx):
            return np.array([landmarks[idx].x, landmarks[idx].y])

        left_corner = pt(61)
        right_corner = pt(291)
        horizontal_dist = np.linalg.norm(left_corner - right_corner)

        if horizontal_dist == 0:
            return 0.0

        # Vertical mouth heights multiple pairs for robustness
        v1 = np.linalg.norm(pt(13) - pt(14))  # Inner lips vertical
        v2 = np.linalg.norm(pt(12) - pt(15))  # Outer lips vertical
        v3 = np.linalg.norm(pt(0) - pt(17))  # Top/bottom outer bounds

        # Average vertical distance divided by horizontal distance
        mar = (v1 + v2 + v3) / (3.0 * horizontal_dist)
        return float(mar)

    def get_face_box(self, frame, landmarks):
        """
        Get bounding box around detected facial landmarks.

        Parameters:
            frame: The input image frame.
            landmarks: List of facial landmarks.

        Returns:
            A tuple (x_min, y_min, x_max, y_max) representing bounding box coordinates.
        """
        if landmarks is None:
            return None

        height, width, _ = frame.shape

        points = []

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]

        x_min = min(x_coordinates)
        x_max = max(x_coordinates)
        y_min = min(y_coordinates)
        y_max = max(y_coordinates)

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, width)
        y_max = min(y_max, height)

        return (x_min, y_min, x_max, y_max)

    def close(self):
        self.landmarker.close()
