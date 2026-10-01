import cv2
import numpy as np

"""
EyeDetector class for detecting and cropping eyes from a given image frame based on face landmarks.
"""

# MediaPipe
# https://google-ai-edge.github.io/mediapipe-samples-web/#/vision/face_landmarker
# Code Example Reference
# https://learnopencv.com/driver-drowsiness-detection-using-mediapipe-in-python/
# https://medium.com/@asadullahdal/iris-segmentation-mediapipe-python-a4deb711aae3
# https://medium.com/@sabarishds03/pupil-patrol-building-live-eye-tracking-with-mediapipe-facemesh-88210bbca3f1
# https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV


class EyeDetector:

    # MediaPipe left and right eye landmark index
    LEFT_EYE = [33, 160, 158, 133, 153, 144]

    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, padding=30):
        """
        Initialize EyeDetector with optional padding around detected eye regions.

        Parameters:
            padding: The number of pixels to pad around the detected eye regions.
        """
        self.padding = padding

    def detect(self, frame, face_landmarks):
        """
        Detect eyes in the given frame based on face landmarks.

        Parameters:
            frame: Input image frame.
            face_landmarks: List of face landmarks.

        Returns:
            List of bounding boxes for detected eyes.
        """
        if face_landmarks is None:
            return []

        height, width, _ = frame.shape

        eye_boxes = []

        for eye_landmarks in [self.LEFT_EYE, self.RIGHT_EYE]:
            x_coordinates = []
            y_coordinates = []
            # Iterate through each landmark index for the current eye
            for index in eye_landmarks:
                landmark = face_landmarks[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                x_coordinates.append(x)
                y_coordinates.append(y)

            # Original eye region
            x_min = min(x_coordinates)
            x_max = max(x_coordinates)
            y_min = min(y_coordinates)
            y_max = max(y_coordinates)

            # Add padding
            x_min -= self.padding
            x_max += self.padding
            y_min -= self.padding
            y_max += self.padding

            # Make the crop square
            crop_width = x_max - x_min
            crop_height = y_max - y_min

            crop_size = max(crop_width, crop_height)

            # Calculate the center of the eye region
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Calcalate the new x_min, x_max, y_min, y_max for the square crop
            x_min = center_x - crop_size // 2
            x_max = x_min + crop_size

            y_min = center_y - crop_size // 2
            y_max = y_min + crop_size

            # Keep crop inside frame
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)

            # Append the calculated eye box to the list of eye boxes
            eye_boxes.append((x_min, y_min, x_max, y_max))

        return eye_boxes

    def draw_indicators(self, frame, eye_boxes):
        """
        Draw rectangles around detected eye regions on given frame.

        Parameters:
            frame: Input image frame.
            eye_boxes: List of bounding boxes for detected eyes.
        """

        for x_min, y_min, x_max, y_max in eye_boxes:

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 128, 0),
                2,
            )

    def get_ear(self, eye_landmarks, landmarks, frame=None):
        """
        Calculate Eye Aspect Ratio (EAR) for detected eye.

        Parameters:
            eye_landmarks: List of indices for eye landmarks.
            landmarks: List of all facial landmarks.

        Returns:
            EAR value as a float. Returns 0.0 if calculation fails.
        """

        if landmarks is None:
            return 0.0

        pts = []
        if frame is not None:
            height, width, _ = frame.shape
            for index in eye_landmarks:
                landmark = landmarks[index]
                pts.append(np.array([landmark.x * width, landmark.y * height]))
        else:
            for index in eye_landmarks:
                landmark = landmarks[index]
                pts.append(np.array([landmark.x, landmark.y]))

        pts = np.array(pts)

        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])

        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def crop_eyes(self, frame, eye_boxes):
        """
        Crop detected eye regions from given frame.

        Parameters:
            frame: Input image frame.
            eye_boxes: List of bounding boxes for detected eyes.

        Returns:
            List of cropped eye images.
        """

        eye_crops = []

        for x_min, y_min, x_max, y_max in eye_boxes:

            eye = frame[y_min:y_max, x_min:x_max]

            eye_crops.append(eye)

        return eye_crops
