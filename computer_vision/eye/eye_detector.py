import cv2

"""
EyeDetector class for detecting and cropping eyes from a given image frame based on face landmarks.
"""

# MediaPipe
# https://google-ai-edge.github.io/mediapipe-samples-web/#/vision/face_landmarker
# Code Example Reference
# https://medium.com/@asadullahdal/iris-segmentation-mediapipe-python-a4deb711aae3


class EyeDetector:

    # MediaPipe left and right eye landmark index
    LEFT_EYE = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

    RIGHT_EYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ]

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
