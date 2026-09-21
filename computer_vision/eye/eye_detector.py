import cv2


class EyeDetector:

    # MediaPipe Face Landmarker eye landmark indices
    LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]

    RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

    def __init__(self, padding=30):
        self.padding = padding

    def detect(self, frame, face_landmarks):
        if face_landmarks is None:
            return []

        height, width, _ = frame.shape

        eye_boxes = []

        for eye_landmarks in [self.LEFT_EYE, self.RIGHT_EYE]:

            x_coordinates = []
            y_coordinates = []

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

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            x_min = center_x - crop_size // 2
            x_max = x_min + crop_size

            y_min = center_y - crop_size // 2
            y_max = y_min + crop_size

            # Keep crop inside frame
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)

            eye_boxes.append((x_min, y_min, x_max, y_max))

        return eye_boxes

    def draw_indicators(self, frame, eye_boxes):

        for x_min, y_min, x_max, y_max in eye_boxes:

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 128, 0),
                2,
            )

    def crop_eyes(self, frame, eye_boxes):

        eye_crops = []

        for x_min, y_min, x_max, y_max in eye_boxes:

            eye = frame[y_min:y_max, x_min:x_max]

            eye_crops.append(eye)

        return eye_crops
