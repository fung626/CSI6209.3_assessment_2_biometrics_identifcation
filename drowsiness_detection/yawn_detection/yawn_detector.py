import time
from collections import deque


class YawnDetector:

    def __init__(self, threshold_seconds=1.0, mar_threshold=0.6, window_size=60):
        self.threshold_seconds = threshold_seconds
        self.mar_threshold = mar_threshold
        self.yawn_start_time = None

        # Sliding window buffer for tracking recent mouth aperture history
        self.history_buffer = deque(maxlen=window_size)

        # Latching state to prevent rapid alarm flickering
        self.alert_active = False
        self.alert_cooldown_until = 0.0

        # Session statistics
        self.yawn_count = 0

    def update(
        self, mar_score: float, face_detected: bool = True
    ) -> tuple[bool, float, float]:
        """
        Tracks mouth state over time to filter out talking/laughing
        and detect a sustained yawn.

        Parameters:
            mar_score: Current Mouth Aspect Ratio score.
            face_detected: Whether MediaPipe successfully found a face mesh.

        Returns:
            is_yawning_active: Indicates if a yawn is currently detected.
            current_duration: Duration of the current yawn.
            yawn_ratio: Percentage of frames the mouth was open over the rolling window.
        """
        current_time = time.time()

        # Handle lost face tracking safely
        if not face_detected:
            self.yawn_start_time = None
            return False, 0.0, self.get_yawn_ratio()

        mouth_is_open = mar_score >= self.mar_threshold
        self.history_buffer.append(1 if mouth_is_open else 0)
        yawn_ratio = self.get_yawn_ratio()

        if mouth_is_open:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time

            duration = current_time - self.yawn_start_time

            # Check if duration meets the yawn time threshold
            if duration >= self.threshold_seconds:
                if not self.alert_active:
                    self.yawn_count += 1
                self.alert_active = True
                self.alert_cooldown_until = (
                    current_time + 1.0
                )  # Keep alert active briefly after closing

            return self.alert_active, duration, yawn_ratio

        else:
            self.yawn_start_time = None

            # Handle latching cooldown so alerts don't flicker abruptly
            if current_time < self.alert_cooldown_until:
                return True, 0.0, yawn_ratio

            self.alert_active = False
            return False, 0.0, yawn_ratio

    def get_yawn_ratio(self) -> float:
        """Calculates the percentage of frames the mouth was open over the rolling window."""
        if not self.history_buffer:
            return 0.0
        return sum(self.history_buffer) / len(self.history_buffer) * 100.0
