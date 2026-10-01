import time
from collections import deque

"""
EyeClosureDetector module.

Detects eye closure and calculates PERCLOS (percentage of eye closure) over a rolling window.
"""


class EyeClosureDetector:

    def __init__(self, threshold_seconds=1.5, window_size=60):
        """
        Parameters:
          threshold_seconds: Duration of continuous eye closure.
        """
        self.threshold_seconds = threshold_seconds
        self.history_buffer = deque(maxlen=window_size)
        self.closed_start_time = None
        self.alert_active = False
        self.micro_sleep_count = 0
        self.alert_cooldown_until = 0.0

    def get_avg_ear(self, left_ear_score, right_ear_score) -> bool:
        """
        Returns whether a micro-sleep is currently active.

        Returns:
            Boolean indicating if micro-sleep is active.
        """

        avg_ear = (left_ear_score + right_ear_score) / 2.0
        return avg_ear

    def eyes_are_closed(self, left_ear_score, right_ear_score, threshold):
        """
        Determine if eyes are closed based on EAR scores and a threshold.

        Parameters:
            left_ear_score: EAR score for left eye.
            right_ear_score: EAR score for right eye.
            threshold: EAR threshold to consider eyes as closed.

        Returns:
            Boolean indicating if eyes are closed.
        """
        avg_ear = self.get_avg_ear(left_ear_score, right_ear_score)
        return avg_ear < threshold

    def get_perclos(self) -> float:
        """
        Calculates percentage of eye closure over rolling window buffer.

        Returns:
            Float representing percentage of eye closure.
        """
        if not self.history_buffer:
            return 0.0
        return sum(self.history_buffer) / len(self.history_buffer) * 100.0

    def update(
        self, left_ear_score: float, right_ear_score: float, face_detected: bool = True
    ) -> tuple[bool, float, float]:
        """
        Updates state based on current eye status.

        Parameters:
            left_ear_score: EAR score for left eye.
            right_ear_score: EAR score for right eye.
            face_detected: Boolean indicating if a face is detected in the frame.
            face_detected: Boolean indicating if a face is detected in the frame.
        Returns:
            is_micro_sleep: Indicates if a micro-sleep detected.
            closure_duration: Duration of the current eye closure.
            perclos: Percentage of eye closure over the rolling window buffer.
        """
        current_time = time.time()

        # Handle lost face tracking gracefully
        if not face_detected:
            self.closed_start_time = None
            return False, 0.0, self.get_perclos()

        print(f"Left EAR: {left_ear_score:.3f}, Right EAR: {right_ear_score:.3f}")

        avg_ear = (left_ear_score + right_ear_score) / 2.0
        eyes_are_closed = avg_ear < 0.20

        print(f"Avg EAR: {avg_ear:.3f}, Eyes are closed: {eyes_are_closed}")

        # Record state into sliding buffer
        self.history_buffer.append(1 if eyes_are_closed else 0)
        perclos = self.get_perclos()

        if eyes_are_closed:
            if self.closed_start_time is None:
                self.closed_start_time = current_time

            closure_duration = current_time - self.closed_start_time

            # Check if threshold is breached
            if closure_duration >= self.threshold_seconds:
                if not self.alert_active:
                    self.micro_sleep_count += 1
                self.alert_active = True
                self.alert_cooldown_until = (
                    current_time + 1.0
                )  # Keep alert active for at least 1s after opening eyes

            return self.alert_active, closure_duration, perclos

        else:
            # Eyes are open, check if we are still within latching cooldown period
            self.closed_start_time = None

            if current_time < self.alert_cooldown_until:
                return True, 0.0, perclos  # Keep alert latched briefly

            self.alert_active = False
            return False, 0.0, perclos
