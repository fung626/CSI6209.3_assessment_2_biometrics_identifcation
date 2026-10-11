import os
import platform
import threading
import time


class BuzzerAlert:
    def __init__(self, cooldown_seconds=3.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0

    def trigger(self, is_drowsy: bool):
        if not is_drowsy:
            return
        current_time = time.time()
        if current_time - self.last_alert_time >= self.cooldown_seconds:
            self._play_sound()
            self.last_alert_time = current_time

    def trigger(self, is_drowsy: bool):
        if not is_drowsy:
            return

        current_time = time.time()

        if current_time - self.last_alert_time >= self.cooldown_seconds:
            self._play_sound_async()
            self.last_alert_time = current_time

    def _play_sound_async(self):
        sound_thread = threading.Thread(target=self._play_audio_file, daemon=True)
        sound_thread.start()

    def _play_audio_file(self):
        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                os.system("afplay /System/Library/Sounds/Glass.aiff")
            elif system == "Windows":  # Windows
                import winsound

                winsound.Beep(2500, 1000)
            else:  # Linux / Ubuntu
                print("\a")
        except Exception as e:
            print(f"Audio alert failed to play: {e}")
