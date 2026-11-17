import numpy as np


class AutoGainController:
    """Smooth software automatic gain control normalizing microphone speech toward target RMS."""

    def __init__(
        self,
        target_rms: float = 0.05,
        min_gain: float = 0.4,
        max_gain: float = 3.5,
        attack_rate: float = 0.15,
        release_rate: float = 0.05,
        enabled: bool = True,
    ):
        self.target_rms = target_rms
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.attack_rate = attack_rate
        self.release_rate = release_rate
        self.current_gain = 1.0
        self.enabled = enabled

    def process_frame(self, frame: np.ndarray, is_speech: bool = True) -> np.ndarray:
        """
        Scale audio frame amplitude smoothly toward target RMS.
        Uses soft-knee hyperbolic tangent (tanh) limiting to guarantee zero digital clipping.
        """
        if not self.enabled or len(frame) == 0:
            return frame

        rms = float(np.sqrt(np.mean(frame**2)))
        if rms > 1e-4 and is_speech:
            desired_gain = self.target_rms / rms
            desired_gain = float(np.clip(desired_gain, self.min_gain, self.max_gain))

            if desired_gain < self.current_gain:
                self.current_gain += self.attack_rate * (desired_gain - self.current_gain)
            else:
                self.current_gain += self.release_rate * (desired_gain - self.current_gain)

        scaled = frame * self.current_gain

        # Soft peak limiting above 0.85 to eliminate digital clipping
        if np.any(np.abs(scaled) > 0.85):
            scaled = np.where(
                np.abs(scaled) > 0.85,
                np.sign(scaled) * (0.85 + 0.15 * np.tanh((np.abs(scaled) - 0.85) / 0.15)),
                scaled,
            )

        return scaled.astype(np.float32)
