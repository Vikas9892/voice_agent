import numpy as np


class AmbientNoiseProfiler:
    """Tracks environmental background noise floor, signal-to-noise ratio (SNR), and computes adaptive VAD sensitivity."""

    def __init__(
        self,
        initial_floor_rms: float = 0.005,
        alpha: float = 0.03,
        min_threshold: float = 0.35,
        max_threshold: float = 0.75,
    ):
        self.noise_floor_rms = initial_floor_rms
        self.alpha = alpha  # Exponential moving average smoothing factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.current_snr_db = 0.0
        self.current_frame_rms = 0.0
        self.calibrating = False
        self._calibration_samples = []

    @property
    def noise_floor_dbfs(self) -> float:
        """Noise floor expressed in dBFS (decibels relative to full scale)."""
        return float(20.0 * np.log10(max(self.noise_floor_rms, 1e-6)))

    def update(self, frame: np.ndarray, is_speech: bool = False) -> float:
        """Update running noise floor estimate from non-speech frames and compute SNR."""
        if len(frame) == 0:
            return self.noise_floor_rms

        rms = float(np.sqrt(np.mean(frame**2)))
        self.current_frame_rms = rms

        if self.calibrating:
            self._calibration_samples.append(rms)
            self.noise_floor_rms = float(np.mean(self._calibration_samples))
            return self.noise_floor_rms

        if not is_speech:
            # Update background noise floor using exponential moving average
            self.noise_floor_rms = (1.0 - self.alpha) * self.noise_floor_rms + self.alpha * rms

        # Compute instantaneous SNR relative to estimated noise floor
        ratio = max(rms, 1e-6) / max(self.noise_floor_rms, 1e-6)
        self.current_snr_db = float(20.0 * np.log10(ratio))
        return self.noise_floor_rms

    def compute_adaptive_threshold(self, base_threshold: float = 0.5) -> float:
        """
        Dynamically scale VAD threshold based on ambient noise floor.
        - High noise floor (> 0.02) -> increase threshold to suppress false barge-in.
        - Quiet room (< 0.004) -> lower threshold to capture soft whisper onset.
        """
        ref_noise_floor = 0.005
        diff = self.noise_floor_rms - ref_noise_floor
        adjustment = diff * 10.0
        adaptive = base_threshold + adjustment
        return float(np.clip(adaptive, self.min_threshold, self.max_threshold))

    def start_calibration(self):
        """Begin calibration sample accumulation."""
        self.calibrating = True
        self._calibration_samples.clear()

    def finish_calibration(self) -> dict:
        """Conclude calibration and set calibrated noise floor."""
        self.calibrating = False
        if self._calibration_samples:
            self.noise_floor_rms = float(np.percentile(self._calibration_samples, 50))
        return {
            "noise_floor_rms": round(self.noise_floor_rms, 5),
            "noise_floor_dbfs": round(self.noise_floor_dbfs, 1),
            "recommended_threshold": round(self.compute_adaptive_threshold(0.5), 2),
        }
