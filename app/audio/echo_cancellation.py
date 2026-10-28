import numpy as np
from typing import Tuple


class AcousticEchoSuppressor:
    """
    Lightweight software acoustic echo suppression (AES) for half/full-duplex desktop audio.
    Suppresses microphone speaker loopback during agent playback while preserving loud user barge-in.
    """

    def __init__(
        self,
        echo_energy_threshold: float = 0.15,
        barge_in_energy_threshold: float = 0.28,
        suppression_factor: float = 0.08,
    ):
        self.echo_energy_threshold = echo_energy_threshold
        self.barge_in_energy_threshold = barge_in_energy_threshold
        self.suppression_factor = suppression_factor
        self.is_playback_active: bool = False

    def set_playback_active(self, active: bool):
        """Update whether speakers are currently outputting agent audio."""
        self.is_playback_active = active

    def process_frame(self, frame: np.ndarray, is_agent_speaking: bool | None = None) -> Tuple[np.ndarray, bool]:
        """
        Process an incoming microphone frame.
        Returns (filtered_frame, is_suppressed_echo).
        If the agent is speaking and mic energy is below the barge-in breakthrough threshold,
        the frame is attenuated to eliminate false speech detection.
        """
        active = self.is_playback_active if is_agent_speaking is None else is_agent_speaking
        if not active:
            return frame, False

        rms = float(np.sqrt(np.mean(frame**2)))

        if rms < self.echo_energy_threshold:
            # Complete speaker leakage: suppress
            attenuated = frame * self.suppression_factor
            return attenuated, True
        elif rms < self.barge_in_energy_threshold:
            # Intermediate acoustic reflection: attenuate partially
            attenuated = frame * 0.3
            return attenuated, True
        else:
            # Genuine loud human barge-in: pass through cleanly to trigger interrupt!
            return frame, False
