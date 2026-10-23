import numpy as np


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtract DC mean offset from audio signal to eliminate microphone bias."""
    if len(audio) == 0:
        return audio
    return (audio - np.mean(audio)).astype(np.float32)


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """High-pass filter boosting high-frequency speech consonants for clearer STT."""
    if len(audio) <= 1:
        return audio
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)


def apply_noise_gate(audio: np.ndarray, threshold_rms: float = 0.008, attenuation: float = 0.1) -> np.ndarray:
    """
    Lightweight audio noise gate attenuating stationary ambient background noise.
    Frames below threshold_rms have their amplitude smoothly scaled down.
    """
    if len(audio) == 0:
        return audio
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < threshold_rms:
        # Attenuate noise frame without cutting abruptly
        return (audio * attenuation).astype(np.float32)
    return audio


def preprocess_audio(
    audio: np.ndarray,
    dc_offset: bool = True,
    noise_gate: bool = True,
    threshold_rms: float = 0.008,
) -> np.ndarray:
    """Sequential pipeline applying DC removal and noise gating to input audio."""
    if len(audio) == 0:
        return audio
    out = audio
    if dc_offset:
        out = remove_dc_offset(out)
    if noise_gate:
        out = apply_noise_gate(out, threshold_rms=threshold_rms)
    return out
