import numpy as np
import sounddevice as sd
from app.audio.preprocessing import preprocess_audio


def record_audio(duration_sec: float, sample_rate: int = 16000, preprocess: bool = True) -> np.ndarray:
    """Record audio from default microphone for a fixed duration. Returns float32 mono array."""
    num_samples = int(duration_sec * sample_rate)
    recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = recording.flatten()
    if preprocess:
        audio = preprocess_audio(audio)
    return audio
