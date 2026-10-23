import numpy as np


def play_audio(audio: np.ndarray, sample_rate: int = 22050, blocking: bool = True) -> None:
    """Play float32 or int16 audio array through default sound output device."""
    if len(audio) == 0:
        return
    import sounddevice as sd
    sd.play(audio, samplerate=sample_rate)
    if blocking:
        sd.wait()


def stop_playback() -> None:
    """Instantly interrupt and stop current audio output buffer."""
    try:
        import sounddevice as sd
        sd.stop()
    except Exception:
        pass
