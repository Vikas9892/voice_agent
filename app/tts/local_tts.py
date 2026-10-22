import os
import numpy as np


class LocalTTS:
    def __init__(self, voice_model_path: str = "models/en_US-lessac-medium.onnx"):
        self.voice_model_path = voice_model_path
        self.voice = None
        if os.path.exists(voice_model_path):
            try:
                from piper.voice import PiperVoice
                self.voice = PiperVoice.load(voice_model_path)
            except Exception:
                self.voice = None

    def synthesize(self, text: str, sample_rate: int = 22050) -> np.ndarray:
        """Synthesize text into a 1D float32 audio array."""
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        if self.voice is not None:
            # Piper returns int16 bytes chunks
            chunks = []
            for audio_chunk in self.voice.synthesize(text):
                chunks.append(np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16))
            if chunks:
                int16_audio = np.concatenate(chunks)
                return int16_audio.astype(np.float32) / 32768.0

        # shortcut: fallback sine tone generator if voice ONNX model is not downloaded yet
        duration = min(max(len(text) * 0.04, 0.3), 2.0)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
