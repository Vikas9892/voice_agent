import os
from typing import Iterable, Iterator, Optional, Tuple, Union
import numpy as np

from app.tts.prosody import ProsodyConfig


class LocalTTS:
    def __init__(
        self,
        voice_model_path: str = "models/en_US-lessac-medium.onnx",
        prosody: Optional[ProsodyConfig] = None,
    ):
        self.voice_model_path = voice_model_path
        self.prosody = prosody or ProsodyConfig()
        self.voice = None
        if os.path.exists(voice_model_path):
            try:
                from piper.voice import PiperVoice
                self.voice = PiperVoice.load(voice_model_path)
            except Exception:
                self.voice = None

    def synthesize(
        self,
        text: str,
        sample_rate: int = 22050,
        prosody: Optional[ProsodyConfig] = None,
    ) -> np.ndarray:
        """Synthesize text into a 1D float32 audio array applying prosody settings."""
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        cfg = prosody or self.prosody

        if self.voice is not None:
            chunks = []
            try:
                synth_iter = self.voice.synthesize(
                    text,
                    length_scale=cfg.length_scale,
                    noise_scale=cfg.noise_scale,
                    noise_w=cfg.noise_w,
                    sentence_silence=cfg.sentence_silence,
                )
            except TypeError:
                synth_iter = self.voice.synthesize(text)

            for audio_chunk in synth_iter:
                chunks.append(np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16))
            if chunks:
                int16_audio = np.concatenate(chunks)
                return int16_audio.astype(np.float32) / 32768.0

        # shortcut: fallback sine tone generator scaled by speech_rate and pitch_factor
        effective_rate = max(min(cfg.speech_rate, 2.0), 0.5)
        duration = min(max((len(text) * 0.04) / effective_rate, 0.2), 2.5)
        freq = 440.0 * max(min(cfg.pitch_factor, 2.0), 0.5)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def stream_synthesize(
        self,
        sentences: Union[str, Iterable[str]],
        sample_rate: int = 22050,
        prosody: Optional[ProsodyConfig] = None,
    ) -> Iterator[Tuple[str, np.ndarray]]:
        """Synthesize a sequence of text sentences, yielding (sentence, audio_array) pairs."""
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence in sentences:
            clean = sentence.strip()
            if not clean:
                continue
            audio = self.synthesize(clean, sample_rate=sample_rate, prosody=prosody)
            if len(audio) > 0:
                yield clean, audio
