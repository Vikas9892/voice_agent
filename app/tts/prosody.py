import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProsodyConfig:
    """Configurable speech prosody and articulation parameters for Piper and audio synthesis."""

    speech_rate: float = 1.0       # Speed multiplier: 1.0 is normal, 1.25 is fast, 0.8 is slow
    noise_scale: float = 0.667     # Phoneme variation / expressiveness
    noise_w: float = 0.800         # Phoneme duration stability
    sentence_silence: float = 0.20 # Inter-sentence silence in seconds
    pitch_factor: float = 1.0      # Pitch scale factor for fallback audio generation
    voice_name: str = "en_US-lessac-medium"

    @property
    def length_scale(self) -> float:
        """
        Piper synthesis uses length_scale where lower = faster.
        Map user-facing speech_rate (e.g. 1.25x fast) to length_scale (1.0 / 1.25 = 0.8).
        """
        rate = max(min(float(self.speech_rate), 2.0), 0.5)
        return round(1.0 / rate, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speech_rate": round(self.speech_rate, 2),
            "length_scale": self.length_scale,
            "noise_scale": round(self.noise_scale, 3),
            "noise_w": round(self.noise_w, 3),
            "sentence_silence": round(self.sentence_silence, 2),
            "pitch_factor": round(self.pitch_factor, 2),
            "voice_name": self.voice_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProsodyConfig":
        return cls(
            speech_rate=float(data.get("speech_rate", 1.0)),
            noise_scale=float(data.get("noise_scale", 0.667)),
            noise_w=float(data.get("noise_w", 0.800)),
            sentence_silence=float(data.get("sentence_silence", 0.20)),
            pitch_factor=float(data.get("pitch_factor", 1.0)),
            voice_name=str(data.get("voice_name", "en_US-lessac-medium")),
        )


class ProsodyManager:
    """Maintains active prosody configuration and catalog of available voice models."""

    AVAILABLE_VOICES: List[Dict[str, str]] = [
        {"id": "en_US-lessac-medium", "name": "Lessac (US Female/Neutral)", "locale": "en_US"},
        {"id": "en_US-amy-medium", "name": "Amy (US Female)", "locale": "en_US"},
        {"id": "en_US-ryan-medium", "name": "Ryan (US Male)", "locale": "en_US"},
        {"id": "en_GB-alan-medium", "name": "Alan (GB Male)", "locale": "en_GB"},
    ]

    def __init__(self, initial_config: Optional[ProsodyConfig] = None):
        self.config = initial_config or ProsodyConfig()

    def update_config(self, **kwargs) -> ProsodyConfig:
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                val_type = type(getattr(self.config, k))
                setattr(self.config, k, val_type(v))
        return self.config

    def set_rate(self, rate: float) -> float:
        self.config.speech_rate = max(min(float(rate), 2.0), 0.5)
        return self.config.speech_rate

    def set_voice(self, voice_name: str) -> bool:
        self.config.voice_name = voice_name
        return True

    def list_voices(self) -> List[Dict[str, str]]:
        return list(self.AVAILABLE_VOICES)


default_prosody_manager = ProsodyManager()
