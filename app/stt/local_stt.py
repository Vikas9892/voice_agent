import numpy as np
from faster_whisper import WhisperModel


class LocalSTT:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
    ):
        # shortcut: loads model synchronously on first init, prewarm in Phase 3
        self.beam_size = beam_size
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16kHz float32 mono audio array into text."""
        if len(audio) == 0:
            return ""
        segments, _ = self.model.transcribe(audio, beam_size=self.beam_size, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
