import numpy as np
from app.config import settings
from app.llm.main_model import LocalLLM
from app.tts.local_tts import LocalTTS


def test_settings_loaded():
    assert settings.audio_sample_rate == 16000
    assert settings.stt_model_size == "base.en"


def test_llm_fallback():
    llm = LocalLLM(base_url="http://invalid-host:9999")
    resp = llm.generate("Hello voice agent")
    assert "Hello voice agent" in resp


def test_tts_synthesis():
    tts = LocalTTS(voice_model_path="nonexistent.onnx")
    audio = tts.synthesize("Testing local audio generation")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
