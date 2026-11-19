from unittest.mock import MagicMock
import numpy as np
import pytest

from app.llm.personas import PersonaManager
from app.tts.local_tts import LocalTTS
from app.tts.prosody import ProsodyConfig, ProsodyManager


def test_prosody_config_defaults_and_length_scale():
    cfg = ProsodyConfig()
    assert cfg.speech_rate == 1.0
    assert cfg.length_scale == 1.0

    fast_cfg = ProsodyConfig(speech_rate=1.25)
    assert fast_cfg.length_scale == 0.8

    slow_cfg = ProsodyConfig(speech_rate=0.8)
    assert slow_cfg.length_scale == 1.25

    # Out of bounds clamping (0.5 to 2.0)
    super_fast = ProsodyConfig(speech_rate=4.0)
    assert super_fast.length_scale == 0.5

    super_slow = ProsodyConfig(speech_rate=0.1)
    assert super_slow.length_scale == 2.0


def test_prosody_config_dict_roundtrip():
    cfg = ProsodyConfig(
        speech_rate=1.2,
        noise_scale=0.7,
        noise_w=0.85,
        sentence_silence=0.25,
        pitch_factor=1.1,
        voice_name="en_US-amy-medium",
    )
    d = cfg.to_dict()
    assert d["speech_rate"] == 1.2
    assert d["length_scale"] == 0.833
    assert d["voice_name"] == "en_US-amy-medium"

    restored = ProsodyConfig.from_dict(d)
    assert restored.speech_rate == 1.2
    assert restored.noise_scale == 0.7
    assert restored.voice_name == "en_US-amy-medium"


def test_prosody_manager():
    mgr = ProsodyManager()
    voices = mgr.list_voices()
    assert len(voices) >= 4
    assert any(v["id"] == "en_US-lessac-medium" for v in voices)

    mgr.set_rate(1.5)
    assert mgr.config.speech_rate == 1.5
    assert mgr.config.length_scale == 0.667

    mgr.set_rate(10.0)
    assert mgr.config.speech_rate == 2.0

    mgr.set_voice("en_US-ryan-medium")
    assert mgr.config.voice_name == "en_US-ryan-medium"

    mgr.update_config(noise_scale=0.55)
    assert mgr.config.noise_scale == 0.55


def test_local_tts_fallback_prosody():
    tts = LocalTTS(voice_model_path="nonexistent.onnx")

    # Higher rate -> shorter duration
    audio_normal = tts.synthesize("Hello world, this is a test speech synthesis utterance.", sample_rate=22050)
    audio_fast = tts.synthesize(
        "Hello world, this is a test speech synthesis utterance.",
        sample_rate=22050,
        prosody=ProsodyConfig(speech_rate=1.5),
    )
    assert len(audio_fast) < len(audio_normal)

    # Stream synthesize yields audio with prosody applied
    chunks = list(tts.stream_synthesize(["First sentence.", "Second sentence."], prosody=ProsodyConfig(speech_rate=1.5)))
    assert len(chunks) == 2
    assert chunks[0][0] == "First sentence."
    assert isinstance(chunks[0][1], np.ndarray)


def test_local_tts_piper_kwargs():
    tts = LocalTTS(voice_model_path="nonexistent.onnx")
    mock_voice = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.audio_int16_bytes = (np.zeros(100, dtype=np.int16)).tobytes()
    mock_voice.synthesize.return_value = [mock_chunk]
    tts.voice = mock_voice

    cfg = ProsodyConfig(speech_rate=1.25, noise_scale=0.5, noise_w=0.7, sentence_silence=0.15)
    audio = tts.synthesize("Test phrase", prosody=cfg)

    assert len(audio) == 100
    mock_voice.synthesize.assert_called_once_with(
        "Test phrase",
        length_scale=cfg.length_scale,
        noise_scale=cfg.noise_scale,
        noise_w=cfg.noise_w,
        sentence_silence=cfg.sentence_silence,
    )


def test_personas_prosody_metadata():
    mgr = PersonaManager()
    personas = mgr.list_personas()
    assert len(personas) >= 4

    p_dict = {p["name"]: p for p in personas}
    assert "speech_rate" in p_dict["assistant"]
    assert "voice_model" in p_dict["assistant"]
    assert p_dict["technologist"]["speech_rate"] == 1.15
    assert p_dict["tutor"]["speech_rate"] == 0.90
