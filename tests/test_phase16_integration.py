from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from app.api.server import create_app
from app.llm.personas import default_persona_manager
from app.tts.local_tts import LocalTTS
from app.tts.prosody import ProsodyConfig, ProsodyManager


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.tts = LocalTTS(voice_model_path="nonexistent.onnx")
    orch.prosody_manager = ProsodyManager()
    orch.llm = MagicMock()
    orch.session_recorder = MagicMock()
    orch.session_recorder.base_dir = "test_sessions"
    return orch


def test_get_tts_config(mock_orchestrator):
    app = create_app(orchestrator=mock_orchestrator)
    client = TestClient(app)

    response = client.get("/api/tts/config")
    assert response.status_code == 200
    data = response.json()
    assert "speech_rate" in data
    assert "length_scale" in data
    assert "voice_name" in data
    assert data["speech_rate"] == 1.0


def test_update_tts_config(mock_orchestrator):
    app = create_app(orchestrator=mock_orchestrator)
    client = TestClient(app)

    payload = {"speech_rate": 1.25, "pitch_factor": 1.1, "voice_name": "en_US-amy-medium"}
    response = client.post("/api/tts/config", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "ok"
    assert res_data["config"]["speech_rate"] == 1.25
    assert res_data["config"]["length_scale"] == 0.8
    assert res_data["config"]["voice_name"] == "en_US-amy-medium"

    # Verify orchestrator TTS has updated prosody
    assert mock_orchestrator.tts.prosody.speech_rate == 1.25


def test_get_tts_voices(mock_orchestrator):
    app = create_app(orchestrator=mock_orchestrator)
    client = TestClient(app)

    response = client.get("/api/tts/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert len(data["voices"]) >= 4
    assert "active_voice" in data


def test_persona_switching_updates_prosody(mock_orchestrator):
    app = create_app(orchestrator=mock_orchestrator)
    client = TestClient(app)

    # Switch to 'technologist' (speech_rate = 1.15)
    response = client.post("/api/personas/active", json={"name": "technologist"})
    assert response.status_code == 200
    data = response.json()
    assert data["active"] == "technologist"
    assert data["speech_rate"] == 1.15

    # Orchestrator TTS prosody rate should have updated
    assert mock_orchestrator.tts.prosody.speech_rate == 1.15

    # Switch to 'tutor' (speech_rate = 0.90)
    response = client.post("/api/personas/active", json={"name": "tutor"})
    assert response.status_code == 200
    assert response.json()["speech_rate"] == 0.90
    assert mock_orchestrator.tts.prosody.speech_rate == 0.90


def test_static_html_contains_speech_rate_slider():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    assert 'id="speech-rate-slider"' in html
    assert 'id="speech-rate-val"' in html
