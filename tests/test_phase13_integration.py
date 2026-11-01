import json
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.api.server import create_app
from app.audio.session_recorder import SessionAudioRecorder


def test_sessions_rest_api_endpoints(tmp_path):
    recorder = SessionAudioRecorder(base_dir=str(tmp_path), session_id="test_rest_session")
    recorder.record_turn(
        turn_id=1,
        user_audio=np.zeros(1600, dtype=np.float32),
        agent_audio=np.ones(2205, dtype=np.float32),
        user_text="What is the weather?",
        agent_text="It is clear and mild.",
        metrics={"perceived_latency_ms": 280},
    )

    app = create_app()
    app.state.session_recorder = recorder
    client = TestClient(app)

    # 1. List sessions
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert any(s["session_id"] == "test_rest_session" for s in data["sessions"])

    # 2. Get session details
    resp = client.get("/api/sessions/test_rest_session")
    assert resp.status_code == 200
    s_data = resp.json()
    assert s_data["session_id"] == "test_rest_session"
    assert s_data["total_turns"] == 1
    assert len(s_data["turns"]) == 1

    # 3. Stream user audio WAV
    resp_user_audio = client.get("/api/sessions/test_rest_session/audio/1/user")
    assert resp_user_audio.status_code == 200
    assert resp_user_audio.headers["content-type"] == "audio/wav"
    assert len(resp_user_audio.content) > 0

    # 4. Stream agent audio WAV
    resp_agent_audio = client.get("/api/sessions/test_rest_session/audio/1/agent")
    assert resp_agent_audio.status_code == 200
    assert resp_agent_audio.headers["content-type"] == "audio/wav"
    assert len(resp_agent_audio.content) > 0

    # 5. Export session as markdown
    resp_export = client.post("/api/sessions/test_rest_session/export", json={"format": "markdown"})
    assert resp_export.status_code == 200
    assert "# Voice Session: test_rest_session" in resp_export.text
    assert "What is the weather?" in resp_export.text

    # 6. Delete session
    resp_del = client.delete("/api/sessions/test_rest_session")
    assert resp_del.status_code == 200
    assert resp_del.json()["status"] == "deleted"

    # Confirm 404 after deletion
    resp_not_found = client.get("/api/sessions/test_rest_session")
    assert resp_not_found.status_code == 404


def test_ws_voice_records_session_turn(tmp_path):
    recorder = SessionAudioRecorder(base_dir=str(tmp_path), session_id="ws_record_session")
    
    class MockTurnManager:
        def process_frame(self, frame):
            fake_state = type("State", (), {"name": "USER_TURN_FINALIZING"})()
            return fake_state, np.zeros(16000, dtype=np.float32)

    class MockSTT:
        def transcribe(self, audio):
            return "Hello from browser mic"

    class MockTTS:
        def synthesize(self, text, sample_rate=22050):
            return np.ones(500, dtype=np.float32)

    class MockLLM:
        def generate(self, text):
            return "Hello from agent"

    class MockOrchestrator:
        def __init__(self):
            self.turn_manager = MockTurnManager()
            self.executor = None
            self.tool_matcher = None
            self.session_recorder = recorder
            self.stt = MockSTT()
            self.llm = MockLLM()
            self.tts = MockTTS()

    orch = MockOrchestrator()
    app = create_app(orchestrator=orch)
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "voice_session_ready"

        # Send audio frame (512 float32 samples = 2048 bytes)
        ws.send_bytes(np.zeros(512, dtype=np.float32).tobytes())

        # Receive processing
        p = ws.receive_json()
        assert p["type"] == "processing_turn"

        # Receive transcript
        t = ws.receive_json()
        assert t["text"] == "Hello from browser mic"

        # Receive agent_response
        r = ws.receive_json()
        assert r["text"] == "Hello from agent"

        # Receive synthesized audio bytes
        audio_chunk = ws.receive_bytes()
        assert len(audio_chunk) > 0

        # Receive playback_complete
        c = ws.receive_json()
        assert c["type"] == "playback_complete"

    # Verify recorder saved the turn
    manifest = recorder._load_manifest()
    assert manifest["total_turns"] == 1
    assert manifest["turns"][0]["user_text"] == "Hello from browser mic"
    assert manifest["turns"][0]["agent_text"] == "Hello from agent"
    assert (tmp_path / "ws_record_session" / "turn_1_user.wav").exists()
    assert (tmp_path / "ws_record_session" / "turn_1_agent.wav").exists()

