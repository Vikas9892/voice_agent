import json
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.api.server import create_app
from app.conversation.turn_manager import TurnManager
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.state import TurnState


class MockSTT:
    def transcribe(self, audio):
        return "Hello world"


class MockTTS:
    def synthesize(self, text, sample_rate=22050):
        # return 100 float32 samples
        return np.ones(100, dtype=np.float32)


class MockLLM:
    def __init__(self):
        self.system_prompt = "default"
        self.temperature = 0.7

    def set_persona(self, persona):
        self.system_prompt = persona.system_prompt
        self.temperature = persona.temperature

    def generate(self, text):
        return f"Greetings: {text}"


class MockTurnManager:
    def __init__(self):
        self.call_count = 0

    def process_frame(self, frame):
        self.call_count += 1
        if self.call_count == 3:
            # Emit completed audio on 3rd frame
            return TurnState.USER_TURN_FINALIZING, np.zeros(16000, dtype=np.float32)
        return TurnState.USER_SPEAKING, None


def test_full_duplex_websocket_voice_session():
    stt = MockSTT()
    tts = MockTTS()
    llm = MockLLM()
    tm = MockTurnManager()

    class MockOrchestrator:
        def __init__(self):
            self.turn_manager = tm
            self.stt = stt
            self.tts = tts
            self.llm = llm
            self.executor = None
            self.tool_matcher = None

    orch = MockOrchestrator()
    app = create_app(orchestrator=orch)
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        init_data = ws.receive_json()
        assert init_data["type"] == "voice_session_ready"

        # Send 3 chunks of 512 float32 samples (512 * 4 = 2048 bytes each)
        chunk = np.zeros(512, dtype=np.float32).tobytes()
        ws.send_bytes(chunk)
        ws.send_bytes(chunk)
        ws.send_bytes(chunk)

        # Expect processing messages
        msg1 = ws.receive_json()
        assert msg1["type"] == "processing_turn"

        msg2 = ws.receive_json()
        assert msg2["type"] == "transcript"
        assert msg2["text"] == "Hello world"

        msg3 = ws.receive_json()
        assert msg3["type"] == "agent_response"
        assert msg3["text"] == "Greetings: Hello world"

        # Expect binary synthesized audio bytes
        audio_bytes = ws.receive_bytes()
        assert len(audio_bytes) == 100 * 4  # 100 float32s = 400 bytes

        msg4 = ws.receive_json()
        assert msg4["type"] == "playback_complete"
