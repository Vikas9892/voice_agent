import asyncio
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.api.server import create_app, broadcaster
from app.audio.echo_cancellation import AcousticEchoSuppressor
from app.conversation.memory import ConversationMemory
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager
from app.llm.main_model import LocalLLM
from app.tools.builtin import FastToolMatcher


class MockWhisperSTT:
    def __init__(self, transcript: str):
        self.transcript = transcript

    def transcribe(self, audio):
        return self.transcript


class MockPiperTTS:
    def synthesize(self, text, sample_rate=22050):
        return np.zeros(400, dtype=np.float32)


class MockVAD:
    def __init__(self, is_speech_val=False):
        self.is_speech_val = is_speech_val

    def is_speech(self, frame, sample_rate=16000):
        return self.is_speech_val

    def reset(self):
        pass


@pytest.mark.asyncio
async def test_full_system_integration_multi_turn_with_memory_and_tools():
    # 1. Setup multi-turn memory and LLM
    memory = ConversationMemory(max_turns=3)
    llm = LocalLLM(memory=memory)

    # 2. Setup VAD, Echo suppressor, and TurnManager
    echo_suppressor = AcousticEchoSuppressor()
    vad = MockVAD(is_speech_val=False)
    turn_manager = TurnManager(vad=vad, echo_suppressor=echo_suppressor)

    # 3. Setup STT, TTS, Tools, and Orchestrator
    stt = MockWhisperSTT("What is 12 plus 18?")
    tts = MockPiperTTS()
    tool_matcher = FastToolMatcher()

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=stt,
        llm=llm,
        tts=tts,
        broadcaster=broadcaster,
        tool_matcher=tool_matcher,
    )

    # 4. Execute Turn 1 (Tool query)
    audio_frame = np.zeros(16000, dtype=np.float32)
    await orchestrator._handle_turn(audio_frame, turn_id=1)

    assert "tool_execution_completed" in orchestrator.active_metrics.timestamps
    # Memory was not invoked for tool turn
    assert len(memory.turns) == 0

    # 5. Execute Turn 2 (Chat query with memory)
    stt.transcript = "I love exploring mountain hiking trails."
    await orchestrator._handle_turn(audio_frame, turn_id=2)

    assert len(memory.turns) == 1
    assert memory.turns[0].user == "I love exploring mountain hiking trails."

    # 6. Verify Dashboard API Server Integration
    app = create_app(orchestrator=orchestrator)
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["streaming_active"] is True

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "Voice Agent Dashboard" in index_resp.text
