import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.api.server import create_app
from app.audio.vad import SileroVAD
from app.conversation.cancellation import CancellationCoordinator
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager
from app.llm.main_model import LocalLLM
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS
from fastapi.testclient import TestClient


def test_local_tts_stream_synthesize():
    tts = LocalTTS(voice_model_path="nonexistent.onnx")
    sentences = ["First sentence.", "Second sentence with more words."]
    results = list(tts.stream_synthesize(sentences))

    assert len(results) == 2
    for sentence, audio in results:
        assert isinstance(sentence, str)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0


@pytest.mark.asyncio
async def test_orchestrator_pipelined_sentence_playback():
    vad = SileroVAD(threshold=0.5)
    turn_mgr = TurnManager(vad=vad, min_speech_duration_ms=50, min_silence_duration_ms=50)
    stt = LocalSTT()
    llm = LocalLLM()
    tts = LocalTTS()
    cancellation = CancellationCoordinator()

    # Mock STT to return fixed transcript
    stt.transcribe = MagicMock(return_value="Tell me about python.")
    # Mock LLM stream_generate to yield multiple tokens that form sentences
    llm.stream_generate = MagicMock(return_value=iter([
        "Python ", "is ", "great. ", "It ", "is ", "very ", "popular!"
    ]))

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_mgr,
        stt=stt,
        llm=llm,
        tts=tts,
        cancellation=cancellation,
    )

    events_emitted = []
    orchestrator._emit = lambda event, data: events_emitted.append((event, data))

    async def fake_wait(duration, metrics):
        return True

    orchestrator._wait_playback = fake_wait

    turn_id = cancellation.start_new_turn()
    dummy_audio = np.zeros(16000, dtype=np.float32)

    await orchestrator._handle_turn(dummy_audio, turn_id)

    # Verify that multiple agent sentences were emitted
    sentence_events = [data["text"] for event, data in events_emitted if event == "agent_sentence"]
    assert len(sentence_events) >= 1
    assert any("Python is great." in s for s in sentence_events)

    # Check metrics
    turn_completed_events = [data for event, data in events_emitted if event == "turn_completed"]
    assert len(turn_completed_events) == 1
    assert "perceived_latency_ms" in turn_completed_events[0]


@pytest.mark.asyncio
async def test_orchestrator_pipelined_barge_in():
    vad = SileroVAD(threshold=0.5)
    turn_mgr = TurnManager(vad=vad, min_speech_duration_ms=50, min_silence_duration_ms=50)
    stt = LocalSTT()
    llm = LocalLLM()
    tts = LocalTTS()
    cancellation = CancellationCoordinator()

    stt.transcribe = MagicMock(return_value="Long response test.")
    llm.stream_generate = MagicMock(return_value=iter([
        "Sentence one is playing now. ", "Sentence two should be cancelled."
    ]))

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_mgr,
        stt=stt,
        llm=llm,
        tts=tts,
        cancellation=cancellation,
    )

    events_emitted = []
    orchestrator._emit = lambda event, data: events_emitted.append((event, data))

    async def fake_wait_playback(duration, metrics):
        # Simulate user interruption during sentence 1 playback
        cancellation.cancel_current_turn()
        return False

    orchestrator._wait_playback = fake_wait_playback

    turn_id = cancellation.start_new_turn()
    dummy_audio = np.zeros(16000, dtype=np.float32)

    await orchestrator._handle_turn(dummy_audio, turn_id)

    interrupted_events = [data for event, data in events_emitted if event == "turn_interrupted"]
    assert len(interrupted_events) == 1


def test_ws_voice_streaming_sentences():
    orchestrator = MagicMock()
    orchestrator.executor = None
    orchestrator.turn_manager = MagicMock()
    dummy_audio = np.zeros(512, dtype=np.float32)
    fake_state = MagicMock()
    fake_state.name = "USER_TURN_FINALIZING"
    orchestrator.turn_manager.process_frame = MagicMock(
        return_value=(fake_state, dummy_audio)
    )
    orchestrator.stt.transcribe = MagicMock(return_value="Hello streaming")
    orchestrator.tool_matcher.match_and_execute = MagicMock(return_value=None)
    orchestrator.llm.stream_generate = MagicMock(return_value=iter([
        "First sentence. ", "Second sentence!"
    ]))
    orchestrator.tts.synthesize = MagicMock(return_value=np.ones(1000, dtype=np.float32))

    app = create_app(orchestrator=orchestrator)
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        ready_msg = ws.receive_json()
        assert ready_msg["type"] == "voice_session_ready"

        # Send audio frame bytes
        frame_bytes = (dummy_audio).tobytes()
        ws.send_bytes(frame_bytes)

        # Receive processing_turn
        proc = ws.receive_json()
        assert proc["type"] == "processing_turn"

        # Receive transcript
        tr = ws.receive_json()
        assert tr["type"] == "transcript"

        # Receive agent_response full text
        resp = ws.receive_json()
        assert resp["type"] == "agent_response"

        # Receive first audio bytes
        b1 = ws.receive_bytes()
        assert len(b1) > 0

        # Receive second audio bytes
        b2 = ws.receive_bytes()
        assert len(b2) > 0

        # Receive playback_complete
        complete = ws.receive_json()
        assert complete["type"] == "playback_complete"

