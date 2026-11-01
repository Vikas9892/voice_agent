import asyncio
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.audio.session_recorder import SessionAudioRecorder
from app.audio.vad import SileroVAD
from app.conversation.cancellation import CancellationCoordinator
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager
from app.llm.main_model import LocalLLM
from app.stt.local_stt import LocalSTT
from app.tts.local_tts import LocalTTS


def test_session_recorder_init_and_manifest(tmp_path):
    recorder = SessionAudioRecorder(base_dir=str(tmp_path), session_id="test_session_1")
    manifest = recorder._load_manifest()

    assert manifest["session_id"] == "test_session_1"
    assert manifest["total_turns"] == 0
    assert manifest["turns"] == []
    assert (tmp_path / "test_session_1" / "manifest.json").exists()


def test_session_recorder_record_turn_and_wavs(tmp_path):
    recorder = SessionAudioRecorder(base_dir=str(tmp_path), session_id="test_session_wav")
    user_audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    agent_audio = np.linspace(-0.8, 0.8, 2205, dtype=np.float32)

    entry = recorder.record_turn(
        turn_id=1,
        user_audio=user_audio,
        agent_audio=agent_audio,
        user_sample_rate=16000,
        agent_sample_rate=22050,
        user_text="What time is it?",
        agent_text="It is 5 PM.",
        metrics={"perceived_latency_ms": 320},
        persona="assistant",
    )

    assert entry["turn_id"] == 1
    assert entry["user_audio_file"] == "turn_1_user.wav"
    assert entry["agent_audio_file"] == "turn_1_agent.wav"
    assert entry["user_text"] == "What time is it?"

    # Check files exist on disk
    user_path = recorder.get_audio_path(1, "user")
    agent_path = recorder.get_audio_path(1, "agent")
    assert user_path is not None and user_path.exists()
    assert agent_path is not None and agent_path.exists()

    # Verify manifest updated
    loaded = SessionAudioRecorder.load_session("test_session_wav", base_dir=str(tmp_path))
    assert loaded["total_turns"] == 1
    assert len(loaded["turns"]) == 1


def test_session_recorder_listing_and_export(tmp_path):
    recorder = SessionAudioRecorder(base_dir=str(tmp_path), session_id="test_export")
    recorder.record_turn(
        turn_id=1,
        user_text="Hello",
        agent_text="Hi there!",
        metrics={"perceived_latency_ms": 250},
    )

    sessions = SessionAudioRecorder.list_all_sessions(base_dir=str(tmp_path))
    assert len(sessions) >= 1
    assert any(s["session_id"] == "test_export" for s in sessions)

    # Test JSON export
    json_export = SessionAudioRecorder.export_session("test_export", fmt="json", base_dir=str(tmp_path))
    assert json_export is not None
    data = json.loads(json_export)
    assert data["session_id"] == "test_export"

    # Test Markdown export
    md_export = SessionAudioRecorder.export_session("test_export", fmt="markdown", base_dir=str(tmp_path))
    assert md_export is not None
    assert "# Voice Session: test_export" in md_export
    assert "- **User:** Hello" in md_export
    assert "- **Agent:** Hi there!" in md_export

    # Test deletion
    deleted = SessionAudioRecorder.delete_session("test_export", base_dir=str(tmp_path))
    assert deleted is True
    assert not (tmp_path / "test_export").exists()


@pytest.mark.asyncio
async def test_orchestrator_session_recording_integration(tmp_path):
    recorder = SessionAudioRecorder(base_dir=str(tmp_path), session_id="orch_session")
    vad = SileroVAD(threshold=0.5)
    turn_mgr = TurnManager(vad=vad, min_speech_duration_ms=50, min_silence_duration_ms=50)
    stt = LocalSTT()
    llm = LocalLLM()
    tts = LocalTTS()
    cancellation = CancellationCoordinator()

    stt.transcribe = MagicMock(return_value="Testing session recording.")
    llm.stream_generate = MagicMock(return_value=iter(["Session recorded successfully."]))

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_mgr,
        stt=stt,
        llm=llm,
        tts=tts,
        cancellation=cancellation,
        session_recorder=recorder,
    )
    orchestrator._wait_playback = MagicMock(side_effect=lambda *args: asyncio.sleep(0.001, result=True))

    turn_id = cancellation.start_new_turn()
    dummy_audio = np.zeros(16000, dtype=np.float32)

    await orchestrator._handle_turn(dummy_audio, turn_id)

    # Small pause to allow async executor write
    await asyncio.sleep(0.05)

    manifest = recorder._load_manifest()
    assert manifest["total_turns"] == 1
    turn_meta = manifest["turns"][0]
    assert turn_meta["turn_id"] == turn_id
    assert turn_meta["user_text"] == "Testing session recording."
    assert "Session recorded successfully." in turn_meta["agent_text"]

