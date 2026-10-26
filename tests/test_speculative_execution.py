import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.conversation.cancellation import CancellationCoordinator
from app.conversation.gate import PlaybackGate
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager
from app.intent.filler_policy import FillerPolicy
from app.intent.readiness import IntentReadinessDetector


def test_playback_gate_speech_collision_prevention():
    policy = FillerPolicy()
    gate = PlaybackGate(policy)

    # When user is speaking, gate MUST block audio playback
    gate.set_user_speaking(True)
    assert not gate.can_play_audio()
    assert not gate.should_play_filler("Looking into that.")
    assert policy.stats()["discarded"] == 1
    assert policy.stats()["played"] == 0

    # When user stops speaking, playback is allowed
    gate.set_user_speaking(False)
    assert gate.can_play_audio()
    assert gate.should_play_filler("Looking into that.")
    assert policy.stats()["played"] == 1


@pytest.mark.asyncio
async def test_speculative_filler_trigger_and_handover():
    mock_vad = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "what is the capital of France"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The capital of France is Paris."

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(800, dtype=np.float32)

    mock_filler_model = MagicMock()
    mock_filler_model.generate_filler.return_value = "Let me check."

    policy = FillerPolicy()
    readiness = IntentReadinessDetector()
    turn_manager = TurnManager(vad=mock_vad)
    coord = CancellationCoordinator()

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=mock_stt,
        llm=mock_llm,
        tts=mock_tts,
        cancellation=coord,
        filler_model=mock_filler_model,
        filler_policy=policy,
        readiness_detector=readiness,
    )

    # 1. Trigger speculation on partial speech audio snapshot
    snapshot = np.zeros(16000, dtype=np.float32)
    await orchestrator._check_and_trigger_speculation(snapshot, current_turn_id=1)

    assert orchestrator.speculative_task is not None
    spec_task = orchestrator.speculative_task

    # 2. Complete turn handover
    with patch("app.conversation.orchestrator.play_audio"):
        await orchestrator._handle_turn(snapshot, turn_id=1, speculative_task=spec_task)

    # Speculative filler was executed and handed off
    mock_filler_model.generate_filler.assert_called()
    mock_llm.generate.assert_called_once()
    assert policy.stats()["played"] == 1


@pytest.mark.asyncio
async def test_speculative_filler_discarded_if_main_ready_first():
    mock_vad = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "what is quantum computing"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Quantum computing uses qubits."

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(800, dtype=np.float32)

    policy = FillerPolicy()
    turn_manager = TurnManager(vad=mock_vad)
    coord = CancellationCoordinator()

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=mock_stt,
        llm=mock_llm,
        tts=mock_tts,
        cancellation=coord,
        filler_model=None,  # No filler model -> no filler played
        filler_policy=policy,
    )

    snapshot = np.zeros(16000, dtype=np.float32)
    with patch("app.conversation.orchestrator.play_audio"):
        await orchestrator._handle_turn(snapshot, turn_id=1)

    assert policy.stats()["played"] == 0
