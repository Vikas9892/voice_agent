import asyncio
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.conversation.cancellation import CancellationCoordinator
from app.conversation.gate import PlaybackGate
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager
from app.intent.filler_policy import FillerPolicy


def test_playback_gate_discard():
    policy = FillerPolicy()
    gate = PlaybackGate(policy)

    # 1. Main answer arrives first -> filler should be discarded!
    gate.mark_main_ready()
    assert not gate.should_play_filler("Let me check that.")
    assert policy.stats()["discarded"] == 1
    assert policy.stats()["played"] == 0


def test_playback_gate_play():
    policy = FillerPolicy()
    gate = PlaybackGate(policy)

    # 2. Main answer not ready -> filler eligible
    assert gate.should_play_filler("Let me check that.")
    assert policy.stats()["played"] == 1
    assert policy.stats()["discarded"] == 0


@pytest.mark.asyncio
async def test_orchestrator_filler_execution():
    mock_vad = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "Can you compare faster whisper and whisper cpp?"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Faster whisper is optimized with CTranslate2."

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(1000, dtype=np.float32)

    mock_filler_model = MagicMock()
    mock_filler_model.generate_filler.return_value = "Let me check."

    policy = FillerPolicy()
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
    )

    dummy_audio = np.zeros(16000, dtype=np.float32)

    with patch("app.conversation.orchestrator.play_audio"):
        await orchestrator._handle_turn(dummy_audio, turn_id=1)

    # Verify that filler model was invoked on eligible query
    mock_filler_model.generate_filler.assert_called_once()
    mock_llm.generate.assert_called_once()
    assert policy.stats()["generated"] == 1
