import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.conversation.cancellation import CancellationCoordinator
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager


def test_cancellation_coordinator():
    coord = CancellationCoordinator()
    turn_1 = coord.start_new_turn()
    assert turn_1 == 1
    assert not coord.is_cancelled()

    with patch("app.conversation.cancellation.stop_playback") as mock_stop:
        coord.cancel_current_turn()
        assert coord.is_cancelled()
        mock_stop.assert_called_once()

    turn_2 = coord.start_new_turn()
    assert turn_2 == 2
    assert not coord.is_cancelled()


@pytest.mark.asyncio
async def test_orchestrator_barge_in_cancels_playback():
    mock_vad = MagicMock()
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "Tell me a story"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Once upon a time in a faraway kingdom..."

    mock_tts = MagicMock()
    # 2 seconds of audio
    mock_tts.synthesize.return_value = np.zeros(44100, dtype=np.float32)

    turn_manager = TurnManager(vad=mock_vad)
    coord = CancellationCoordinator()

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=turn_manager,
        stt=mock_stt,
        llm=mock_llm,
        tts=mock_tts,
        cancellation=coord,
    )

    dummy_audio = np.zeros(16000, dtype=np.float32)

    with patch("app.conversation.orchestrator.play_audio") as mock_play, \
         patch("app.conversation.orchestrator.stop_playback") as mock_stop:

        # Launch handle_turn task
        task = asyncio.create_task(orchestrator._handle_turn(dummy_audio, turn_id=1))

        # Give it a brief moment to start playback
        await asyncio.sleep(0.05)
        assert turn_manager.state.value == "AGENT_SPEAKING"

        # Simulate user barge-in interruption!
        coord.cancel_current_turn()

        # Wait for task to finish aborting
        await task

        mock_stop.assert_called()
