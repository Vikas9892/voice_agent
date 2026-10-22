import numpy as np
from unittest.mock import MagicMock
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager


def test_turn_manager_lifecycle():
    mock_vad = MagicMock()
    # 512 samples frame
    frame = np.zeros(512, dtype=np.float32)

    manager = TurnManager(
        vad=mock_vad,
        sample_rate=16000,
        frame_samples=512,
        min_speech_duration_ms=64,   # 2 frames
        min_silence_duration_ms=96,  # 3 frames
        barge_in_frames=2,
    )

    assert manager.state == TurnState.IDLE

    # 1. User starts speaking (2 frames needed)
    mock_vad.is_speech.return_value = True
    state, audio = manager.process_frame(frame)
    assert state == TurnState.IDLE  # Only 1 frame so far

    state, audio = manager.process_frame(frame)
    assert state == TurnState.USER_SPEAKING  # 2nd frame triggers speaking

    # 2. User pauses (1 silence frame)
    mock_vad.is_speech.return_value = False
    state, audio = manager.process_frame(frame)
    assert state == TurnState.USER_PAUSED

    # 3. User resumes speaking
    mock_vad.is_speech.return_value = True
    state, audio = manager.process_frame(frame)
    assert state == TurnState.USER_SPEAKING

    # 4. User finishes speaking: silence for 3 frames triggers finalization
    mock_vad.is_speech.return_value = False
    state, audio = manager.process_frame(frame)  # pause frame 1
    assert state == TurnState.USER_PAUSED
    assert audio is None

    state, audio = manager.process_frame(frame)  # pause frame 2
    assert state == TurnState.USER_PAUSED
    assert audio is None

    state, audio = manager.process_frame(frame)  # pause frame 3 -> endpoint!
    assert state == TurnState.USER_TURN_FINALIZING
    assert audio is not None
    assert len(audio) > 0


def test_barge_in_detection():
    mock_vad = MagicMock()
    frame = np.zeros(512, dtype=np.float32)

    manager = TurnManager(vad=mock_vad, barge_in_frames=2)
    manager.set_agent_speaking()
    assert manager.state == TurnState.AGENT_SPEAKING

    mock_vad.is_speech.return_value = True
    state, _ = manager.process_frame(frame)
    assert state == TurnState.AGENT_SPEAKING  # 1 frame not enough

    state, _ = manager.process_frame(frame)
    assert state == TurnState.INTERRUPTED  # 2nd frame triggers barge-in!
