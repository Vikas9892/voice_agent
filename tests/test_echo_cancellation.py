from unittest.mock import MagicMock
import numpy as np
import pytest

from app.audio.echo_cancellation import AcousticEchoSuppressor
from app.audio.vad import SileroVAD
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager


def test_echo_suppressor_attenuation():
    aes = AcousticEchoSuppressor(echo_energy_threshold=0.15, barge_in_energy_threshold=0.28)
    aes.set_playback_active(True)

    # Low-amplitude speaker leakage frame
    low_frame = np.full(512, 0.05, dtype=np.float32)
    filtered, is_echo = aes.process_frame(low_frame)
    assert is_echo is True
    assert np.max(filtered) < np.max(low_frame)

    # High-amplitude barge-in frame
    loud_frame = np.full(512, 0.40, dtype=np.float32)
    filtered_loud, is_loud_echo = aes.process_frame(loud_frame)
    assert is_loud_echo is False
    assert np.allclose(filtered_loud, loud_frame)


def test_turn_manager_with_echo_suppressor():
    mock_vad = MagicMock(spec=SileroVAD)
    mock_vad.is_speech.return_value = True  # VAD would trigger on anything

    aes = AcousticEchoSuppressor(echo_energy_threshold=0.15, barge_in_energy_threshold=0.28)
    tm = TurnManager(
        vad=mock_vad,
        barge_in_frames=2,
        echo_suppressor=aes,
    )

    # Agent is speaking
    tm.set_agent_speaking()
    assert tm.state == TurnState.AGENT_SPEAKING

    # Feed speaker leakage audio frame (RMS ~0.08)
    leakage_frame = np.full(512, 0.08, dtype=np.float32)
    state, _ = tm.process_frame(leakage_frame)
    # Should remain in AGENT_SPEAKING because echo is suppressed!
    assert state == TurnState.AGENT_SPEAKING
    assert tm.consecutive_speech_frames == 0

    # Feed genuine loud user barge-in frames (RMS ~0.35)
    barge_frame = np.full(512, 0.35, dtype=np.float32)
    tm.process_frame(barge_frame)
    state, _ = tm.process_frame(barge_frame)
    # 2 consecutive frames exceed barge_in_frames -> triggers INTERRUPTED!
    assert state == TurnState.INTERRUPTED
