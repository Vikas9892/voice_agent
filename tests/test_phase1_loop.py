import numpy as np
from unittest.mock import MagicMock
from app.main import run_turn


def test_run_turn_with_speech():
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "What time is it?"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "It is 12 o'clock."

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(1000, dtype=np.float32)

    dummy_audio = np.zeros(16000, dtype=np.float32)
    metrics = run_turn(mock_stt, mock_llm, mock_tts, duration_sec=1.0, simulated_audio=dummy_audio)

    mock_stt.transcribe.assert_called_once()
    mock_llm.generate.assert_called_once_with("What time is it?")
    mock_tts.synthesize.assert_called_once_with("It is 12 o'clock.", sample_rate=22050)

    assert "stt_ms" in metrics
    assert "llm_ms" in metrics
    assert "tts_ms" in metrics
    assert "total_turnaround_ms" in metrics


def test_run_turn_no_speech():
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = ""

    mock_llm = MagicMock()
    mock_tts = MagicMock()

    dummy_audio = np.zeros(16000, dtype=np.float32)
    metrics = run_turn(mock_stt, mock_llm, mock_tts, duration_sec=1.0, simulated_audio=dummy_audio)

    mock_stt.transcribe.assert_called_once()
    mock_llm.generate.assert_not_called()
    mock_tts.synthesize.assert_not_called()
    assert "stt_ms" in metrics
