from unittest.mock import MagicMock
import numpy as np

from app.lifecycle import PrewarmingManager


def test_prewarming_manager_warmup_all():
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = False

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = ""

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "hello"

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(100, dtype=np.float32)

    manager = PrewarmingManager(
        vad=mock_vad,
        stt=mock_stt,
        llm=mock_llm,
        tts=mock_tts,
        max_workers=2,
    )

    timings = manager.warmup_all()

    mock_vad.is_speech.assert_called_once()
    mock_stt.transcribe.assert_called_once()
    mock_llm.generate.assert_called_once_with("hi")
    mock_tts.synthesize.assert_called_once_with("ready")

    assert "vad_ms" in timings
    assert "stt_ms" in timings
    assert "llm_ms" in timings
    assert "tts_ms" in timings
    assert "total_cold_start_ms" in timings

    # Verify executor works
    future = manager.executor.submit(lambda x: x * 2, 21)
    assert future.result() == 42

    manager.shutdown()
