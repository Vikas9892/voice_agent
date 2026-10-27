import json
import os
from unittest.mock import MagicMock, patch
import numpy as np

from app.audio.vad import SileroVAD
from app.config import settings
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager
from app.lifecycle import PrewarmingManager, configure_cpu_affinity
from app.stt.local_stt import LocalSTT
from benchmarks.tune_pipeline import profile_chunk_sizes, run_tuning_suite


def test_vad_variable_chunk_sizes():
    vad = SileroVAD()
    for samples in (256, 512, 1024):
        frame = np.zeros(samples, dtype=np.float32)
        prob = vad.get_speech_prob(frame)
        assert 0.0 <= prob <= 1.0


def test_turn_manager_chunk_size_scaling():
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True

    # 16ms frames (256 samples @ 16kHz)
    tm_16 = TurnManager(vad=mock_vad, sample_rate=16000, frame_samples=256, min_speech_duration_ms=160)
    assert tm_16.frame_duration_ms == 16.0
    assert tm_16.min_speech_frames == 10

    # 32ms frames (512 samples @ 16kHz)
    tm_32 = TurnManager(vad=mock_vad, sample_rate=16000, frame_samples=512, min_speech_duration_ms=160)
    assert tm_32.frame_duration_ms == 32.0
    assert tm_32.min_speech_frames == 5

    # 64ms frames (1024 samples @ 16kHz)
    tm_64 = TurnManager(vad=mock_vad, sample_rate=16000, frame_samples=1024, min_speech_duration_ms=160)
    assert tm_64.frame_duration_ms == 64.0
    assert tm_64.min_speech_frames == 2


def test_cpu_affinity_configuration():
    configure_cpu_affinity(num_threads=4)
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    assert os.environ.get("MKL_NUM_THREADS") == "4"

    mock_vad = MagicMock()
    mock_stt = MagicMock()
    mock_llm = MagicMock()
    mock_tts = MagicMock()

    mgr = PrewarmingManager(vad=mock_vad, stt=mock_stt, llm=mock_llm, tts=mock_tts, max_workers=3)
    assert mgr.executor._max_workers == 3
    mgr.shutdown()


def test_local_stt_thread_and_beam_config():
    with patch("app.stt.local_stt.WhisperModel") as mock_whisper:
        stt = LocalSTT(
            model_size="base.en",
            device="cpu",
            compute_type="int8",
            beam_size=2,
            cpu_threads=4,
        )
        assert stt.beam_size == 2
        mock_whisper.assert_called_once_with(
            "base.en",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )


def test_tune_pipeline_mocked(tmp_path):
    output_file = tmp_path / "tuning_results.json"
    with patch("benchmarks.tune_pipeline.LocalSTT") as mock_stt_cls:
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "testing pipeline tuning"
        mock_stt_cls.return_value = mock_stt

        res = run_tuning_suite(output_file=str(output_file))
        assert "chunk_size_profiles" in res
        assert "16ms" in res["chunk_size_profiles"]
        assert "32ms" in res["chunk_size_profiles"]
        assert "64ms" in res["chunk_size_profiles"]
        assert "recommendations" in res
        assert os.path.exists(output_file)
