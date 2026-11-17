import numpy as np
import pytest

from app.audio.agc import AutoGainController
from app.audio.noise_profiler import AmbientNoiseProfiler
from app.audio.preprocessing import preprocess_audio
from app.audio.vad import SileroVAD
from app.conversation.state import TurnState
from app.conversation.turn_manager import TurnManager


def test_ambient_noise_profiler_tracking():
    profiler = AmbientNoiseProfiler(initial_floor_rms=0.005, alpha=0.1)
    assert profiler.noise_floor_rms == 0.005
    assert profiler.noise_floor_dbfs < -40.0

    # Feed quiet noise frame
    quiet_frame = np.random.normal(0, 0.003, 512).astype(np.float32)
    profiler.update(quiet_frame, is_speech=False)
    assert profiler.noise_floor_rms < 0.005

    # Feed speech frame with is_speech=True: noise floor should NOT increase dramatically
    speech_frame = np.random.normal(0, 0.15, 512).astype(np.float32)
    old_floor = profiler.noise_floor_rms
    profiler.update(speech_frame, is_speech=True)
    assert profiler.noise_floor_rms == old_floor  # Speech frame ignored for background estimate
    assert profiler.current_snr_db > 10.0


def test_adaptive_threshold_scaling():
    profiler = AmbientNoiseProfiler(min_threshold=0.35, max_threshold=0.75)

    # In a quiet room (floor = 0.002)
    profiler.noise_floor_rms = 0.002
    low_thresh = profiler.compute_adaptive_threshold(base_threshold=0.5)
    assert low_thresh < 0.5
    assert low_thresh >= 0.35

    # In a loud room (floor = 0.025)
    profiler.noise_floor_rms = 0.025
    high_thresh = profiler.compute_adaptive_threshold(base_threshold=0.5)
    assert high_thresh > 0.5
    assert high_thresh <= 0.75


def test_noise_profiler_calibration():
    profiler = AmbientNoiseProfiler()
    profiler.start_calibration()
    assert profiler.calibrating is True

    # Feed 10 calibration frames with mean RMS ~0.008
    for _ in range(10):
        frame = np.full(512, 0.008, dtype=np.float32)
        profiler.update(frame)

    result = profiler.finish_calibration()
    assert profiler.calibrating is False
    assert result["noise_floor_rms"] == pytest.approx(0.008, abs=1e-4)
    assert "recommended_threshold" in result


def test_auto_gain_controller_amplification_and_limiting():
    agc = AutoGainController(target_rms=0.05, max_gain=3.0, min_gain=0.5)

    # Test quiet speech amplification
    quiet = np.full(512, 0.01, dtype=np.float32)
    for _ in range(15):
        processed = agc.process_frame(quiet, is_speech=True)
    assert agc.current_gain > 1.0
    assert float(np.sqrt(np.mean(processed**2))) > 0.01

    # Test loud burst soft peak limiting
    loud = np.full(512, 0.95, dtype=np.float32)
    limited = agc.process_frame(loud, is_speech=True)
    assert np.max(limited) <= 1.0
    assert np.min(limited) >= -1.0


def test_agc_integration_in_preprocessing():
    agc = AutoGainController(target_rms=0.08, max_gain=2.5)
    audio = np.full(512, 0.02, dtype=np.float32)
    out = preprocess_audio(audio, agc=agc)
    assert len(out) == len(audio)


def test_turn_manager_with_noise_profiler_and_agc():
    vad = SileroVAD(threshold=0.5)
    profiler = AmbientNoiseProfiler(initial_floor_rms=0.005)
    agc = AutoGainController(enabled=True)

    tm = TurnManager(
        vad=vad,
        min_speech_duration_ms=64,
        min_silence_duration_ms=64,
        noise_profiler=profiler,
        agc=agc,
    )

    # Background frame: profiler updates, state remains IDLE
    bg_frame = np.full(512, 0.002, dtype=np.float32)
    state, audio = tm.process_frame(bg_frame)
    assert state == TurnState.IDLE
    assert audio is None
    assert profiler.noise_floor_rms < 0.005

