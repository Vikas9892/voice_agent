import numpy as np
import pytest
from starlette.testclient import TestClient
from unittest.mock import MagicMock

from app.api.server import create_app
from app.audio.agc import AutoGainController
from app.audio.noise_profiler import AmbientNoiseProfiler
from app.audio.vad import SileroVAD
from app.conversation.orchestrator import StreamingVoiceOrchestrator
from app.conversation.turn_manager import TurnManager


def test_audio_noise_profile_endpoints():
    profiler = AmbientNoiseProfiler(initial_floor_rms=0.006)
    agc = AutoGainController(target_rms=0.04, enabled=True)

    app = create_app()
    app.state.noise_profiler = profiler
    app.state.agc = agc
    client = TestClient(app)

    # 1. GET /api/audio/noise-profile
    resp = client.get("/api/audio/noise-profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "noise_floor_rms" in data
    assert "noise_floor_dbfs" in data
    assert "current_snr_db" in data
    assert "adaptive_threshold" in data
    assert data["agc_enabled"] is True

    # 2. POST /api/audio/calibrate
    resp_cal = client.post("/api/audio/calibrate")
    assert resp_cal.status_code == 200
    cal_data = resp_cal.json()
    assert cal_data["status"] == "calibrated"
    assert "profile" in cal_data
    assert "recommended_threshold" in cal_data["profile"]

    # 3. POST /api/audio/agc
    resp_agc = client.post("/api/audio/agc", json={"enabled": False, "target_rms": 0.08})
    assert resp_agc.status_code == 200
    assert resp_agc.json()["enabled"] is False
    assert resp_agc.json()["target_rms"] == 0.08
    assert agc.enabled is False
    assert agc.target_rms == 0.08


def test_orchestrator_audio_callback_emits_noise_metadata():
    vad = SileroVAD(threshold=0.5)
    profiler = AmbientNoiseProfiler(initial_floor_rms=0.005)
    agc = AutoGainController(enabled=True)
    tm = TurnManager(vad=vad, noise_profiler=profiler, agc=agc)

    events_emitted = []
    broadcaster = MagicMock()
    broadcaster.broadcast_threadsafe = lambda ev, data: events_emitted.append((ev, data))

    orchestrator = StreamingVoiceOrchestrator(
        turn_manager=tm,
        stt=MagicMock(),
        llm=MagicMock(),
        tts=MagicMock(),
        broadcaster=broadcaster,
    )

    # Simulate 4 audio callback frames
    frame = np.full((512, 1), 0.003, dtype=np.float32)
    for _ in range(4):
        orchestrator._audio_callback(frame, 512, None, None)

    # Verify audio_level event with noise metadata
    audio_level_events = [data for ev, data in events_emitted if ev == "audio_level"]
    assert len(audio_level_events) >= 1
    last_event = audio_level_events[-1]
    assert "noise_floor_rms" in last_event
    assert "snr_db" in last_event
    assert "adaptive_threshold" in last_event

