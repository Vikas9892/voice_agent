import asyncio
import json
import pytest
from starlette.testclient import TestClient

from app.api.server import create_app, broadcaster
from app.observability.metrics import MetricsLogger
from app.config import settings


def test_api_health_endpoint():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "local-voice-agent"
    assert "uptime_seconds" in data


def test_api_config_endpoints():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    cfg = resp.json()
    assert "vad_threshold" in cfg
    assert "barge_in_enabled" in cfg

    # Test update
    orig_thresh = settings.vad_threshold
    try:
        update_resp = client.post("/api/config", json={"vad_threshold": 0.65})
        assert update_resp.status_code == 200
        assert settings.vad_threshold == 0.65
    finally:
        settings.vad_threshold = orig_thresh


def test_api_metrics_endpoint():
    logger = MetricsLogger()
    m = logger.create_turn_metrics(turn_id=1)
    m.mark("user_turn_end_detected")
    m.mark("agent_first_audio_played")
    logger.record_turn(m)

    app = create_app(metrics_logger=logger)
    client = TestClient(app)
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["turn_history"]) == 1
    assert "aggregates" in data


def test_websocket_connection_and_ping():
    app = create_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/events") as ws:
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connection_established"
        assert "vad_threshold" in init_msg["data"]

        ws.send_json({"action": "ping"})
        reply = ws.receive_json()
        assert reply["type"] == "pong"


@pytest.mark.asyncio
async def test_event_broadcaster():
    b = broadcaster
    # Ensure broadcast doesn't crash when empty
    await b.broadcast("test_event", {"val": 123})
    assert len(b.active_connections) == 0
