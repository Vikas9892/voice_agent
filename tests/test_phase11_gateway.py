import json
import pytest
from starlette.testclient import TestClient

from app.api.server import create_app
from app.llm.main_model import LocalLLM
from app.llm.personas import Persona, PersonaManager, default_persona_manager


def test_persona_manager():
    pm = PersonaManager()
    personas = pm.list_personas()
    assert len(personas) >= 4
    names = [p["name"] for p in personas]
    assert "assistant" in names
    assert "tutor" in names

    assert pm.set_active("tutor") is True
    assert pm.active_persona.name == "tutor"
    assert pm.set_active("nonexistent") is False


def test_llm_persona_integration():
    pm = PersonaManager()
    llm = LocalLLM()
    tutor_p = pm.get_persona("tutor")
    llm.set_persona(tutor_p)
    assert llm.system_prompt == tutor_p.system_prompt
    assert llm.temperature == tutor_p.temperature


def test_personas_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # List personas
    resp = client.get("/api/personas")
    assert resp.status_code == 200
    data = resp.json()
    assert "personas" in data
    assert "active" in data

    # Change active persona
    resp = client.post("/api/personas/active", json={"name": "technologist"})
    assert resp.status_code == 200
    assert resp.json()["active"] == "technologist"

    # Invalid persona
    resp = client.post("/api/personas/active", json={"name": "invalid_xyz"})
    assert resp.status_code == 404


def test_websocket_voice_handshake():
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/voice") as ws:
        init_data = ws.receive_json()
        assert init_data["type"] == "voice_session_ready"
        assert "sample_rate" in init_data

        # Test ping
        ws.send_text(json.dumps({"action": "ping"}))
        reply = ws.receive_json()
        assert reply["type"] == "pong"

        # Test set_persona
        ws.send_text(json.dumps({"action": "set_persona", "name": "tutor"}))
        reply = ws.receive_json()
        assert reply["type"] == "persona_updated"
        assert reply["persona"] == "tutor"
