from starlette.testclient import TestClient
from app.api.server import create_app


def test_dashboard_index_serving():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Voice Agent Dashboard" in resp.text
    assert "waveform-canvas" in resp.text
    assert "Live Speech Monitor" in resp.text


def test_dashboard_static_assets_serving():
    app = create_app()
    client = TestClient(app)

    # CSS
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert "--bg-primary" in css_resp.text

    # JS
    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "connectWs" in js_resp.text
