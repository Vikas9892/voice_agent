import asyncio
import json
import os
import time
from typing import Any, Dict, List, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.observability.metrics import MetricsLogger


class EventBroadcaster:
    """Manages active WebSocket connections and broadcasts real-time pipeline events."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Broadcast structured event payload to all connected clients."""
        if not self.active_connections:
            return
        payload = json.dumps({"type": event_type, "data": data, "timestamp": time.time()})
        stale = set()
        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.add(ws)
        if stale:
            async with self._lock:
                self.active_connections -= stale

    def broadcast_threadsafe(self, event_type: str, data: Dict[str, Any], loop: asyncio.AbstractEventLoop | None = None):
        """Helper to broadcast from non-async thread or callbacks."""
        try:
            target_loop = loop or asyncio.get_running_loop()
            target_loop.create_task(self.broadcast(event_type, data))
        except RuntimeError:
            pass


broadcaster = EventBroadcaster()


def create_app(orchestrator=None, metrics_logger: MetricsLogger | None = None) -> FastAPI:
    """Build and configure the FastAPI web application."""
    start_time = time.time()
    logger = metrics_logger or MetricsLogger()

    app = FastAPI(
        title="Local Voice Agent Dashboard",
        description="Local-first real-time speech telemetry and control interface",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State references
    app.state.orchestrator = orchestrator
    app.state.metrics_logger = logger
    app.state.broadcaster = broadcaster

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "local-voice-agent",
            "uptime_seconds": round(time.time() - start_time, 2),
            "streaming_active": orchestrator is not None,
            "connected_clients": len(broadcaster.active_connections),
        }

    @app.get("/api/config")
    async def get_config():
        return {
            "audio_sample_rate": settings.audio_sample_rate,
            "vad_threshold": settings.vad_threshold,
            "vad_min_silence_duration_ms": settings.vad_min_silence_duration_ms,
            "stt_model_size": settings.stt_model_size,
            "stt_compute_type": settings.stt_compute_type,
            "llm_model": settings.llm_model,
            "tts_voice_model": settings.tts_voice_model,
            "barge_in_enabled": settings.barge_in_enabled,
        }

    @app.post("/api/config")
    async def update_config(updates: Dict[str, Any]):
        for key, value in updates.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
                if orchestrator and hasattr(orchestrator, "turn_manager") and key == "vad_threshold":
                    orchestrator.turn_manager.vad.threshold = float(value)
        await broadcaster.broadcast("config_updated", updates)
        return {"status": "updated", "config": updates}

    @app.get("/api/metrics")
    async def get_metrics():
        log = app.state.metrics_logger
        return {
            "turn_history": log.history[-20:],
            "aggregates": log.aggregate_kpis(),
        }

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        await broadcaster.connect(websocket)
        # Send initial handshake state
        await websocket.send_text(
            json.dumps({
                "type": "connection_established",
                "data": {
                    "uptime_seconds": round(time.time() - start_time, 2),
                    "vad_threshold": settings.vad_threshold,
                    "barge_in_enabled": settings.barge_in_enabled,
                },
                "timestamp": time.time(),
            })
        )
        try:
            while True:
                msg = await websocket.receive_text()
                try:
                    data = json.loads(msg)
                    if data.get("action") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
                    elif data.get("action") == "toggle_barge_in":
                        settings.barge_in_enabled = not settings.barge_in_enabled
                        await broadcaster.broadcast("barge_in_toggled", {"enabled": settings.barge_in_enabled})
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            await broadcaster.disconnect(websocket)
        except Exception:
            await broadcaster.disconnect(websocket)

    # Static UI directory mounting if exists
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def serve_index():
            index_path = os.path.join(static_dir, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            return HTMLResponse(content="<h1>Voice Agent API Active</h1><p>UI loading...</p>")

    return app
