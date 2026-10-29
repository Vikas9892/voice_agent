import asyncio
import json
import os
import time
from typing import Any, Dict, List, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from app.config import settings
from app.llm.personas import default_persona_manager
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

    @app.get("/api/personas")
    async def get_personas():
        return {
            "personas": default_persona_manager.list_personas(),
            "active": default_persona_manager.active_persona.name,
        }

    @app.post("/api/personas/active")
    async def set_active_persona(payload: Dict[str, str]):
        name = payload.get("name", "")
        if default_persona_manager.set_active(name):
            persona = default_persona_manager.active_persona
            if orchestrator and hasattr(orchestrator, "llm") and hasattr(orchestrator.llm, "set_persona"):
                orchestrator.llm.set_persona(persona)
            await broadcaster.broadcast("persona_changed", {"name": persona.name, "display_name": persona.display_name})
            return {"status": "ok", "active": persona.name}
        raise HTTPException(status_code=404, detail="Persona not found")

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
                    "active_persona": default_persona_manager.active_persona.name,
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

    @app.websocket("/ws/voice")
    async def websocket_voice(websocket: WebSocket):
        """Full-duplex audio stream for browser Web Audio microphone capture and playback."""
        await websocket.accept()
        audio_buffer = bytearray()
        chunk_size_bytes = settings.audio_chunk_samples * 4  # 512 samples * 4 bytes for float32
        turn_manager = getattr(orchestrator, "turn_manager", None) if orchestrator else None

        await websocket.send_text(
            json.dumps({
                "type": "voice_session_ready",
                "sample_rate": settings.audio_sample_rate,
                "persona": default_persona_manager.active_persona.name,
            })
        )

        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    data = message["bytes"]
                    audio_buffer.extend(data)
                    while len(audio_buffer) >= chunk_size_bytes:
                        raw_chunk = bytes(audio_buffer[:chunk_size_bytes])
                        del audio_buffer[:chunk_size_bytes]
                        frame = np.frombuffer(raw_chunk, dtype=np.float32).copy()

                        if turn_manager:
                            new_state, completed_audio = turn_manager.process_frame(frame)
                            state_name = new_state.name if hasattr(new_state, "name") else str(new_state)
                            if state_name == "INTERRUPTED":
                                await websocket.send_text(json.dumps({"type": "barge_in"}))
                            elif completed_audio is not None and len(completed_audio) > 0:
                                await websocket.send_text(json.dumps({"type": "processing_turn"}))
                                loop = asyncio.get_running_loop()
                                transcript = ""
                                if orchestrator and hasattr(orchestrator, "stt"):
                                    transcript = await loop.run_in_executor(
                                        getattr(orchestrator, "executor", None),
                                        orchestrator.stt.transcribe,
                                        completed_audio,
                                    )

                                if transcript and transcript.strip():
                                    await websocket.send_text(json.dumps({"type": "transcript", "text": transcript}))
                                    tool_matcher = getattr(orchestrator, "tool_matcher", None)
                                    tool_match = tool_matcher.match_and_execute(transcript) if tool_matcher else None
                                    if tool_match:
                                        _, _, response = tool_match
                                    else:
                                        response = (
                                            await loop.run_in_executor(
                                                getattr(orchestrator, "executor", None),
                                                orchestrator.llm.generate,
                                                transcript,
                                            )
                                            if orchestrator and hasattr(orchestrator, "llm")
                                            else f"Echo: {transcript}"
                                        )

                                    await websocket.send_text(json.dumps({"type": "agent_response", "text": response}))

                                    if orchestrator and hasattr(orchestrator, "tts"):
                                        tts_audio = await loop.run_in_executor(
                                            getattr(orchestrator, "executor", None),
                                            orchestrator.tts.synthesize,
                                            response,
                                            settings.tts_sample_rate,
                                        )
                                        if len(tts_audio) > 0:
                                            await websocket.send_bytes(tts_audio.tobytes())
                                            await websocket.send_text(
                                                json.dumps({
                                                    "type": "playback_complete",
                                                    "sample_rate": settings.tts_sample_rate,
                                                })
                                            )
                elif "text" in message and message["text"]:
                    try:
                        req = json.loads(message["text"])
                        action = req.get("action")
                        if action == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        elif action == "interrupt":
                            if orchestrator and hasattr(orchestrator, "cancellation"):
                                orchestrator.cancellation.cancel_current_turn()
                            await websocket.send_text(json.dumps({"type": "barge_in"}))
                        elif action == "set_persona":
                            pname = req.get("name", "")
                            if default_persona_manager.set_active(pname):
                                if orchestrator and hasattr(orchestrator, "llm"):
                                    orchestrator.llm.set_persona(default_persona_manager.active_persona)
                                await websocket.send_text(json.dumps({"type": "persona_updated", "persona": pname}))
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

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
