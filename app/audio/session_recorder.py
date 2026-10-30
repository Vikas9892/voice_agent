import json
import os
import shutil
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class SessionAudioRecorder:
    """Manages audio recording, WAV serialization, and session metadata persistence."""

    def __init__(self, base_dir: str = "data/sessions", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.session_dir / "manifest.json"
        self._init_manifest()

    def _init_manifest(self):
        if not self.manifest_file.exists():
            data = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "created_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_turns": 0,
                "turns": [],
            }
            self._save_manifest(data)

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"session_id": self.session_id, "turns": [], "total_turns": 0}

    def _save_manifest(self, data: Dict[str, Any]):
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sample_rate: int):
        """Write float32 numpy audio array to 16-bit PCM WAV."""
        if audio is None or len(audio) == 0:
            return
        clipped = np.clip(audio, -1.0, 1.0)
        int16_data = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())

    def record_turn(
        self,
        turn_id: int,
        user_audio: Optional[np.ndarray] = None,
        agent_audio: Optional[np.ndarray] = None,
        user_sample_rate: int = 16000,
        agent_sample_rate: int = 22050,
        user_text: str = "",
        agent_text: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        persona: str = "assistant",
    ) -> Dict[str, Any]:
        """Save user and agent audio to WAV and append turn record to session manifest."""
        turn_entry = {
            "turn_id": turn_id,
            "timestamp": time.time(),
            "user_text": user_text,
            "agent_text": agent_text,
            "persona": persona,
            "user_audio_file": None,
            "agent_audio_file": None,
            "metrics": metrics or {},
        }

        if user_audio is not None and len(user_audio) > 0:
            user_wav = f"turn_{turn_id}_user.wav"
            self._write_wav(self.session_dir / user_wav, user_audio, user_sample_rate)
            turn_entry["user_audio_file"] = user_wav

        if agent_audio is not None and len(agent_audio) > 0:
            agent_wav = f"turn_{turn_id}_agent.wav"
            self._write_wav(self.session_dir / agent_wav, agent_audio, agent_sample_rate)
            turn_entry["agent_audio_file"] = agent_wav

        manifest = self._load_manifest()
        manifest["turns"].append(turn_entry)
        manifest["total_turns"] = len(manifest["turns"])
        manifest["updated_at"] = time.time()
        self._save_manifest(manifest)
        return turn_entry

    def get_audio_path(self, turn_id: int, speaker: str) -> Optional[Path]:
        """Get absolute path to recorded audio for turn and speaker ('user' or 'agent')."""
        filename = f"turn_{turn_id}_{speaker}.wav"
        file_path = self.session_dir / filename
        return file_path if file_path.exists() else None

    @classmethod
    def list_all_sessions(cls, base_dir: str = "data/sessions") -> List[Dict[str, Any]]:
        """List all recorded sessions with summary statistics."""
        base = Path(base_dir)
        if not base.exists():
            return []
        sessions = []
        for sdir in sorted(base.iterdir(), reverse=True):
            if sdir.is_dir():
                mf = sdir / "manifest.json"
                if mf.exists():
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            sessions.append({
                                "session_id": data.get("session_id", sdir.name),
                                "created_iso": data.get("created_iso"),
                                "total_turns": data.get("total_turns", 0),
                            })
                    except Exception:
                        pass
        return sessions

    @classmethod
    def load_session(cls, session_id: str, base_dir: str = "data/sessions") -> Optional[Dict[str, Any]]:
        """Load full session manifest."""
        mf = Path(base_dir) / session_id / "manifest.json"
        if mf.exists():
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    @classmethod
    def export_session(cls, session_id: str, fmt: str = "json", base_dir: str = "data/sessions") -> Optional[str]:
        """Export session as JSON string or formatted Markdown."""
        data = cls.load_session(session_id, base_dir=base_dir)
        if not data:
            return None
        if fmt.lower() in ("markdown", "md"):
            lines = [
                f"# Voice Session: {data.get('session_id')}",
                f"- **Created:** {data.get('created_iso', 'Unknown')}",
                f"- **Total Turns:** {data.get('total_turns', 0)}",
                "",
                "## Conversation Transcript",
                "",
            ]
            for t in data.get("turns", []):
                tid = t.get("turn_id")
                u_text = t.get("user_text", "")
                a_text = t.get("agent_text", "")
                persona = t.get("persona", "assistant")
                perceived = t.get("metrics", {}).get("perceived_latency_ms", "--")
                lines.append(f"### Turn {tid} ({persona}) — Latency: {perceived}ms")
                lines.append(f"- **User:** {u_text}")
                lines.append(f"- **Agent:** {a_text}")
                lines.append("")
            return "\n".join(lines)
        return json.dumps(data, indent=2)

    @classmethod
    def delete_session(cls, session_id: str, base_dir: str = "data/sessions") -> bool:
        """Safely delete session folder and its audio files."""
        sdir = Path(base_dir) / session_id
        if sdir.exists() and sdir.is_dir():
            shutil.rmtree(sdir)
            return True
        return False
