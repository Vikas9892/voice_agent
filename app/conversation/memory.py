import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnRecord:
    user: str
    agent: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Manages conversational dialogue history with sliding-window token budgeting and session persistence."""

    def __init__(
        self,
        max_turns: int = 5,
        max_chars_per_turn: int = 240,
        session_dir: str = "sessions",
    ):
        self.max_turns = max_turns
        self.max_chars_per_turn = max_chars_per_turn
        self.session_dir = session_dir
        self.turns: List[TurnRecord] = []
        self.rolling_summary: Optional[str] = None

    def add_turn(self, user_text: str, agent_text: str):
        """Append a completed dialogue turn and apply sliding-window compression."""
        u_clean = user_text.strip()[: self.max_chars_per_turn]
        a_clean = agent_text.strip()[: self.max_chars_per_turn]
        if not u_clean:
            return
        self.turns.append(TurnRecord(user=u_clean, agent=a_clean))
        if len(self.turns) > self.max_turns:
            self._compress_oldest_turns()

    def _compress_oldest_turns(self):
        """Compress expired turns into a rolling context summary to keep LLM TTFT sub-300ms."""
        evicted = self.turns.pop(0)
        topic_snip = f"User asked '{evicted.user[:50]}...', agent replied '{evicted.agent[:50]}...'."
        if self.rolling_summary:
            self.rolling_summary = f"{self.rolling_summary} Then: {topic_snip}"
            # Keep rolling summary bounded
            if len(self.rolling_summary) > 200:
                self.rolling_summary = self.rolling_summary[-200:]
        else:
            self.rolling_summary = f"Earlier context: {topic_snip}"

    def get_chat_messages(self, system_prompt: str, current_user_prompt: str) -> List[Dict[str, str]]:
        """Construct Ollama chat messages payload with system prompt, compressed context, and history."""
        messages: List[Dict[str, str]] = []

        sys_content = system_prompt
        if self.rolling_summary:
            sys_content += f"\n[Prior Dialogue Context: {self.rolling_summary}]"
        messages.append({"role": "system", "content": sys_content})

        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.agent})

        messages.append({"role": "user", "content": current_user_prompt})
        return messages

    def save_session(self, session_id: str = "default_session") -> str:
        """Persist dialogue session to disk."""
        os.makedirs(self.session_dir, exist_ok=True)
        path = os.path.join(self.session_dir, f"{session_id}.json")
        data = {
            "session_id": session_id,
            "saved_at": time.time(),
            "rolling_summary": self.rolling_summary,
            "turns": [asdict(t) for t in self.turns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def load_session(self, session_id: str = "default_session") -> bool:
        """Load dialogue session from disk."""
        path = os.path.join(self.session_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.rolling_summary = data.get("rolling_summary")
            self.turns = [
                TurnRecord(user=t["user"], agent=t["agent"], timestamp=t.get("timestamp", time.time()))
                for t in data.get("turns", [])
            ]
            return True
        except Exception:
            return False

    def clear(self):
        """Reset conversation memory."""
        self.turns.clear()
        self.rolling_summary = None
