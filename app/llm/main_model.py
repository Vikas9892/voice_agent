import json
from typing import TYPE_CHECKING, Iterator, Optional
import httpx

if TYPE_CHECKING:
    from app.conversation.memory import ConversationMemory


class LocalLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        system_prompt: str = "You are a concise voice assistant. Reply in 1-2 short sentences.",
        temperature: float = 0.7,
        max_tokens: int = 150,
        memory: Optional["ConversationMemory"] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = memory

    def set_persona(self, persona):
        """Update system prompt and temperature from a Persona instance."""
        self.system_prompt = persona.system_prompt
        self.temperature = persona.temperature

    def stream_generate(self, user_prompt: str) -> Iterator[str]:
        """Stream response tokens from local Ollama instance with graceful fallback."""
        url = f"{self.base_url}/api/chat"
        if self.memory:
            messages = self.memory.get_chat_messages(self.system_prompt, user_prompt)
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": True,
        }

        yielded_tokens = []
        stream_ok = False
        try:
            with httpx.Client(timeout=30.0) as client:
                with client.stream("POST", url, json=payload) as res:
                    if res.status_code == 200:
                        stream_ok = True
                        for line in res.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("message", {}).get("content", "")
                                if token:
                                    yielded_tokens.append(token)
                                    yield token
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                pass
        except Exception:
            pass

        if not stream_ok or not yielded_tokens:
            # shortcut: fallback echo if Ollama server is not running locally
            fallback = f"I received your request: '{user_prompt}'."
            words = fallback.split()
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yielded_tokens.append(token)
                yield token

        full_response = "".join(yielded_tokens).strip()
        if self.memory and full_response:
            self.memory.add_turn(user_prompt, full_response)

    def generate(self, user_prompt: str) -> str:
        """Generate response text from local Ollama instance with graceful fallback."""
        return "".join(self.stream_generate(user_prompt)).strip()
