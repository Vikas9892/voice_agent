import httpx
from typing import TYPE_CHECKING, Optional

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

    def generate(self, user_prompt: str) -> str:
        """Generate response text from local Ollama instance with graceful fallback."""
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
            "stream": False,
        }
        response = None
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    response = res.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass

        if not response:
            # shortcut: fallback echo if Ollama server is not running locally yet
            response = f"I received your request: '{user_prompt}'."

        if self.memory:
            self.memory.add_turn(user_prompt, response)

        return response
