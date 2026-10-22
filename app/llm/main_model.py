import httpx


class LocalLLM:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        system_prompt: str = "You are a concise voice assistant. Reply in 1-2 short sentences.",
        temperature: float = 0.7,
        max_tokens: int = 150,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, user_prompt: str) -> str:
        """Generate response text from local Ollama instance with graceful fallback."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": False,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass
        # shortcut: fallback echo if Ollama server is not running locally yet
        return f"I received your request: '{user_prompt}'."
