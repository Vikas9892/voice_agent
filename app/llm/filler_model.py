import httpx


class FillerModel:
    """Ultra-lightweight SLM client for generating micro-filler acknowledgments."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
        temperature: float = 0.3,
        max_tokens: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = (
            "You are a conversational voice assistant. Output ONLY a brief 2-4 word "
            "neutral acknowledgment (such as 'Let me check that', 'Looking into that', "
            "or 'Let's see'). Never answer the question or make claims."
        )

    def generate_filler(self, partial_text: str) -> str | None:
        """Query SLM for contextual filler with fast timeout and fallback."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"User query: {partial_text}"},
            ],
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": False,
        }
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    text = res.json().get("message", {}).get("content", "").strip()
                    # Sanitize quotes and punctuation
                    cleaned = text.strip('"\'').strip()
                    if cleaned and len(cleaned.split()) <= 6:
                        return cleaned
        except Exception:
            pass
        return None
