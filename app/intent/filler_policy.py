from collections import deque
import random
import re


GREETING_PATTERNS = {"hi", "hello", "hey", "good morning", "good evening", "how are you", "what's up"}
SHORT_RESPONSES = {"yes", "no", "yeah", "nope", "okay", "ok", "thanks", "thank you", "bye", "goodbye"}

FALLBACK_FILLERS = [
    "Let me check that.",
    "Looking into that.",
    "Let's see.",
    "One moment.",
    "Thinking through that.",
    "Got it, checking now.",
]


class FillerPolicy:
    """Enforces safety guardrails, context eligibility, and anti-repetition for fillers."""

    def __init__(self, history_len: int = 3):
        self.recent_fillers: deque[str] = deque(maxlen=history_len)
        self.generated_count: int = 0
        self.played_count: int = 0
        self.discarded_count: int = 0

    def is_eligible(self, text: str) -> bool:
        """Evaluate whether user query justifies generating and playing a filler."""
        cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
        words = cleaned.split()

        # 1. Skip very short utterances
        if len(words) < 3:
            return False

        # 2. Skip greetings and polite closures
        if cleaned in GREETING_PATTERNS or words[0] in GREETING_PATTERNS:
            return False

        if cleaned in SHORT_RESPONSES:
            return False

        return True

    def select_filler(self, user_text: str, slm_filler: str | None = None) -> str:
        """Select safe, non-repetitive filler from SLM candidate or fallback pool."""
        self.generated_count += 1

        # Check candidate from SLM
        if slm_filler:
            clean_candidate = slm_filler.strip()
            if clean_candidate not in self.recent_fillers:
                return clean_candidate

        # Select from pool avoiding recently used phrases
        available = [f for f in FALLBACK_FILLERS if f not in self.recent_fillers]
        if not available:
            available = FALLBACK_FILLERS
        choice = random.choice(available)
        return choice

    def mark_played(self, filler: str):
        """Record filler as audibly played and register in anti-repetition history."""
        self.played_count += 1
        self.recent_fillers.append(filler)

    def mark_discarded(self):
        """Record that a generated filler was discarded before playback."""
        self.discarded_count += 1

    def stats(self) -> dict:
        """Return filler usage statistics."""
        return {
            "generated": self.generated_count,
            "played": self.played_count,
            "discarded": self.discarded_count,
            "discard_rate": round(self.discarded_count / max(self.generated_count, 1), 2),
        }
