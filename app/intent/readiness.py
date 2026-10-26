from dataclasses import dataclass
import re

QUESTION_STARTERS = {
    "what", "why", "how", "who", "where", "when", "which",
    "can", "could", "would", "should", "is", "are", "do", "does", "did", "will", "tell", "explain",
}

DANGLING_TAILS = {
    "and", "or", "but", "because", "with", "to", "that", "in", "of", "for", "the", "a", "an", "if",
}


@dataclass
class IntentReadinessResult:
    is_ready: bool
    score: float
    is_question: bool
    reason: str


class IntentReadinessDetector:
    """Analyzes incremental partial transcripts to determine if user intent is formed."""

    def __init__(self, min_words: int = 4, threshold: float = 0.65):
        self.min_words = min_words
        self.threshold = threshold
        self.last_transcript: str = ""
        self.stability_count: int = 0

    def reset(self):
        """Reset state between turns."""
        self.last_transcript = ""
        self.stability_count = 0

    def evaluate(self, partial_text: str) -> IntentReadinessResult:
        """Evaluate syntactic completeness, question classification, and stability."""
        clean = partial_text.strip().lower()
        words = re.findall(r"\b\w+\b", clean)

        if len(words) < self.min_words:
            self._update_stability(clean)
            return IntentReadinessResult(
                is_ready=False,
                score=0.0,
                is_question=False,
                reason="too_few_words",
            )

        # 1. Stability tracking
        if clean.startswith(self.last_transcript) and len(self.last_transcript) > 0:
            self.stability_count += 1
        else:
            self.stability_count = 1
        self.last_transcript = clean

        # 2. Question classification
        first_word = words[0]
        has_question_starter = first_word in QUESTION_STARTERS or (
            len(words) >= 2 and f"{words[0]} {words[1]}" in {"can you", "tell me", "what is", "how do"}
        )
        has_question_mark = "?" in partial_text
        is_question = has_question_starter or has_question_mark

        # 3. Syntactic completeness heuristics
        # shortcut: simple keyword heuristics, upgrade to dependency parser if complex syntax is needed
        is_dangling = words[-1] in DANGLING_TAILS

        # 4. Composite confidence score
        score = 0.0
        if is_question:
            score += 0.4
        if len(words) >= self.min_words:
            score += min(len(words) / 10.0, 0.3)
        if not is_dangling:
            score += 0.2
        if self.stability_count >= 2:
            score += 0.1

        score = min(score, 1.0)
        is_ready = (score >= self.threshold) and not is_dangling

        reason = "intent_ready" if is_ready else ("dangling_clause" if is_dangling else "low_confidence")
        return IntentReadinessResult(
            is_ready=is_ready,
            score=round(score, 2),
            is_question=is_question,
            reason=reason,
        )

    def _update_stability(self, clean: str):
        if clean == self.last_transcript:
            self.stability_count += 1
        else:
            self.stability_count = 1
        self.last_transcript = clean
