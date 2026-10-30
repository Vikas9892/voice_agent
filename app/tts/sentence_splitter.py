import re
from typing import List

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.",
    "e.g.", "i.e.", "etc.", "approx.", "est.", "inc.", "ltd.",
    "co.", "dept.", "fig.", "no.", "st.", "ave.", "rd.",
}


class StreamingSentenceChunker:
    """Buffers streaming token fragments and yields complete, speakable sentence chunks."""

    def __init__(self, min_sentence_chars: int = 15):
        self.min_sentence_chars = min_sentence_chars
        self._buffer: str = ""

    def reset(self):
        self._buffer = ""

    def add_token(self, token: str) -> List[str]:
        """Add a token to the buffer and return any completed sentences."""
        self._buffer += token
        ready_sentences: List[str] = []

        while True:
            split_pos = self._find_sentence_boundary(self._buffer)
            if split_pos is None:
                break

            sentence = self._buffer[:split_pos].strip()
            self._buffer = self._buffer[split_pos:].lstrip()

            if sentence:
                ready_sentences.append(sentence)

        return ready_sentences

    def flush(self) -> List[str]:
        """Flush and return any remaining text in the buffer."""
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []

    def _find_sentence_boundary(self, text: str) -> int | None:
        """Find the earliest valid sentence boundary index in text, or None."""
        # Punctuation must be followed by whitespace or newline
        matches = list(re.finditer(r"([.!?]+|;)\s+|(\n+)", text))
        for match in matches:
            punct_end = match.end()
            # If newline group matched
            if match.group(2) is not None:
                cand_end = match.start(2)
                candidate = text[:cand_end].strip()
                if candidate:
                    return punct_end
                continue

            cand_end = match.end(1)
            candidate = text[:cand_end].strip()

            if len(candidate) < self.min_sentence_chars:
                continue

            words = candidate.split()
            last_word = words[-1].lower() if words else ""

            # Check if boundary is a known abbreviation (e.g., 'Dr.', 'e.g.')
            if last_word in ABBREVIATIONS:
                continue

            # Check if boundary is a decimal number (e.g., '3.14')
            if re.search(r"\d+\.\s*$", candidate):
                continue

            # Check for unfinished ellipsis (e.g., '..' waiting for third '.')
            if candidate.endswith("..") and not candidate.endswith("..."):
                continue

            return punct_end

        return None
