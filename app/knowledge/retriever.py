import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "tell", "that", "the", "this", "to", "what", "when",
    "where", "which", "who", "with", "you", "your",
}


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric terms excluding conversational stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _chunk_text(text: str, max_chunk_chars: int = 450) -> List[str]:
    """Split text into paragraph-aware chunks bounded by max_chunk_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            # Split long paragraph by sentence or fixed window
            words = para.split()
            buf = ""
            for word in words:
                if len(buf) + len(word) + 1 > max_chunk_chars and buf:
                    chunks.append(buf)
                    buf = word
                else:
                    buf = f"{buf} {word}".strip()
            if buf:
                chunks.append(buf)
        elif len(current) + len(para) + 2 <= max_chunk_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


@dataclass
class DocumentChunk:
    doc_id: str
    title: str
    content: str
    chunk_index: int
    term_freqs: Counter = field(default_factory=Counter)
    length: int = 0


class LocalKnowledgeBase:
    """Zero-dependency local BM25 retriever for grounding voice LLM responses in local documents."""

    def __init__(self, max_chunk_chars: int = 450, k1: float = 1.5, b: float = 0.75):
        self.max_chunk_chars = max_chunk_chars
        self.k1 = k1
        self.b = b
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._chunks: List[DocumentChunk] = []
        self._doc_freq: Counter = Counter()
        self._avg_chunk_len: float = 1.0

    def _rebuild_stats(self):
        self._doc_freq.clear()
        total_len = 0
        for chunk in self._chunks:
            total_len += chunk.length
            for term in chunk.term_freqs:
                self._doc_freq[term] += 1
        self._avg_chunk_len = (total_len / len(self._chunks)) if self._chunks else 1.0

    def add_document(self, title: str, content: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Index a document into searchable chunks. Replaces existing document if doc_id matches."""
        clean_title = title.strip()
        clean_content = content.strip()
        if not clean_title or not clean_content:
            raise ValueError("Document title and content must be non-empty.")

        did = doc_id.strip() if doc_id and doc_id.strip() else f"doc_{uuid.uuid4().hex[:8]}"
        if did in self._documents:
            self._chunks = [c for c in self._chunks if c.doc_id != did]

        raw_chunks = _chunk_text(clean_content, max_chunk_chars=self.max_chunk_chars)
        for idx, text in enumerate(raw_chunks):
            # Include title terms so queries referencing document title match strongly
            tokens = _tokenize(f"{clean_title} {text}")
            self._chunks.append(
                DocumentChunk(
                    doc_id=did,
                    title=clean_title,
                    content=text,
                    chunk_index=idx,
                    term_freqs=Counter(tokens),
                    length=len(tokens),
                )
            )

        meta = {
            "doc_id": did,
            "title": clean_title,
            "chunk_count": len(raw_chunks),
            "char_count": len(clean_content),
        }
        self._documents[did] = meta
        self._rebuild_stats()
        return meta

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and all of its chunks from the index."""
        if doc_id not in self._documents:
            return False
        del self._documents[doc_id]
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        self._rebuild_stats()
        return True

    def load_directory(self, directory: str) -> int:
        """Load and index all .md and .txt files in a local directory."""
        base = Path(directory)
        if not base.exists() or not base.is_dir():
            return 0
        loaded = 0
        for path in sorted(base.iterdir()):
            if path.is_file() and path.suffix.lower() in (".md", ".txt"):
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        self.add_document(title=path.stem, content=content, doc_id=path.stem)
                        loaded += 1
                except Exception:
                    continue
        return loaded

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return summaries of all indexed documents."""
        return list(self._documents.values())

    def search(self, query: str, top_k: int = 3, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """Rank chunks using BM25+ scoring and return top_k hits above min_score."""
        if not self._chunks or not query.strip():
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        n_chunks = len(self._chunks)
        scored = []
        for chunk in self._chunks:
            score = 0.0
            for term in set(query_terms):
                tf = chunk.term_freqs.get(term, 0)
                if tf == 0:
                    continue
                df = self._doc_freq.get(term, 0)
                # +1 keeps IDF strictly positive even for 1-document corpora
                idf = math.log(((n_chunks - df + 0.5) / (df + 0.5)) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (chunk.length / self._avg_chunk_len))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)

            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "score": round(score, 3),
            }
            for score, chunk in scored[:top_k]
        ]

    def build_context_block(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.5,
        max_chars: int = 600,
    ) -> str:
        """Build a concise grounded context snippet for injection into the LLM system prompt."""
        hits = self.search(query, top_k=top_k, min_score=min_score)
        if not hits:
            return ""
        parts = []
        used = 0
        for hit in hits:
            snippet = f"[{hit['title']}]: {hit['content']}"
            if used + len(snippet) > max_chars:
                remaining = max_chars - used
                if remaining > 40:
                    parts.append(snippet[:remaining].rstrip() + "...")
                break
            parts.append(snippet)
            used += len(snippet) + 1
        return "\n".join(parts)
