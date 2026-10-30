import pytest
from unittest.mock import MagicMock, patch

from app.conversation.memory import ConversationMemory
from app.llm.main_model import LocalLLM
from app.tts.sentence_splitter import StreamingSentenceChunker, ABBREVIATIONS


def test_chunker_basic_sentence_split():
    chunker = StreamingSentenceChunker(min_sentence_chars=10)
    tokens = ["The weather ", "is sunny today. ", "It is ", "very warm outside!"]
    sentences = []
    for t in tokens:
        sentences.extend(chunker.add_token(t))
    sentences.extend(chunker.flush())

    assert len(sentences) == 2
    assert sentences[0] == "The weather is sunny today."
    assert sentences[1] == "It is very warm outside!"


def test_chunker_preserves_abbreviations_and_numbers():
    chunker = StreamingSentenceChunker(min_sentence_chars=10)
    text = "Dr. Smith bought 3.14 liters of milk at approx. 5 p.m. He then left."
    # Feed token by token
    sentences = []
    for word in text.split(" "):
        sentences.extend(chunker.add_token(word + " "))
    sentences.extend(chunker.flush())

    # Should not split on Dr. or 3.14 or approx.
    assert len(sentences) == 2
    assert "Dr. Smith bought 3.14 liters" in sentences[0]
    assert sentences[1] == "He then left."


def test_chunker_newline_boundary():
    chunker = StreamingSentenceChunker(min_sentence_chars=5)
    tokens = ["First instruction completed.\n", "Second instruction begins."]
    sentences = []
    for t in tokens:
        sentences.extend(chunker.add_token(t))
    sentences.extend(chunker.flush())

    assert len(sentences) == 2
    assert sentences[0] == "First instruction completed."
    assert sentences[1] == "Second instruction begins."


def test_chunker_reset_and_empty_flush():
    chunker = StreamingSentenceChunker()
    chunker.add_token("Incomplete clause without punct")
    assert chunker.flush() == ["Incomplete clause without punct"]
    assert chunker.flush() == []

    chunker.add_token("Some text")
    chunker.reset()
    assert chunker.flush() == []


def test_llm_stream_generate_fallback_and_memory():
    mem = ConversationMemory(max_turns=3)
    llm = LocalLLM(base_url="http://invalid-host:9999", memory=mem)

    stream = list(llm.stream_generate("Tell me the time"))
    joined = "".join(stream).strip()

    assert "Tell me the time" in joined
    assert len(mem.turns) == 1
    assert mem.turns[0].user == "Tell me the time"
    assert mem.turns[0].agent == joined


def test_llm_stream_generate_success_mock():
    mock_lines = [
        b'{"message":{"content":"Hello "},"done":false}\n',
        b'{"message":{"content":"world!"},"done":true}\n',
    ]

    class FakeResponse:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self): return mock_lines

    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stream(self, method, url, json=None):
            return FakeResponse()

    with patch("httpx.Client", return_value=FakeClient()):
        llm = LocalLLM()
        tokens = list(llm.stream_generate("hi"))
        assert tokens == ["Hello ", "world!"]
        assert llm.generate("hi") == "Hello world!"

