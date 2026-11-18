import pytest
from unittest.mock import patch

from app.knowledge.retriever import LocalKnowledgeBase
from app.llm.main_model import LocalLLM


def test_knowledge_base_add_search_and_remove():
    kb = LocalKnowledgeBase()
    doc1 = kb.add_document(
        title="Office WiFi Guide",
        content="The office WiFi network is Studio5G and the guest password is coffee-beans.",
        doc_id="wifi",
    )
    doc2 = kb.add_document(
        title="Release Schedule",
        content="Deployments happen every Tuesday at 10 AM UTC after pytest passes.",
        doc_id="release",
    )

    assert doc1["doc_id"] == "wifi"
    assert len(kb.list_documents()) == 2

    hits = kb.search("What is the office WiFi password?")
    assert len(hits) >= 1
    assert hits[0]["doc_id"] == "wifi"
    assert "coffee-beans" in hits[0]["content"]

    # Irrelevant query should not match
    assert kb.search("quantum entanglement black hole") == []

    # Removing doc removes it from future searches
    assert kb.remove_document("wifi") is True
    assert kb.remove_document("nonexistent") is False
    assert kb.search("office WiFi password") == []


def test_knowledge_base_single_doc_positive_bm25_and_validation():
    kb = LocalKnowledgeBase()
    with pytest.raises(ValueError):
        kb.add_document("", "some content")
    with pytest.raises(ValueError):
        kb.add_document("Title", "   ")

    # Single-document corpus must still produce positive BM25+ score
    kb.add_document("Ryzen Spec", "AMD Ryzen 5 7520U runs Whisper INT8 locally.", doc_id="spec")
    hits = kb.search("Ryzen Whisper")
    assert len(hits) == 1
    assert hits[0]["score"] > 0.5


def test_knowledge_base_directory_loading_and_chunking(tmp_path):
    (tmp_path / "hardware.md").write_text(
        "Section one about CPU threads.\n\n" + ("Detailed benchmark note. " * 30),
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Remember to calibrate microphone in noisy rooms.", encoding="utf-8")
    (tmp_path / "ignored.json").write_text('{"skip": true}', encoding="utf-8")

    kb = LocalKnowledgeBase(max_chunk_chars=200)
    count = kb.load_directory(str(tmp_path))
    assert count == 2

    docs = {d["doc_id"]: d for d in kb.list_documents()}
    assert "hardware" in docs
    assert docs["hardware"]["chunk_count"] >= 2
    assert "notes" in docs

    context = kb.build_context_block("calibrate microphone noisy", max_chars=120)
    assert "[notes]:" in context
    assert len(context) <= 125


def test_llm_injects_rag_context_into_prompt():
    kb = LocalKnowledgeBase()
    kb.add_document("Project Secret", "The voice agent wake code is Falcon-9.", doc_id="secret")

    captured_payloads = []

    class FakeResponse:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self):
            return [b'{"message":{"content":"The wake code is Falcon-9."},"done":true}\n']

    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def stream(self, method, url, json=None):
            captured_payloads.append(json)
            return FakeResponse()

    llm = LocalLLM(knowledge_base=kb, rag_enabled=True)
    with patch("httpx.Client", return_value=FakeClient()):
        out = llm.generate("What is the voice agent wake code?")
        assert "Falcon-9" in out
        assert "Falcon-9" in llm.last_rag_context
        sys_msg = captured_payloads[0]["messages"][0]["content"]
        assert "Relevant local knowledge:" in sys_msg
        assert "Falcon-9" in sys_msg

    # When RAG is disabled, context is not injected
    captured_payloads.clear()
    llm.rag_enabled = False
    with patch("httpx.Client", return_value=FakeClient()):
        llm.generate("What is the voice agent wake code?")
        assert llm.last_rag_context == ""
        sys_msg = captured_payloads[0]["messages"][0]["content"]
        assert "Relevant local knowledge:" not in sys_msg
