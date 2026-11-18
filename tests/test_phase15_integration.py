from unittest.mock import MagicMock
from starlette.testclient import TestClient

from app.api.server import create_app
from app.knowledge.retriever import LocalKnowledgeBase
from app.llm.main_model import LocalLLM


def test_knowledge_rest_api_crud_and_query():
    kb = LocalKnowledgeBase()
    llm = LocalLLM(knowledge_base=kb, rag_enabled=True)
    orch = MagicMock()
    orch.llm = llm
    orch.session_recorder = None
    orch.turn_manager = None

    app = create_app(orchestrator=orch)
    client = TestClient(app)

    # 1. Initially empty
    resp = client.get("/api/knowledge")
    assert resp.status_code == 200
    assert resp.json()["documents"] == []
    assert resp.json()["rag_enabled"] is True

    # 2. Validation rejects empty title or content
    bad = client.post("/api/knowledge/documents", json={"title": "", "content": "text"})
    assert bad.status_code == 400

    # 3. Add a document
    add = client.post(
        "/api/knowledge/documents",
        json={
            "doc_id": "runbook",
            "title": "Incident Runbook",
            "content": "Restart the voice gateway with uv run python app/main.py --dashboard.",
        },
    )
    assert add.status_code == 200
    assert add.json()["document"]["doc_id"] == "runbook"

    # 4. Query returns ranked snippet
    q = client.post("/api/knowledge/query", json={"query": "How do I restart the voice gateway?", "top_k": 2})
    assert q.status_code == 200
    results = q.json()["results"]
    assert len(results) == 1
    assert results[0]["doc_id"] == "runbook"
    assert results[0]["score"] > 0.5

    # Empty query returns 400
    assert client.post("/api/knowledge/query", json={"query": "   "}).status_code == 400

    # 5. Toggle RAG updates orchestrator.llm.rag_enabled
    tog = client.post("/api/knowledge/toggle", json={"enabled": False})
    assert tog.status_code == 200
    assert tog.json()["rag_enabled"] is False
    assert llm.rag_enabled is False

    # 6. Delete document and verify 404 on repeat
    deleted = client.delete("/api/knowledge/documents/runbook")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert client.delete("/api/knowledge/documents/runbook").status_code == 404


def test_dashboard_includes_knowledge_base_ui():
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "Local Knowledge Base (Voice RAG)" in html
    assert 'id="rag-toggle"' in html
    assert 'id="kb-add-btn"' in html
    assert 'id="kb-query-btn"' in html

