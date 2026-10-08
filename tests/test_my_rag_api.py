from unittest.mock import patch

from fastapi.testclient import TestClient

from my_rag_api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "chromadb" in data
    assert "ollama" in data
    assert "status" in data


def test_stats():
    response = client.get("/stats")

    assert response.status_code == 200

    data = response.json()

    assert "document_count" in data
    assert "model" in data


def test_empty_question_returns_422():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_whitespace_question_returns_422():
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ingest():
    response = client.post("/ingest")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Documents ingested successfully."
    assert data["chunks_ingested"] > 0


def test_ask_returns_structured_response():
    response = client.post(
        "/ask",
        json={"question": "What is a docstring?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "chunks_retrieved" in data

    assert isinstance(data["sources"], list)
    assert data["chunks_retrieved"] > 0


def test_ollama_unavailable_returns_503():
    with patch(
        "my_rag_api.ollama.chat",
        side_effect=Exception("Ollama unavailable"),
    ):
        response = client.post(
            "/ask",
            json={"question": "What is a docstring?"},
        )

    assert response.status_code == 503

    def test_ask_with_no_documents_returns_503():
        mock_collection = type(
            "MockCollection",
            (),
            {
                "count": lambda self: 0,
            },
        )()

        with patch(
            "my_rag_api.get_rag_collection",
            return_value=mock_collection,
        ):
            response = client.post(
                "/ask",
                json={"question": "What is a docstring?"},
            )

        assert response.status_code == 503
        assert "No documents have been ingested" in response.json()["detail"]