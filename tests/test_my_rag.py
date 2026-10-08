from pathlib import Path

import pytest

from my_rag import (
    COLLECTION_NAME,
    DB_PATH,
    DOCS_DIR,
    TOP_K,
    chunk_by_paragraph,
    get_collection,
    get_document_paths,
    ingest_document,
    ingest_documents,
    load_document,
    retrieve,
    build_rag_prompt,
    generate_answer,
)


def test_docs_directory_exists():
    """The project should contain a docs directory."""
    assert DOCS_DIR.exists()
    assert DOCS_DIR.is_dir()


def test_six_documents_exist():
    """The document collection should contain six PEP files."""
    documents = get_document_paths()

    assert len(documents) == 6

    for document in documents:
        assert document.suffix == ".txt"
        assert document.is_file()


def test_load_document_returns_text():
    """Documents should load as non-empty strings."""
    documents = get_document_paths()

    text = load_document(documents[0])

    assert isinstance(text, str)
    assert text.strip()


def test_chunk_by_paragraph():
    """Paragraphs should become separate non-empty chunks."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_by_paragraph(text)

    assert chunks == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_chunk_removes_empty_paragraphs():
    """Empty paragraphs should not become chunks."""
    text = "\n\nFirst paragraph.\n\n\n\nSecond paragraph.\n\n"

    chunks = chunk_by_paragraph(text)

    assert chunks == [
        "First paragraph.",
        "Second paragraph.",
    ]


def test_empty_document_returns_no_chunks():
    """An empty document should produce no chunks."""
    chunks = chunk_by_paragraph("   \n\n   ")

    assert chunks == []







def test_retrieve_returns_top_three_chunks():
    """Retrieval should return the requested number of chunks."""
    collection = get_collection()

    ingest_documents(collection)

    results = retrieve(
        collection,
        "What are Python variable naming conventions?",
    )

    assert len(results) == TOP_K

    for result in results:
        assert "text" in result
        assert "source" in result
        assert result["text"]
        assert result["source"]


def test_retrieve_rejects_empty_question():
    """An empty question should be rejected."""
    collection = get_collection()

    with pytest.raises(ValueError, match="Question cannot be empty."):
        retrieve(collection, "   ")


def test_retrieve_rejects_invalid_top_k():
    """top_k must be greater than zero."""
    collection = get_collection()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero.",
    ):
        retrieve(collection, "What is Python?", top_k=0)

def test_build_rag_prompt_contains_context_and_question():
    """The RAG prompt should contain sources, context, and the question."""
    chunks = [
        {
            "source": "pep8.txt",
            "text": "Variable names follow the same convention as function names.",
        },
        {
            "source": "pep20.txt",
            "text": "Beautiful is better than ugly.",
        },
    ]

    prompt = build_rag_prompt(
        "What are Python variable naming conventions?",
        chunks,
    )

    assert "pep8.txt" in prompt
    assert "Variable names follow" in prompt
    assert "pep20.txt" in prompt
    assert "What are Python variable naming conventions?" in prompt



def test_build_rag_prompt_rejects_empty_question():
    """An empty question should be rejected."""
    chunks = [
        {
            "source": "pep8.txt",
            "text": "Variable names follow the same convention as function names.",
        }
    ]

    with pytest.raises(ValueError, match="Question cannot be empty."):
        build_rag_prompt("   ", chunks)


def test_build_rag_prompt_rejects_missing_context():
    """A RAG prompt requires retrieved context."""
    with pytest.raises(
        ValueError,
        match="No retrieved context was provided.",
    ):
        build_rag_prompt("What is Python?", [])

def test_generate_answer():
    """Ollama should generate an answer from retrieved context."""
    chunks = [
        {
            "source": "pep8.txt",
            "text": "Variable names follow the same convention as function names.",
        }
    ]

    answer = generate_answer(
        "What convention do Python variable names follow?",
        chunks,
    )

    assert isinstance(answer, str)
    assert answer.strip()

