from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import ollama

from my_rag import (
    MODEL_NAME,
    get_collection,
    ingest_documents,
    retrieve,
    build_rag_prompt,
    SYSTEM_PROMPT,
)

# FastAPI application entry point - Docker cache test

app = FastAPI(
    title="PEP RAG API",
    description="A FastAPI interface for the Python PEP RAG assistant.",
    version="1.0.0",
)

# Allow a future frontend to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value






class AskResponse(BaseModel):
    """Structured response returned by /ask."""

    answer: str
    sources: list[str]
    confidence: str
    chunks_retrieved: int


class IngestResponse(BaseModel):
    """Response returned after document ingestion."""

    message: str
    chunks_ingested: int


class StatsResponse(BaseModel):
    """Current RAG system statistics."""

    document_count: int
    model: str


class HealthResponse(BaseModel):
    """Health status of the API dependencies."""

    chromadb: str
    ollama: str
    status: str


def get_rag_collection():
    """Get the persistent ChromaDB collection."""
    return get_collection()


def calculate_confidence(retrieved_chunks: list[dict]) -> str:
    """Determine confidence from the best retrieved distance."""
    if not retrieved_chunks:
        return "low"

    best_distance = min(
        chunk["distance"]
        for chunk in retrieved_chunks
    )

    if best_distance < 0.5:
        return "high"

    if best_distance < 1.0:
        return "medium"

    return "low"


def generate_api_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Generate a complete answer for an API response."""
    prompt = build_rag_prompt(
        question,
        retrieved_chunks,
    )

    try:
        response_stream = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to Ollama. "
                "Make sure Ollama is running and the model is available."
            ),
        ) from exc

    answer_parts = []

    for response in response_stream:
        answer_parts.append(
            response["message"]["content"]
        )

    return "".join(answer_parts).strip()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check whether ChromaDB and Ollama are accessible."""
    chromadb_status = "ok"
    ollama_status = "ok"

    try:
        collection = get_rag_collection()
        collection.count()
    except Exception:
        chromadb_status = "unavailable"

    try:
        ollama.list()
    except Exception:
        ollama_status = "unavailable"

    overall_status = (
        "ok"
        if chromadb_status == "ok" and ollama_status == "ok"
        else "degraded"
    )

    return HealthResponse(
        chromadb=chromadb_status,
        ollama=ollama_status,
        status=overall_status,
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return basic information about the RAG collection."""
    collection = get_rag_collection()

    return StatsResponse(
        document_count=collection.count(),
        model=MODEL_NAME,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load the current documents into ChromaDB."""
    collection = get_rag_collection()

    chunk_count = ingest_documents(collection)

    return IngestResponse(
        message="Documents ingested successfully.",
        chunks_ingested=chunk_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question using the RAG pipeline."""
    collection = get_rag_collection()

    if collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "No documents have been ingested. "
                "Call POST /ingest before asking questions."
            ),
        )

    retrieved_chunks = retrieve(
        collection,
        request.question,
    )

    if not retrieved_chunks:
        return AskResponse(
            answer="I don't know based on the provided documents.",
            sources=[],
            confidence="low",
            chunks_retrieved=0,
        )

    answer = generate_api_answer(
        request.question,
        retrieved_chunks,
    )

    sources = sorted(
        {
            chunk["source"]
            for chunk in retrieved_chunks
        }
    )

    confidence = calculate_confidence(
        retrieved_chunks
    )

    return AskResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        chunks_retrieved=len(retrieved_chunks),
    )

