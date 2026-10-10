RAG Pipeline

## Overview

A local Retrieval-Augmented Generation (RAG) pipeline built with Python, ChromaDB, and Ollama.

The project uses six Python Enhancement Proposals (PEPs) as its document collection. Documents are loaded, split into paragraph-based chunks, stored in persistent ChromaDB, retrieved by semantic similarity, and passed to a local Ollama model for grounded answers.



The application displays the retrieved chunks before the generated answer so the source material can be inspected.
Features

    Six local PEP documents
    Paragraph-based document chunking
    Persistent ChromaDB storage
    Top-3 semantic retrieval
    Source-aware RAG prompts
    Grounded-answer instructions
    Streaming Ollama responses
    Graceful Ollama connection errors
    Input validation
    Interactive CLI
    quit command
    Automated pytest coverage

## Arch.
Documents
    ↓
Load and chunk
    ↓
Persistent ChromaDB
    ↓
Retrieve top 3 chunks
    ↓
Build RAG prompt
    ↓
Ollama
    ↓
Stream answer

## Design

The pipeline is divided into focused functions:

load_document()
chunk_by_paragraph()
get_collection()
ingest_document()
ingest_documents()
retrieve()
build_rag_prompt()
generate_answer()
display_retrieved_chunks()
run_interactive()

This keeps responsibilities separate and makes individual components easier to test and debug.

The project intentionally avoids unnecessary complexity while still addressing common validation and failure cases.

## Project Structure

rag-pipeline/
├── docs/
│   ├── pep20.txt
│   ├── pep257.txt
│   ├── pep484.txt
│   ├── pep526.txt
│   ├── pep585.txt
│   └── pep8.txt
│
├── chroma_db/
├── tests/
│    ├── test_chroma.py
|    ├── test_chunks.py
|    ├── test_ollama.py
|    ├──test_retreval.py
|    └── test_my_rag.py
|
├── download_docs.py
├── ingest.py
├── my_rag.py
├── test_chroma.py
├── test_chunks.py
├── test_ollama.py
├── test_retrieval.py
└── README.md

## Requirements

    Python 3
    ChromaDB
    Ollama
    pytest
    llama3.2:latest

Install the Python dependencies:

pip install -r requirements.txt

Verify Ollama and the model:

ollama list

The project currently uses:

llama3.2:latest

## Running the Application

From the project root:

python my_rag.py

The application starts an interactive question loop:

RAG assistant ready.
Ask a question about PEP coding style and terminology guidelines.
Type 'quit' to exit.

For each question, the application:

    Retrieves the three most relevant chunks.
    Displays the retrieved sources.
    Builds a prompt using the retrieved context.
    Sends the prompt to Ollama.
    Streams the answer to the terminal.

## Testing

Run the complete test suite:

python -m pytest

The test suite covers:

    Document loading
    Paragraph chunking
    Empty document handling
    ChromaDB ingestion
    Duplicate prevention
    Retrieval
    Input validation
    RAG prompt construction
    Ollama generation

Current result:

13 passed

RAG Test Results

The application was tested with three required query types.
1. Answerable

Question:

What is a docstring?

The system retrieved relevant PEP 257 content and generated a grounded answer.

Result: PASS
2. Related but not directly documented

Question:

How should I structure a Python virtual environment?

The system retrieved Python-related content but recognized that the documents did not provide enough information to answer the question.

Result: PASS
3. Outside the document scope

Question:

What is the capital of France?

The system retrieved unrelated chunks but correctly recognized that the context did not contain information about geography.

Result: PASS

These tests demonstrate an important RAG behavior: retrieval may return the closest available documents even when they are not actually relevant. The generation step must still determine whether the retrieved context supports an answer.
Streaming

Ollama responses are streamed to the terminal as they are generated rather than waiting for the complete response.

This provides immediate visual feedback while the model is generating its answer.
Error Handling

The project uses lightweight validation appropriate for the assignment:

    Empty questions are rejected.
    Invalid retrieval parameters are rejected.
    Empty documents do not produce chunks.
    Missing RAG context is rejected.
    Ollama connection failures produce a readable error.
    The interactive loop continues after invalid input.
    quit exits the application cleanly.

FastAPI RAG API

The RAG pipeline is also available as a FastAPI service with Swagger UI for testing.
Endpoints

    POST /ask — Ask a question and receive a grounded answer with sources, confidence, and retrieved chunk count.
    POST /ingest — Ingest documents from the docs/ directory into ChromaDB.
    GET /stats — Return the current document count and Ollama model.
    GET /health — Check ChromaDB and Ollama availability.

## Run the API

uvicorn my_rag_api:app --reload

Open Swagger UI at:

http://127.0.0.1:8000/docs

## API Features

    Pydantic request and response validation
    CORS middleware for frontend connectivity
    Distance-based retrieval and confidence levels
    Structured JSON responses
    422 validation errors for invalid questions
    503 responses when Ollama is unavailable or documents have not been ingested
    Automated API tests with Pytest

### Docker

The image uses Python 3.11 slim and installs dependencies from requirements.txt. Docker layer caching allows dependency installation to be reused when application code changes.

## Run the API

Start the container and connect it to Ollama running on the host:

docker run --rm \
  --name my-rag-api-container \
  -p 8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  my-rag-api

Ensure Ollama is running and the llama3.2:latest model is available.

## Access the API

Open Swagger UI in your browser:

http://localhost:8000/docs


## Notes

    .dockerignore excludes Python caches, virtual environments, Git files, environment files, and local ChromaDB data from the build context.
    ChromaDB data persistence depends on the application's configuration and container storage setup.
    Docker layer caching keeps dependency installation fast when requirements.txt remains unchanged.

