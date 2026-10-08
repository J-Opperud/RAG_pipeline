from pathlib import Path

import chromadb


DOCS_DIR = Path("docs")
DB_PATH = "chroma_db"
COLLECTION_NAME = "pep_documents"


def load_document(path: Path) -> str:
    """Read a document from disk."""
    return path.read_text(encoding="utf-8")


def chunk_by_paragraph(text: str) -> list[str]:
    """Split document text into non-empty paragraphs."""
    paragraphs = text.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def create_collection():
    """Create or open the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def ingest_document(collection, path: Path) -> int:
    """Chunk one document and store its chunks in ChromaDB."""
    text = load_document(path)
    chunks = chunk_by_paragraph(text)

    ids = [
        f"{path.stem}-{index}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {"source": path.name}
        for _ in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def main() -> None:
    """Ingest all text documents into ChromaDB."""
    collection = create_collection()

    total_chunks = 0

    for path in sorted(DOCS_DIR.glob("*.txt")):
        chunk_count = ingest_document(collection, path)

        print(f"{path.name}: {chunk_count} chunks")

        total_chunks += chunk_count

    print(f"\nTotal chunks stored: {total_chunks}")


if __name__ == "__main__":
    main()