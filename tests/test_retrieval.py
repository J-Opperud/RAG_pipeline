import chromadb

DB_PATH = "chroma_db"
COLLECTION_NAME = "pep_documents"


def get_collection():
    """Open the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)

    return client.get_collection(
        name=COLLECTION_NAME
    )


def retrieve(collection, question: str, n_results: int = 3):
    """Retrieve the most relevant chunks for a question."""
    return collection.query(
        query_texts=[question],
        n_results=n_results,
    )


def main() -> None:
    """Test semantic retrieval with a sample question."""
    collection = get_collection()

    question = "What are the naming conventions for Python variables?"

    results = retrieve(collection, question)

    print(f"Question: {question}\n")

    for index, (document, metadata) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
        ),
        start=1,
    ):
        print(f"--- Retrieved Chunk {index} ---")
        print(f"Source: {metadata['source']}")
        print(document[:1000])
        print()


if __name__ == "__main__":
    main()