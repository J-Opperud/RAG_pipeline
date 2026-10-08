import chromadb


DB_PATH = "chroma_db"


def main() -> None:
    """Create a persistent ChromaDB collection and test it."""
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name="pep_documents"
    )

    collection.upsert(
        ids=["test-1"],
        documents=[
            "Python is a programming language known for readable syntax."
        ],
        metadatas=[
            {"source": "test"}
        ],
    )

    results = collection.query(
        query_texts=["What is Python?"],
        n_results=1,
    )

    print("Retrieved document:")
    print(results["documents"][0][0])

    print("\nMetadata:")
    print(results["metadatas"][0][0])


if __name__ == "__main__":
    main()

