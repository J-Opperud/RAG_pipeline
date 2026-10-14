from pathlib import Path


DOCS_DIR = Path("docs")


def load_document(path: Path) -> str:
    """Read a text document and return its contents."""
    return path.read_text(encoding="utf-8")


def chunk_by_paragraph(text: str) -> list[str]:
    """Split document text into non-empty paragraphs."""
    paragraphs = text.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def main() -> None:
    """Load one document and display its paragraph chunks."""
    document_path = DOCS_DIR / "pep8.txt"

    text = load_document(document_path)
    chunks = chunk_by_paragraph(text)

    print(f"Document: {document_path}")
    print(f"Total chunks: {len(chunks)}")

    print("\nFirst 3 chunks:\n")

    for number, chunk in enumerate(chunks[:3], start=1):
        print(f"--- Chunk {number} ---")
        print(chunk[:500])
        print()


if __name__ == "__main__":
    main()