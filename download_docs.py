from pathlib import Path
from urllib.request import urlopen

DOCS_DIR = Path("docs")

PEPS = {
    "pep8.txt": "https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst",
    "pep20.txt": "https://raw.githubusercontent.com/python/peps/main/peps/pep-0020.rst",
    "pep257.txt": "https://raw.githubusercontent.com/python/peps/main/peps/pep-0257.rst",
    "pep484.txt": "https://raw.githubusercontent.com/python/peps/main/peps/pep-0484.rst",
    "pep526.txt": "https://raw.githubusercontent.com/python/peps/main/peps/pep-0526.rst",
    "pep585.txt": "https://raw.githubusercontent.com/python/peps/main/peps/pep-0585.rst",
}


def download_document(filename: str, url: str) -> None:
    """Download one PEP and save it as a text file."""
    destination = DOCS_DIR / filename

    print(f"Downloading {filename}...")

    with urlopen(url) as response:
        content = response.read().decode("utf-8")

    destination.write_text(content, encoding="utf-8")

    print(f"Saved {destination} ({len(content):,} characters)")


def main() -> None:
    """Download all documents used by the RAG pipeline."""
    DOCS_DIR.mkdir(exist_ok=True)

    for filename, url in PEPS.items():
        download_document(filename, url)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()