import ollama


MODEL = "llama3.2:latest"


def main() -> None:
    """Send a simple test prompt to Ollama."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "In one sentence, explain what Python is.",
            }
        ],
    )

    print(response["message"]["content"])


if __name__ == "__main__":
    main()