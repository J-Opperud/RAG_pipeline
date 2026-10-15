import os

import requests

import streamlit as st

import json

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 180

st.set_page_config(
    page_title="PEP Assistant",
    page_icon="ðŸ“š",
    layout="wide",
)


def api_get(endpoint):
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def api_post(endpoint, payload=None, timeout=30):
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


st.title("ðŸ“š PEP RAG Assistant")
st.caption("Ask questions about Python Enhancement Proposals using retrieved documents.")

# Sidebar: system health and collection statistics.
with st.sidebar:
    st.header("System status")

    if st.button("Refresh status", width="stretch"):
        st.rerun()

    try:
        health = api_get("/health")

        if health["status"] == "ok":
            st.success("API: healthy")
        else:
            st.warning(f"API: {health['status']}")

        st.write(f"**ChromaDB:** {health['chromadb']}")
        st.write(f"**Ollama:** {health['ollama']}")

    except requests.RequestException as exc:
        st.error("Backend unavailable")
        st.caption(str(exc))

    st.divider()
    st.header("Collection")

    try:
        stats = api_get("/stats")
        st.metric("Documents / chunks", stats["document_count"])
        st.caption(f"Configured model: {stats['model']}")

    except requests.RequestException:
        st.info("Collection statistics are unavailable.")

    st.divider()

    if st.button("Re-index documents", width="stretch"):
        with st.spinner("Ingesting documents..."):
            try:
                result = api_post("/ingest", timeout=300)
                st.success(result["message"])
                st.write(
                    f"Chunks ingested: {result['chunks_ingested']}"
                )
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")


# Keep conversation history during Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process a submitted question before rendering the conversation.

question = st.chat_input("Ask a question about Python PEPs...")

# Display previous conversation messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            st.caption(
                f"Confidence: {message['confidence']} Â· "
                f"Chunks retrieved: {message['chunks_retrieved']}"
            )

            if message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer = ""
        metadata = {}
        error_message = None

        try:
            with requests.post(
                f"{API_URL}/ask/stream",
                json={"question": question},
                stream=True,
                timeout=(10, REQUEST_TIMEOUT),
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue

                    event = json.loads(line[6:])

                    if event["type"] == "chunk":
                        answer += event["content"]
                        answer_placeholder.markdown(answer + " â–Œ")

                    elif event["type"] == "done":
                        metadata = event

                    elif event["type"] == "error":
                        error_message = event["message"]
                        break


        except (requests.RequestException, ValueError) as exc:
            error_message = f"Unable to get an answer from the API: {exc}"

        if error_message:
            answer_placeholder.error(error_message)
        else:
            answer_placeholder.markdown(answer)

            st.caption(
                f"Confidence: {metadata.get('confidence', 'unavailable')} Â· "
                f"Chunks retrieved: {metadata.get('chunks_retrieved', 0)}"
            )

            sources = metadata.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"- {source}")



    if error_message:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "sources": [],
                "confidence": "unavailable",
                "chunks_retrieved": 0,
            }
        )
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": metadata.get("sources", []),
                "confidence": metadata.get("confidence", "unavailable"),
                "chunks_retrieved": metadata.get("chunks_retrieved", 0),
            }
        )
