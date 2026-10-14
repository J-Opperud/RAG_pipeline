import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 180

st.set_page_config(
    page_title="PEP RAG Assistant",
    page_icon="📚",
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


st.title("📚 PEP RAG Assistant")
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

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.spinner("Searching documents and generating an answer..."):
        try:
            result = api_post(
                "/ask",
                {"question": question},
                timeout=REQUEST_TIMEOUT,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                    "chunks_retrieved": result["chunks_retrieved"],
                }
            )

        except requests.RequestException as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Unable to get an answer from the API: {exc}",
                    "sources": [],
                    "confidence": "unavailable",
                    "chunks_retrieved": 0,
                }
            )

    st.rerun()

# Render conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            st.caption(
                f"Confidence: {message['confidence']} · "
                f"Chunks retrieved: {message['chunks_retrieved']}"
            )

            if message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")
