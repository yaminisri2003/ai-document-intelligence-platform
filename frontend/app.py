# app.py
# The Streamlit frontend for AI Document Intelligence Platform.
#
# Run with: streamlit run frontend/app.py
#
# This file creates a complete chat interface that:
# - Lets users select their domain
# - Upload documents for ingestion
# - Ask questions and see AI answers
# - View source citations
# - Maintains conversation history visually

import streamlit as st
import requests
import uuid
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# URL of our FastAPI backend
# When running locally both run on your machine
API_URL = "http://localhost:8000"

# Available domains with friendly display names
DOMAINS = {
    "hr_enterprise": "HR / Enterprise",
    "healthcare":    "Healthcare",
    "legal":         "Legal",
    "finance":       "Finance",
}

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────
# These values persist across Streamlit reruns.
# Without session_state they would reset on every interaction.

if "messages" not in st.session_state:
    # Stores the full conversation history for display
    # Format: [{"role": "user/assistant", "content": "..."}]
    st.session_state.messages = []

if "session_id" not in st.session_state:
    # Unique ID for this browser session
    # Sent to FastAPI so it knows which conversation to continue
    st.session_state.session_id = str(uuid.uuid4())

if "domain" not in st.session_state:
    # Currently selected domain
    st.session_state.domain = "hr_enterprise"

if "last_sources" not in st.session_state:
    # Sources from the most recent answer
    st.session_state.last_sources = []

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def check_api_health() -> bool:
    """
    Check if the FastAPI backend is running.
    Returns True if healthy, False if not reachable.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def send_message(message: str, domain: str, session_id: str) -> dict:
    """
    Send a chat message to the FastAPI backend.

    Args:
        message:    User's question
        domain:     Active domain
        session_id: Session identifier

    Returns:
        API response dict with answer and sources
    """
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message":    message,
                "domain":     domain,
                "session_id": session_id,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "answer": "Cannot connect to the API. "
                     "Please start the FastAPI server with: "
                     "uvicorn app.api.main:app --reload",
            "sources": [],
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
        }


def ingest_document(file, domain: str) -> dict:
    """
    Upload a document to the FastAPI ingest endpoint.

    Args:
        file:   Streamlit uploaded file object
        domain: Domain to ingest into

    Returns:
        API response dict
    """
    try:
        response = requests.post(
            f"{API_URL}/ingest",
            files={"file": (file.name, file.getvalue(), file.type)},
            data={"domain": domain},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        return {"message": f"Error: {str(e)}", "chunks_created": 0}


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    # API Health Status
    st.subheader("API Status")
    if check_api_health():
        st.success("✅ API is running")
    else:
        st.error("❌ API is not running")
        st.code("uvicorn app.api.main:app --reload")

    st.divider()

    # Domain Selector
    st.subheader("Select Domain")
    selected_domain = st.selectbox(
        "Domain",
        options=list(DOMAINS.keys()),
        format_func=lambda x: DOMAINS[x],
        index=list(DOMAINS.keys()).index(st.session_state.domain),
        label_visibility="collapsed",
    )

    # Update domain if changed
    if selected_domain != st.session_state.domain:
        st.session_state.domain = selected_domain
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.info("Domain changed. Conversation cleared.")

    st.divider()

    # Document Upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx", "md"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("Ingest Document", type="primary"):
            with st.spinner("Processing document..."):
                result = ingest_document(
                    uploaded_file,
                    st.session_state.domain
                )

            if result.get("chunks_created", 0) > 0:
                st.success(
                    f"✅ Ingested {result['chunks_created']} "
                    f"chunks from {result['filename']}"
                )
            else:
                st.error(result.get("message", "Ingestion failed"))

    st.divider()

    # Clear Conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # Session Info
    st.divider()
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    st.caption(f"Messages: {len(st.session_state.messages)}")

# ─────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────

# Header
st.title("🤖 AI Document Intelligence Platform")
st.caption(
    f"Active domain: **{DOMAINS[st.session_state.domain]}** "
    f"| Ask questions about your documents"
)

st.divider()

# Display conversation history
# Loop through all stored messages and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Source citations for last answer
if st.session_state.last_sources:
    with st.expander("📄 Sources used in last answer"):
        for source in st.session_state.last_sources:
            st.caption(
                f"📎 {source.get('source', 'Unknown')} "
                f"| Chunk {source.get('chunk_index', '?')} "
                f"| Distance: {source.get('distance', '?')}"
            )

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────

# st.chat_input creates the message box at the bottom
# It returns the user's message when they press Enter
if prompt := st.chat_input("Ask a question about your documents..."):

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to conversation history
    st.session_state.messages.append({
        "role":    "user",
        "content": prompt,
    })

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = send_message(
                message=prompt,
                domain=st.session_state.domain,
                session_id=st.session_state.session_id,
            )

        answer = result.get("answer", "No response received")
        st.markdown(answer)

    # Add AI response to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
    })

    # Store sources for display
    st.session_state.last_sources = result.get("sources", [])

    # Rerun to update the sources expander
    st.rerun()