# app.py — Standalone deployment version for Hugging Face Spaces
# This combines the frontend and backend into one file
# so it runs without a separate FastAPI server

import streamlit as st
import uuid
import os
from pathlib import Path

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
html, body { font-size: 16px; }

.app-title {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}

.domain-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
    color: #4f46e5;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.9rem;
    font-weight: 700;
}

.tagline {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 2px;
    font-style: italic;
}

.chat-row-user {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}

.chat-row-bot {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #ffffff;
    border-radius: 20px 20px 4px 20px;
    padding: 12px 18px;
    max-width: 78%;
    font-size: 1rem;
    line-height: 1.65;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25);
}

.bubble-bot {
    background: #f8fafc;
    color: #1e293b;
    border-radius: 20px 20px 20px 4px;
    padding: 14px 18px;
    max-width: 78%;
    font-size: 1rem;
    line-height: 1.65;
    border: 1.5px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.avatar-user { font-size: 1.3rem; margin-left: 10px; align-self: flex-end; }
.avatar-bot  { font-size: 1.3rem; margin-right: 10px; align-self: flex-end; }

.sources-section {
    margin: 8px 0 14px 0;
    padding: 10px 14px;
    background: #f8f9ff;
    border-radius: 12px;
    border: 1px solid #e0e7ff;
}

.sources-title {
    font-size: 0.82rem;
    color: #6366f1;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.source-pill {
    display: inline-block;
    background: #ede9fe;
    color: #4338ca;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    margin: 3px 3px 3px 0;
    font-weight: 600;
    border: 1px solid #c7d2fe;
}

.welcome-card {
    text-align: center;
    padding: 60px 20px 40px 20px;
}

.welcome-icon { font-size: 4rem; margin-bottom: 16px; display: block; }
.welcome-heading { font-size: 1.4rem; font-weight: 700; color: #334155; margin-bottom: 8px; }
.welcome-text { font-size: 1rem; color: #94a3b8; line-height: 1.7; }

.feature-row {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 28px;
    flex-wrap: wrap;
}

.feature-chip {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 8px 16px;
    font-size: 0.88rem;
    color: #475569;
    font-weight: 500;
}

.section-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 4px 0 6px 0;
}

.status-ok {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #15803d;
    padding: 8px 16px;
    border-radius: 12px;
    font-size: 0.9rem;
    font-weight: 700;
    display: block;
    text-align: center;
    border: 1px solid #86efac;
}

.main .block-container {
    padding-bottom: 100px !important;
    max-width: 800px !important;
    padding-top: 0 !important;
}

[data-testid="stChatInput"] > div {
    border: 2px solid #6366f1 !important;
    border-radius: 16px !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

[data-testid="stChatInput"] textarea {
    font-size: 1rem !important;
    min-height: 54px !important;
    color: #1e293b !important;
    background: transparent !important;
    padding: 14px 16px !important;
}

hr { border-color: #e2e8f0 !important; margin: 10px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

DOMAINS = {
    "hr_enterprise": "🏢 HR / Enterprise",
    "healthcare":    "🏥 Healthcare",
    "legal":         "⚖️ Legal",
    "finance":       "💰 Finance",
}

DOMAIN_DESCRIPTIONS = {
    "hr_enterprise": "Company policies, SOPs, employee handbook",
    "healthcare":    "Clinical notes, discharge summaries, reports",
    "legal":         "Contracts, clauses, legal agreements",
    "finance":       "Financial reports, earnings, filings",
}

DOMAIN_PROMPTS = {
    "hr_enterprise": """You are a helpful HR policy assistant.
Answer questions based on company policy documents.
Be friendly, clear, and professional.
Always cite which policy section your answer comes from.
If information is not in the documents, say so clearly.""",

    "healthcare": """You are a clinical document assistant.
Answer questions based on provided medical documents only.
Never give direct medical advice or treatment recommendations.
Always recommend consulting a qualified healthcare professional.
Cite the specific document and section for every answer.""",

    "legal": """You are a legal document assistant.
Answer questions based on provided legal documents only.
Never give direct legal advice or legal recommendations.
Always recommend consulting a qualified legal professional.
Cite the specific clause and section for every answer.""",

    "finance": """You are a financial document assistant.
Answer questions based on provided financial documents only.
Never make investment recommendations or predictions.
Always cite the specific report and section for every answer.""",
}

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "domain" not in st.session_state:
    st.session_state.domain = "hr_enterprise"
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────
# AI PIPELINE — loaded once
# ─────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """
    Load the embedding model and LLM client once.
    Cached so they are not reloaded on every rerun.
    """
    from sentence_transformers import SentenceTransformer
    from groq import Groq

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.error("GROQ_API_KEY not set. Add it in Space Settings → Variables.")
        st.stop()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=groq_key)

    return embedding_model, groq_client


def get_vector_store(domain: str):
    """Get or create vector store for the active domain."""
    import chromadb
    client = chromadb.Client()
    collection = client.get_or_create_collection(f"domain_{domain}")
    return collection


def ingest_file(file_bytes, filename: str, domain: str) -> int:
    """Ingest uploaded file into vector store."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tempfile

    embedding_model, _ = load_pipeline()

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    # Load text
    if suffix == ".pdf":
        import fitz
        doc = fitz.open(str(tmp_path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    else:
        text = tmp_path.read_text(encoding="utf-8", errors="ignore")

    tmp_path.unlink()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Embed and store
    collection = get_vector_store(domain)
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return len(chunks)


def search_documents(query: str, domain: str, k: int = 3) -> list:
    """Search vector store for relevant chunks."""
    embedding_model, _ = load_pipeline()
    collection = get_vector_store(domain)

    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "content": doc,
                "metadata": meta,
                "distance": round(dist, 4),
            })

    return retrieved


def generate_answer(
    question: str,
    chunks: list,
    domain: str,
    chat_history: list,
) -> str:
    """Generate answer using Groq LLM."""
    _, groq_client = load_pipeline()

    system_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["hr_enterprise"])

    if chunks:
        context_parts = [
            f"[Section {i+1} from {c['metadata'].get('source','?')}]\n{c['content']}"
            for i, c in enumerate(chunks)
        ]
        context = "\n\n---\n\n".join(context_parts)
        user_message = f"""RELEVANT DOCUMENT SECTIONS:
═══════════════════════════════════
{context}
═══════════════════════════════════

USER QUESTION: {question}

Answer based on the document sections above."""
    else:
        user_message = question

    messages = chat_history + [{"role": "user", "content": user_message}]

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def render_message(role: str, content: str):
    if role == "user":
        st.markdown(
            f'<div class="chat-row-user">'
            f'<div class="bubble-user">{content}</div>'
            f'<div class="avatar-user">👤</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-row-bot">'
            f'<div class="avatar-bot">🤖</div>'
            f'<div class="bubble-bot">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.divider()

    st.markdown('<div class="section-label">API Status</div>',
                unsafe_allow_html=True)
    st.markdown('<span class="status-ok">✅ Ready</span>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-label">🌐 Active Domain</div>',
                unsafe_allow_html=True)
    selected = st.selectbox(
        "domain",
        options=list(DOMAINS.keys()),
        format_func=lambda x: DOMAINS[x],
        index=list(DOMAINS.keys()).index(st.session_state.domain),
        label_visibility="collapsed",
    )
    st.caption(DOMAIN_DESCRIPTIONS.get(selected, ""))

    if selected != st.session_state.domain:
        st.session_state.domain = selected
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown('<div class="section-label">📄 Upload Document</div>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "file",
        type=["pdf", "txt", "docx", "md"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("⬆️ Ingest Document", type="primary",
                     use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    chunks_count = ingest_file(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        st.session_state.domain,
                    )
                    st.success(f"✅ {chunks_count} chunks ingested from **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    st.caption(f"Messages: {len(st.session_state.messages)}")

# ─────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────

st.markdown(
    '<div class="app-title">🤖 AI Document Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="font-size:1rem;color:#64748b;">'
    f'Domain: <span class="domain-badge">{DOMAINS[st.session_state.domain]}</span>'
    f'</div>'
    f'<div class="tagline">'
    f'Semantic search · Grounded answers · Source citations'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()

# Welcome screen
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <span class="welcome-icon">💬</span>
        <div class="welcome-heading">Ready to answer your questions</div>
        <div class="welcome-text">
            Upload a document from the sidebar,<br>
            then ask anything about its contents.
        </div>
        <div class="feature-row">
            <div class="feature-chip">🔍 Semantic Search</div>
            <div class="feature-chip">📎 Source Citations</div>
            <div class="feature-chip">🧠 Memory</div>
            <div class="feature-chip">🛡️ Guardrails</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display conversation
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# Sources
if st.session_state.last_sources:
    pills = "".join([
        f'<span class="source-pill">'
        f'📎 {s.get("metadata",{}).get("source","?")} &nbsp;·&nbsp; '
        f'chunk {s.get("metadata",{}).get("chunk_index","?")} &nbsp;·&nbsp; '
        f'dist {s.get("distance","?")}'
        f'</span>'
        for s in st.session_state.last_sources
    ])
    st.markdown(
        f'<div class="sources-section">'
        f'<div class="sources-title">📄 Sources used</div>'
        f'{pills}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents..."):

    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🤖 Thinking..."):
        chunks = search_documents(prompt, st.session_state.domain)
        answer = generate_answer(
            prompt,
            chunks,
            st.session_state.domain,
            st.session_state.chat_history,
        )

    render_message("assistant", answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = chunks
    st.rerun()