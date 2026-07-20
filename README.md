#  AI Document Intelligence Platform

> A production-grade multi-domain RAG system that answers questions
> from your documents using semantic search, LLM generation, and
> source citations.

**Supports:** Healthcare · Legal · HR/Enterprise · Finance

---

##  Live Demo

**[🚀 Try the Live Demo](https://ai-document-intelligence-platform.streamlit.app)**

---

##  Demo

**Ask questions in natural language:**

```
User: "What are the core working hours?"
AI:   "Remote employees must be available during core hours
       of 10am to 3pm in their local timezone Monday through
       Friday. (Source: remote_work_policy.txt, Section 3)"
```

```
User: "Will the company pay for my internet?"
AI:   "The company will reimburse up to $50 per month for
       internet expenses with valid receipts.
       (Source: remote_work_policy.txt, Section 6)"
```

---

##  Architecture

```
Document Upload (PDF/DOCX/TXT)
         ↓
Ingestion Pipeline (LangChain)
         ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
         ↓
Vector Store (ChromaDB)
         ↓
User Question → Semantic Search → Relevant Chunks
         ↓
LangGraph Agent
(Router → Retrieval → Generator → Guardrail)
         ↓
Groq LLM (Llama 3.1 8B) → Answer + Citations
         ↓
FastAPI Backend → Streamlit Frontend
```

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Inference | Groq API — Llama 3.1 8B |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (local) |
| Agent Workflow | LangGraph |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Document Parsing | PyMuPDF · LangChain |
| Observability | Structured logging |

---

##  Key Features

- **Multi-domain support** — Healthcare, Legal, HR, Finance with domain-specific system prompts and guardrails
- **Semantic search** — finds relevant content even when query uses different words than the document
- **Conversation memory** — multi-turn chat remembers previous messages
- **Source citations** — every answer shows which document chunk it came from
- **LangGraph agent** — intelligent routing skips retrieval for simple questions
- **Production architecture** — modular codebase, structured logging, error handling, config management

---

##  Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yaminisri2003/ai-document-intelligence-platform.git
cd ai-document-intelligence-platform
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
copy .env.example .env
# Add your GROQ_API_KEY to .env
# Get a free key at console.groq.com
```

**5. Start the backend**
```bash
uvicorn app.api.main:app --reload
```

**6. Start the frontend**
```bash
streamlit run frontend/app.py
```

**7. Open your browser**
```
http://localhost:8501
```

---

##  Project Structure

```
ai-document-intelligence-platform/
├── app/
│   ├── api/           # FastAPI endpoints (chat, ingest, health)
│   ├── agents/        # LangGraph state machine (router, retrieval,
│   │                  # generator, guardrail nodes)
│   ├── core/          # LLM client wrapper (Groq API)
│   ├── memory/        # Session management
│   ├── models/        # Pydantic request/response schemas
│   ├── rag/           # Ingestion, embeddings, vector store, pipeline
│   └── utils/         # Config loader, structured logger
├── frontend/          # Streamlit chat interface
├── data/              # Sample documents per domain
├── tests/             # Unit tests for each component
└── configs/           # Domain configuration files
```

---

##  What I Learned Building This

- How RAG grounds LLM answers in real documents preventing hallucination
- Why embeddings enable semantic search beyond keyword matching
- How LangGraph manages stateful agent workflows with conditional routing
- How to separate frontend and backend cleanly through REST APIs
- Production patterns — structured logging, error handling, environment config

---

##  Project Stats

- **20+ commits** following conventional commit standards from day one
- **8 phases** built incrementally with tests at each stage
- **4 domains** supported from a single codebase using config pattern

---

*Built by Yamini Sri — learning GenAI engineering in public*

*[GitHub](https://github.com/yaminisri2003) · [LinkedIn](https://www.linkedin.com/in/yaminisri2003) · [Medium](https://medium.com/@yaminisri2003)*