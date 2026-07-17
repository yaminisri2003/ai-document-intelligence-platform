# main.py
# The FastAPI application — the entry point of our backend.
#
# This file:
# 1. Creates the FastAPI app
# 2. Defines all API endpoints
# 3. Connects requests to our AI pipeline
#
# To run: uvicorn app.api.main:app --reload
# Then visit: http://localhost:8000/docs
# to see automatic API documentation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import tempfile

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SourceInfo,
    IngestResponse,
    HealthResponse,
)
from app.memory.session_manager import (
    get_or_create_session,
    clear_session,
    get_active_sessions,
)
from app.rag.ingestion import DocumentIngestionPipeline
from app.rag.vectorstore import VectorStore
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)

# Create the FastAPI application
# title and description appear in automatic docs
app = FastAPI(
    title="AI Document Intelligence Platform",
    description="""
    A production-grade multi-domain RAG system.
    Supports Healthcare, Legal, HR, and Finance domains.
    """,
    version="1.0.0",
)

# CORS middleware allows the frontend to call this API
# In production you would restrict origins to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# ENDPOINT 1: Health Check
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API is running correctly.

    Used by deployment platforms to verify the service
    is healthy. Returns immediately with no AI processing.

    GET /health
    """
    logger.info("Health check requested")

    return HealthResponse(
        status="healthy",
        message=f"API running | active_sessions={get_active_sessions()}"
    )


# ─────────────────────────────────────────
# ENDPOINT 2: Chat
# ─────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get an AI response.

    Uses RAG to retrieve relevant document chunks
    and generates a grounded answer with citations.
    Maintains conversation history per session.

    POST /chat
    Body: {"message": "...", "domain": "...", "session_id": "..."}
    """

    logger.info(
        f"Chat request | "
        f"session={request.session_id} | "
        f"domain={request.domain} | "
        f"message='{request.message[:50]}'"
    )

    try:
        # Get or create conversation session
        agent = get_or_create_session(
            session_id=request.session_id,
            domain=request.domain,
        )

        # Get answer from the LangGraph agent
        result = agent.chat(request.message)

        # Format sources for response
        sources = [
            SourceInfo(
                source=s.get("source"),
                chunk_index=s.get("chunk_index"),
                distance=s.get("distance"),
            )
            for s in result.get("sources", [])
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            session_id=request.session_id,
            domain=request.domain,
        )

    except Exception as e:
        logger.error(f"Chat endpoint error | {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


# ─────────────────────────────────────────
# ENDPOINT 3: Document Ingestion
# ─────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    domain: str = Form(default="hr_enterprise"),
):
    """
    Upload and ingest a document into the vector store.

    Accepts PDF, TXT, DOCX, or Markdown files.
    Processes the document and stores chunks in ChromaDB.

    POST /ingest
    Form data: file (the document), domain (string)
    """

    logger.info(
        f"Ingest request | "
        f"filename={file.filename} | "
        f"domain={domain}"
    )

    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. "
                   f"Allowed: {allowed_extensions}"
        )

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        # Run ingestion pipeline
        pipeline = DocumentIngestionPipeline(
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = pipeline.ingest(
            file_path=tmp_path,
            domain=domain,
        )

        # Store chunks in vector database
        store = VectorStore(domain=domain)
        # Fix metadata so source shows original filename not temp path
        for chunk in chunks:
            chunk.metadata["source"] = file.filename
            
        store.add_documents(chunks)

        # Clean up temporary file
        tmp_path.unlink()

        logger.info(
            f"Ingestion complete | "
            f"file={file.filename} | "
            f"chunks={len(chunks)}"
        )

        return IngestResponse(
            message="Document ingested successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            domain=domain,
        )

    except Exception as e:
        logger.error(f"Ingest endpoint error | {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting document: {str(e)}"
        )


# ─────────────────────────────────────────
# STARTUP EVENT
# ─────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Runs when the API server starts.
    Validates configuration and logs startup info.
    """
    logger.info("API starting up...")
    settings.validate()
    logger.info(
        f"API ready | "
        f"active_domain={settings.ACTIVE_DOMAIN}"
    )