# schemas.py
# Defines the exact shape of all API requests and responses.
#
# Why this matters:
# Every endpoint knows exactly what data it will receive
# and exactly what data it will return.
# Pydantic validates everything automatically.
# Bad data never reaches your business logic.

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """
    Shape of data sent TO the /chat endpoint.

    Example request body:
    {
        "message": "What are the working hours?",
        "domain": "hr_enterprise",
        "session_id": "user123"
    }
    """

    # The user's message — required field
    message: str

    # Which domain to search — defaults to hr_enterprise
    domain: str = "hr_enterprise"

    # Session ID for conversation continuity
    # Same session_id = same conversation history
    session_id: str = "default"


class SourceInfo(BaseModel):
    """
    Information about one source chunk used in an answer.
    """
    source: Optional[str] = None
    chunk_index: Optional[int] = None
    distance: Optional[float] = None


class ChatResponse(BaseModel):
    """
    Shape of data returned FROM the /chat endpoint.

    Example response:
    {
        "answer": "Core hours are 10am to 3pm...",
        "sources": [{"source": "policy.txt", "chunk_index": 2}],
        "session_id": "user123",
        "domain": "hr_enterprise"
    }
    """

    answer: str
    sources: list[SourceInfo] = []
    session_id: str
    domain: str


class IngestRequest(BaseModel):
    """
    Shape of data sent TO the /ingest endpoint.
    The actual file is sent separately as form data.
    """
    domain: str = "hr_enterprise"


class IngestResponse(BaseModel):
    """
    Shape of data returned FROM the /ingest endpoint.
    """
    message: str
    filename: str
    chunks_created: int
    domain: str


class HealthResponse(BaseModel):
    """
    Shape of data returned FROM the /health endpoint.
    """
    status: str
    message: str