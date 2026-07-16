# session_manager.py
# Manages conversation sessions for the API.
#
# Why sessions?
# HTTP is stateless — each request is independent.
# Sessions give us memory across requests.
#
# How it works:
# Each user gets a session_id (like "user123").
# Their ConversationAgent is stored in a dictionary.
# When they send another message with the same session_id,
# we retrieve their existing agent with full history.
#
# In production this would use Redis for persistence.
# For now we use a simple Python dictionary (in-memory).

from app.agents.graph import ConversationAgent
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Dictionary storing all active sessions
# Key:   session_id (string)
# Value: ConversationAgent instance
_sessions: dict = {}


def get_or_create_session(
    session_id: str,
    domain: str,
) -> ConversationAgent:
    """
    Get existing session or create a new one.

    If session_id exists → return existing agent
    If session_id is new → create fresh agent

    Args:
        session_id: Unique identifier for this conversation
        domain:     Which domain this session uses

    Returns:
        ConversationAgent with conversation history intact
    """

    if session_id not in _sessions:
        logger.info(
            f"Creating new session | "
            f"session_id={session_id} | "
            f"domain={domain}"
        )
        _sessions[session_id] = ConversationAgent(domain=domain)
    else:
        logger.info(
            f"Resuming existing session | "
            f"session_id={session_id} | "
            f"history_turns={len(_sessions[session_id].chat_history)}"
        )

    return _sessions[session_id]


def clear_session(session_id: str) -> bool:
    """
    Clear a specific session's conversation history.

    Args:
        session_id: Session to clear

    Returns:
        True if session existed and was cleared
        False if session did not exist
    """

    if session_id in _sessions:
        _sessions[session_id].reset()
        logger.info(f"Session cleared | session_id={session_id}")
        return True

    return False


def get_active_sessions() -> int:
    """Return count of active sessions."""
    return len(_sessions)