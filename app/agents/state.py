# state.py
# Defines the shared state that flows through our LangGraph.
#
# Think of state as a shared notebook that every node
# can read from and write to.
# When one node finishes, it updates the notebook.
# The next node reads the updated notebook.
#
# TypedDict means we define exactly what keys the state
# has and what type each value is.
# This prevents typos and makes the code self-documenting.

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    The complete state of one conversation turn.

    Every field here is either read or written by at least one node.
    The graph passes this state from node to node automatically.
    """

    # The current question from the user
    question: str

    # Full conversation history as a list of message dicts
    # Format: [{"role": "user", "content": "..."},
    #          {"role": "assistant", "content": "..."}]
    # Starts empty, grows with each conversation turn
    chat_history: list

    # Chunks retrieved from the vector store
    # Empty list if no retrieval was needed
    retrieved_chunks: list

    # The final answer to return to the user
    answer: str

    # Which domain is currently active
    # e.g. "hr_enterprise", "healthcare", "legal"
    domain: str

    # Decision made by the router node
    # True  = this question needs document retrieval
    # False = this question can be answered directly
    needs_retrieval: bool

    # Sources used to generate the answer
    # Used for displaying citations to the user
    sources: list