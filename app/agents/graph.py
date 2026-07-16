# graph.py
# This file builds the LangGraph state machine.
# It connects all nodes with edges to create
# the complete conversation workflow.
#
# The graph defines:
# - Which nodes exist
# - Which order they run in
# - Which conditions change the flow
#
# Once built, you call graph.invoke() with a question
# and it automatically runs through all the right nodes.

from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes import (
    router_node,
    retrieval_node,
    generator_node,
    guardrail_node,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def should_retrieve(state: dict) -> str:
    """
    Conditional edge function.
    Called after the router node to decide next step.

    Returns the name of the next node to run.

    Args:
        state: Current agent state

    Returns:
        "retrieve" if documents are needed
        "generate" if we can answer directly
    """

    if state.get("needs_retrieval", True):
        logger.info("Edge decision: routing to retrieval")
        return "retrieve"
    else:
        logger.info("Edge decision: skipping retrieval")
        return "generate"


def build_graph():
    """
    Build and compile the LangGraph state machine.

    The graph structure:
    START
      ↓
    router_node
      ↓ (conditional)
    ┌─────────────────┐
    │                 │
    retrieve      generate
    (if needed)   (if simple)
    │                 │
    └──────┬──────────┘
           ↓
      generator_node
           ↓
      guardrail_node
           ↓
          END

    Returns:
        Compiled LangGraph ready to invoke
    """

    logger.info("Building LangGraph state machine...")

    # Create a new graph using our AgentState schema
    workflow = StateGraph(AgentState)

    # Add all nodes to the graph
    # First argument: the name we use to reference this node
    # Second argument: the function to call for this node
    workflow.add_node("router",    router_node)
    workflow.add_node("retrieve",  retrieval_node)
    workflow.add_node("generate",  generator_node)
    workflow.add_node("guardrail", guardrail_node)

    # Set the entry point — which node runs first
    workflow.set_entry_point("router")

    # Add conditional edge after router
    # should_retrieve() decides which node runs next
    workflow.add_conditional_edges(
        "router",           # after this node
        should_retrieve,    # call this function
        {
            "retrieve": "retrieve",  # if returns "retrieve"
            "generate": "generate",  # if returns "generate"
        }
    )

    # After retrieval, always go to generator
    workflow.add_edge("retrieve", "generate")

    # After generator, always go to guardrail
    workflow.add_edge("generate", "guardrail")

    # After guardrail, end the graph
    workflow.add_edge("guardrail", END)

    # Compile the graph — this validates the structure
    # and prepares it for execution
    compiled = workflow.compile()

    logger.info("LangGraph state machine built successfully")

    return compiled


class ConversationAgent:
    """
    High-level wrapper around the LangGraph.

    Manages conversation history and provides
    a simple interface for multi-turn conversations.

    Usage:
        agent = ConversationAgent(domain="hr_enterprise")
        response = agent.chat("What are the working hours?")
        response = agent.chat("What about timezones?")
        # Second question remembers the first
    """

    def __init__(self, domain: str):
        """
        Initialize the conversation agent.

        Args:
            domain: Which domain to use
        """

        self.domain = domain
        self.graph = build_graph()

        # Conversation history stored here
        # Grows with each turn
        self.chat_history = []

        logger.info(
            f"ConversationAgent ready | domain={domain}"
        )

    def chat(self, question: str) -> dict:
        """
        Send a message and get a response.
        Automatically maintains conversation history.

        Args:
            question: The user's message

        Returns:
            Dict with answer, sources, and question
        """

        logger.info(
            f"Agent chat | "
            f"question='{question[:50]}' | "
            f"history_turns={len(self.chat_history)}"
        )

        # Build initial state for this turn
        initial_state = {
            "question":        question,
            "chat_history":    self.chat_history.copy(),
            "retrieved_chunks": [],
            "answer":          "",
            "domain":          self.domain,
            "needs_retrieval": True,
            "sources":         [],
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Extract the answer
        answer = final_state.get("answer", "")
        sources = final_state.get("sources", [])

        # Update conversation history for next turn
        # Add user message
        self.chat_history.append({
            "role":    "user",
            "content": question,
        })
        # Add assistant response
        self.chat_history.append({
            "role":    "assistant",
            "content": answer,
        })

        logger.info(
            f"Agent response | "
            f"answer_length={len(answer)} | "
            f"history_turns={len(self.chat_history)}"
        )

        return {
            "answer":   answer,
            "sources":  sources,
            "question": question,
            "domain":   self.domain,
        }

    def reset(self):
        """Clear conversation history to start fresh."""
        self.chat_history = []
        logger.info("Conversation history cleared")