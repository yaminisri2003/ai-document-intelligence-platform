# nodes.py
# Contains all node functions for our LangGraph.
#
# Each function:
# - Takes the current state as input
# - Does one specific job
# - Returns a dict of state fields to update
#
# LangGraph automatically merges the returned dict
# into the existing state before calling the next node.

from app.rag.vectorstore import VectorStore
from app.core.llm_client import llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


def router_node(state: dict) -> dict:
    """
    NODE 1: Router
    Decides if the question needs document retrieval.

    Simple questions like greetings do not need retrieval.
    Document questions do.

    Why this matters:
    Skipping retrieval for simple questions makes the system
    faster and avoids irrelevant chunks being sent to the LLM.

    Args:
        state: Current agent state

    Returns:
        Updated state with needs_retrieval set
    """

    question = state["question"]
    logger.info(f"Router node | question='{question[:50]}'")

    # Simple keyword-based routing for now
    # In a more advanced system this would use the LLM itself
    # to classify the question
    greetings = ["hi", "hello", "hey", "thanks",
                 "thank you", "bye", "goodbye"]

    question_lower = question.lower().strip()

    # Check if this is a simple greeting
    is_greeting = any(
        question_lower.startswith(g) for g in greetings
    )

    needs_retrieval = not is_greeting

    logger.info(
        f"Router decision | "
        f"needs_retrieval={needs_retrieval}"
    )

    return {"needs_retrieval": needs_retrieval}


def retrieval_node(state: dict) -> dict:
    """
    NODE 2: Retriever
    Finds relevant document chunks for the question.

    Only runs if router decided needs_retrieval=True.

    Args:
        state: Current agent state

    Returns:
        Updated state with retrieved_chunks and sources
    """

    question = state["question"]
    domain = state["domain"]

    logger.info(
        f"Retrieval node | "
        f"domain={domain} | "
        f"question='{question[:50]}'"
    )

    # Get the vector store for this domain
    vector_store = VectorStore(domain=domain)

    # Search for relevant chunks
    chunks = vector_store.similarity_search(
        query=question,
        k=3,
    )

    logger.info(f"Retrieved {len(chunks)} chunks")

    # Format sources for citation display
    sources = [
        {
            "source":      c["metadata"].get("source"),
            "chunk_index": c["metadata"].get("chunk_index"),
            "distance":    c["distance"],
        }
        for c in chunks
    ]

    return {
        "retrieved_chunks": chunks,
        "sources": sources,
    }


def generator_node(state: dict) -> dict:
    """
    NODE 3: Generator
    Creates the final answer using LLM.

    Uses retrieved chunks as context if available.
    Uses chat history for conversation continuity.

    This is where RAG + Memory come together.

    Args:
        state: Current agent state

    Returns:
        Updated state with the final answer
    """

    question = state["question"]
    chat_history = state.get("chat_history", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    domain = state["domain"]
    needs_retrieval = state.get("needs_retrieval", True)

    logger.info(
        f"Generator node | "
        f"has_chunks={len(retrieved_chunks) > 0} | "
        f"history_turns={len(chat_history)}"
    )

    # Build system prompt based on domain
    system_prompt = _get_system_prompt(domain)

    # Build the user message combining context and question
    if retrieved_chunks and needs_retrieval:

        # Format retrieved chunks as context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            context_parts.append(
                f"[Section {i+1} from {source}]\n"
                f"{chunk['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Build prompt with context
        user_message = f"""RELEVANT DOCUMENT SECTIONS:
═══════════════════════════════════
{context}
═══════════════════════════════════

USER QUESTION: {question}

Answer based on the document sections above."""

    else:
        # No retrieval needed — direct answer
        user_message = question

    # Include chat history for multi-turn memory
    # Build messages list with history + current question
    messages = chat_history + [
        {"role": "user", "content": user_message}
    ]

    # Generate answer using full conversation history
    answer = llm_client.chat_with_history(
        messages=messages,
        system_prompt=system_prompt,
        temperature=0.1,
    )

    logger.info(
        f"Answer generated | length={len(answer)}"
    )

    return {"answer": answer}


def guardrail_node(state: dict) -> dict:
    """
    NODE 4: Guardrail
    Applies domain-specific safety checks.

    Different domains have different rules:
    - Healthcare: never give direct medical advice
    - Legal:      never give direct legal advice
    - HR:         escalate sensitive HR matters

    For now we implement basic keyword checking.
    In production this would be more sophisticated.

    Args:
        state: Current agent state

    Returns:
        Updated state with safety-checked answer
    """

    answer = state["answer"]
    domain = state["domain"]

    logger.info(f"Guardrail node | domain={domain}")

    # Domain-specific guardrail rules
    guardrails = {
        "healthcare": {
            "keywords": ["you should take", "I recommend taking",
                        "prescribed dose", "take this medication"],
            "warning":  "\n\n⚠️ Please consult a qualified "
                       "healthcare professional for medical advice."
        },
        "legal": {
            "keywords": ["you should sign", "legally you must",
                        "I advise you to"],
            "warning":  "\n\n⚠️ Please consult a qualified "
                       "legal professional for legal advice."
        },
        "hr_enterprise": {
            "keywords": ["you should quit", "sue the company",
                        "this is illegal"],
            "warning":  "\n\n⚠️ For sensitive HR matters, "
                       "please contact HR directly."
        },
    }

    # Check if domain has guardrails defined
    if domain in guardrails:
        rules = guardrails[domain]
        answer_lower = answer.lower()

        # Check if any flagged keywords appear in the answer
        triggered = any(
            kw in answer_lower
            for kw in rules["keywords"]
        )

        if triggered:
            # Append safety warning to the answer
            answer = answer + rules["warning"]
            logger.warning(
                f"Guardrail triggered | domain={domain}"
            )

    return {"answer": answer}


def _get_system_prompt(domain: str) -> str:
    """
    Returns the appropriate system prompt for each domain.

    Each domain has different behavior rules and tone.
    This is the domain config pattern in action.
    """

    prompts = {
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

    return prompts.get(domain, prompts["hr_enterprise"])