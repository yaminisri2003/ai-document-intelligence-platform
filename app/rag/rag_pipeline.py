# rag_pipeline.py
# This module is the heart of the entire project.
# It connects all three previous phases into one pipeline.
#
# The complete flow:
# User question
#   → retrieve relevant chunks from vector store
#   → build a prompt with those chunks as context
#   → send prompt to LLM
#   → return grounded answer with source citations
#
# This is RAG — Retrieval Augmented Generation.
# Retrieval  = finding relevant chunks
# Augmented  = adding those chunks to the prompt
# Generation = LLM generating an answer from them

from app.rag.vectorstore import VectorStore
from app.core.llm_client import llm_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for a specific domain.

    Connects vector store retrieval with LLM generation.
    Each domain has its own pipeline instance with its own
    system prompt and vector store collection.
    """

    def __init__(self, domain: str, system_prompt: str = None):
        """
        Initialize RAG pipeline for a domain.

        Args:
            domain:        e.g. "hr_enterprise", "healthcare"
            system_prompt: Custom instructions for this domain.
                          If None, uses a generic helpful prompt.
        """

        self.domain = domain

        # Each domain has its own vector store collection
        # This ensures HR questions only search HR documents
        # and healthcare questions only search medical documents
        self.vector_store = VectorStore(domain=domain)

        # The system prompt defines how the LLM behaves
        # for this specific domain
        self.system_prompt = system_prompt or self._default_system_prompt()

        logger.info(
            f"RAGPipeline ready | "
            f"domain={domain} | "
            f"chunks_available={self.vector_store.get_chunk_count()}"
        )

    def _default_system_prompt(self) -> str:
        """
        Default system prompt used when no custom one is provided.

        This prompt does three important things:
        1. Defines the AI's role
        2. Restricts answers to provided documents only
        3. Handles cases where information is not found
        """
        return """You are a helpful document assistant.
Answer the user's question using ONLY the information 
provided in the document sections below.

Important rules:
- If the answer is clearly in the documents, answer it directly
- Always mention which section your answer comes from
- If the answer is NOT in the provided documents, say exactly:
  "I don't have that information in the provided documents."
- Never guess or use information from outside the documents
- Keep your answer clear and concise"""

    def _build_prompt(
        self,
        question: str,
        retrieved_chunks: list,
    ) -> str:
        """
        Build the complete prompt to send to the LLM.

        This combines the retrieved chunks with the user question
        into a structured prompt that guides the LLM to answer
        only from the provided context.

        Args:
            question:         The user's question
            retrieved_chunks: List of relevant chunks from vector store

        Returns:
            Complete formatted prompt string
        """

        # Format each retrieved chunk with its source information
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source", "Unknown")
            content = chunk["content"]
            distance = chunk["distance"]

            context_parts.append(
                f"[Section {i+1} from {source}]\n{content}"
            )

        # Join all chunks into one context block
        context = "\n\n---\n\n".join(context_parts)

        # Build the complete prompt
        prompt = f"""RELEVANT DOCUMENT SECTIONS:
═══════════════════════════════════════
{context}
═══════════════════════════════════════

USER QUESTION:
{question}

YOUR ANSWER:"""

        return prompt

    def answer(
        self,
        question: str,
        k: int = 3,
    ) -> dict:
        """
        Answer a question using RAG.

        This is the main method you call from outside.
        It runs the complete pipeline end to end.

        Args:
            question: The user's question in natural language
            k:        Number of chunks to retrieve (default 3)

        Returns:
            Dict containing:
            - answer:   The LLM's response
            - sources:  Which chunks were used
            - question: The original question
            - domain:   Which domain was searched
        """

        logger.info(
            f"RAG pipeline started | "
            f"domain={self.domain} | "
            f"question='{question[:50]}'"
        )

        # Step 1: Retrieve relevant chunks
        logger.info("Step 1: Retrieving relevant chunks...")
        retrieved_chunks = self.vector_store.similarity_search(
            query=question,
            k=k,
        )

        if not retrieved_chunks:
            logger.warning("No chunks retrieved — vector store may be empty")
            return {
                "answer": "No documents found. Please ingest documents first.",
                "sources": [],
                "question": question,
                "domain": self.domain,
            }

        # Step 2: Build the prompt with context
        logger.info("Step 2: Building prompt with retrieved context...")
        prompt = self._build_prompt(question, retrieved_chunks)

        # Step 3: Send to LLM and get answer
        logger.info("Step 3: Sending to LLM for answer generation...")
        answer = llm_client.chat(
            user_message=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
            # Low temperature = consistent factual answers
            # We want the same question to give the same answer
        )

        # Step 4: Format and return results
        sources = [
            {
                "source":      chunk["metadata"].get("source"),
                "chunk_index": chunk["metadata"].get("chunk_index"),
                "distance":    chunk["distance"],
            }
            for chunk in retrieved_chunks
        ]

        logger.info(
            f"RAG pipeline complete | "
            f"answer_length={len(answer)} | "
            f"sources_used={len(sources)}"
        )

        return {
            "answer":   answer,
            "sources":  sources,
            "question": question,
            "domain":   self.domain,
        }