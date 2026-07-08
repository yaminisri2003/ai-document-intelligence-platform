# vectorstore.py
# This module manages ChromaDB — our vector database.
#
# ChromaDB stores document chunk embeddings on disk and
# allows us to search them by semantic similarity.
#
# Two main operations:
# 1. add_documents()      → store chunks with their embeddings
# 2. similarity_search()  → find relevant chunks for a question
#
# Each domain gets its own separate collection in ChromaDB.
# This keeps healthcare documents separate from HR documents
# so queries never mix results across domains.

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.rag.embeddings import embedding_model
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.

    One VectorStore instance per domain.
    Each domain has its own isolated collection.
    """

    def __init__(self, domain: str):
        """
        Initialize vector store for a specific domain.

        Args:
            domain: e.g. "hr_enterprise", "healthcare", "legal"
                   Each domain gets its own ChromaDB collection.
                   This prevents cross-domain contamination.
        """

        self.domain = domain

        # Collection name is based on domain
        # e.g. "domain_hr_enterprise" or "domain_healthcare"
        self.collection_name = f"domain_{domain}"

        # PersistentClient saves everything to disk.
        # Data survives when you close the terminal.
        # Next time you start, your stored chunks are still there.
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get existing collection or create a new one
        # This is safe to call multiple times —
        # if collection exists it returns it,
        # if not it creates it
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"domain": domain},
        )

        logger.info(
            f"VectorStore ready | "
            f"domain={domain} | "
            f"collection={self.collection_name} | "
            f"existing_chunks={self.collection.count()}"
        )

    def add_documents(self, documents: list) -> None:
        """
        Embed and store a list of document chunks.

        This is called once during ingestion for each document.

        What happens step by step:
        1. Extract text from each Document object
        2. Generate embeddings for all texts at once (batch)
        3. Create unique IDs for each chunk
        4. Store texts + embeddings + metadata in ChromaDB

        Args:
            documents: List of LangChain Document objects
                      from the ingestion pipeline
        """

        if not documents:
            logger.warning("add_documents called with empty list")
            return

        logger.info(
            f"Storing {len(documents)} chunks | "
            f"domain={self.domain}"
        )

        # Extract the text content from each Document object
        texts = [doc.page_content for doc in documents]

        # Extract metadata from each Document object
        metadatas = [doc.metadata for doc in documents]

        # Create unique ID for each chunk
        # Format: filename_chunkindex
        # e.g. "remote_work_policy.txt_0" or "remote_work_policy.txt_1"
        ids = [
            f"{doc.metadata.get('source', 'unknown')}_{i}"
            for i, doc in enumerate(documents)
        ]

        # Generate embeddings for ALL chunks at once
        # This is much faster than embedding one by one
        logger.info("Generating embeddings for all chunks...")
        embeddings = embedding_model.embed_batch(texts)

        # Store everything in ChromaDB
        # ChromaDB saves to disk automatically
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(
            f"Successfully stored {len(documents)} chunks | "
            f"total_in_collection={self.collection.count()}"
        )

    def similarity_search(
        self,
        query: str,
        k: int = 3,
    ) -> list:
        """
        Find the most relevant chunks for a question.

        This is the RETRIEVAL step of RAG.

        What happens:
        1. Convert query to embedding (384 numbers)
        2. Compare against all stored chunk embeddings
        3. Return top K most similar chunks

        Args:
            query: The user's question in natural language
            k:     How many chunks to return (3 is a good default)
                  More chunks = more context but more noise
                  Fewer chunks = less noise but may miss information

        Returns:
            List of dicts, each containing:
            - content:  the chunk text
            - metadata: source file, domain, chunk index
            - distance: similarity score (lower = more similar)
        """

        logger.info(
            f"Searching for relevant chunks | "
            f"query='{query[:50]}...' | "
            f"k={k}"
        )

        # Convert the question to an embedding
        # Must use the SAME model used during ingestion
        # Different models would give incomparable numbers
        query_embedding = embedding_model.embed_text(query)

        # Search ChromaDB for most similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Format results into clean readable dicts
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved.append({
                    "content":  doc,
                    "metadata": meta,
                    "distance": round(dist, 4),
                })

        logger.info(
            f"Retrieved {len(retrieved)} chunks | "
            f"best_distance={retrieved[0]['distance'] if retrieved else 'N/A'}"
        )

        return retrieved

    def get_chunk_count(self) -> int:
        """Return total number of chunks stored."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Delete all chunks in this domain's collection.
        Use with caution — this cannot be undone.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        logger.warning(
            f"Cleared all chunks from domain: {self.domain}"
        )