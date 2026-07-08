# test_rag_pipeline.py
# Run with: python -m tests.test_rag_pipeline
#
# This is the most important test so far.
# It runs the complete RAG pipeline end to end:
# Question → Retrieve chunks → Send to LLM → Get answer
#
# After this test you will have a working AI system
# that answers questions from your document.

from pathlib import Path
from app.rag.ingestion import DocumentIngestionPipeline
from app.rag.vectorstore import VectorStore
from app.rag.rag_pipeline import RAGPipeline


def setup_vectorstore():
    """
    Ingest the document and store in ChromaDB.
    This only needs to run once.
    After this the chunks persist on disk.
    """
    print("Setting up vector store with HR policy document...")

    pipeline = DocumentIngestionPipeline(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = pipeline.ingest(
        file_path=Path("data/hr_enterprise/remote_work_policy.txt"),
        domain="hr_enterprise"
    )

    store = VectorStore(domain="hr_enterprise")
    store.clear()
    store.add_documents(chunks)

    print(f"Vector store ready with {store.get_chunk_count()} chunks\n")


def test_rag():

    print("\n=== Testing Complete RAG Pipeline ===\n")

    # Setup: ingest document into vector store
    setup_vectorstore()

    # Create the RAG pipeline for HR domain
    rag = RAGPipeline(domain="hr_enterprise")

    # Test with real questions
    questions = [
        "What internet speed do I need for remote work?",
        "Will the company pay for my internet expenses?",
        "What are the consequences of policy violations?",
        "What are the core working hours for remote employees?",
    ]

    for question in questions:
        print(f"{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        result = rag.answer(question, k=2)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES USED:")
        for source in result['sources']:
            print(
                f"  - {source['source']} "
                f"(chunk {source['chunk_index']}, "
                f"distance={source['distance']})"
            )
        print()

    print("=== RAG Pipeline Test Complete ===")


if __name__ == "__main__":
    test_rag()