# test_vectorstore.py
# Run with: python -m tests.test_vectorstore
#
# This test demonstrates the complete Phase 3 pipeline:
# Ingest document → embed chunks → store → search by meaning
#
# This is the moment RAG becomes real.
# We ask a question using different words than the document
# and the system finds the right answer anyway.

from pathlib import Path
from app.rag.ingestion import DocumentIngestionPipeline
from app.rag.vectorstore import VectorStore


def test_vectorstore():

    print("\n=== Testing Vector Store ===\n")

    # Step 1: Ingest the document into chunks
    print("--- Step 1: Ingesting document ---")
    pipeline = DocumentIngestionPipeline(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = pipeline.ingest(
        file_path=Path("data/hr_enterprise/remote_work_policy.txt"),
        domain="hr_enterprise"
    )
    print(f"Created {len(chunks)} chunks\n")

    # Step 2: Store chunks in ChromaDB
    print("--- Step 2: Storing in vector database ---")
    store = VectorStore(domain="hr_enterprise")

    # Clear any existing data first so we start fresh
    store.clear()

    # Now store the chunks
    store.add_documents(chunks)
    print(f"Stored {store.get_chunk_count()} chunks in ChromaDB\n")

    # Step 3: Search with semantic queries
    # Notice we use DIFFERENT words than the document
    print("--- Step 3: Semantic Search Results ---\n")

    queries = [
        "Can I work from home?",
        "What internet speed do I need?",
        "What happens if I break the rules?",
        "Will the company pay for my internet?",
    ]

    for query in queries:
        print(f"Question: '{query}'")
        results = store.similarity_search(query, k=1)

        if results:
            print(f"Best match (distance={results[0]['distance']}):")
            print(f"{results[0]['content'][:200]}...")
            print(f"Source: {results[0]['metadata']['source']}")
        print()

    print("=== Test Complete ===")


if __name__ == "__main__":
    test_vectorstore()