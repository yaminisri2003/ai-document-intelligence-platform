# test_ingestion.py
# Run this file to test the document ingestion pipeline.
# Command: python tests/test_ingestion.py

from pathlib import Path
from app.rag.ingestion import DocumentIngestionPipeline


def test_ingestion():

    print("\n=== Testing Document Ingestion Pipeline ===\n")

    # Create the pipeline
    pipeline = DocumentIngestionPipeline(
        chunk_size=500,
        chunk_overlap=100
    )

    # Run ingestion on our sample document
    chunks = pipeline.ingest(
        file_path=Path("data/hr_enterprise/remote_work_policy.txt"),
        domain="hr_enterprise"
    )

    # Show results
    print(f"\nTotal chunks created: {len(chunks)}")

    print(f"\n--- CHUNK 1 CONTENT ---")
    print(chunks[0].page_content)

    print(f"\n--- CHUNK 1 METADATA ---")
    print(chunks[0].metadata)

    print(f"\n--- CHUNK 2 CONTENT ---")
    print(chunks[1].page_content)

    print(f"\n--- CHUNK 2 METADATA ---")
    print(chunks[1].metadata)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_ingestion()