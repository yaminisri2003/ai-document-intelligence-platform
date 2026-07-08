# test_embeddings.py
# Run with: python -m tests.test_embeddings
#
# This test demonstrates the most important concept in RAG:
# Similar meanings produce similar numbers (embeddings)
# even when the words are completely different.

from app.rag.embeddings import embedding_model


def test_embeddings():

    print("\n=== Testing Embedding Model ===\n")

    # Test 1: Basic embedding
    print("--- Test 1: Convert text to numbers ---")
    text = "Employees can work remotely from home"
    embedding = embedding_model.embed_text(text)

    print(f"Text: '{text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 numbers: {embedding[:5]}")
    print(f"(Total of {len(embedding)} numbers represent this meaning)\n")

    # Test 2: The magic of semantic similarity
    # These sentences use DIFFERENT words but have SIMILAR meanings
    print("--- Test 2: Semantic Similarity Magic ---")

    sentence_a = "Can I work from home?"
    sentence_b = "remote work arrangements are available"
    sentence_c = "What is the stock market doing today?"

    embedding_a = embedding_model.embed_text(sentence_a)
    embedding_b = embedding_model.embed_text(sentence_b)
    embedding_c = embedding_model.embed_text(sentence_c)

    # Calculate similarity manually using dot product
    # (ChromaDB does this automatically — we do it manually here
    #  just to see the concept clearly)
    def simple_similarity(emb1, emb2):
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = sum(a ** 2 for a in emb1) ** 0.5
        magnitude2 = sum(b ** 2 for b in emb2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)

    sim_ab = simple_similarity(embedding_a, embedding_b)
    sim_ac = simple_similarity(embedding_a, embedding_c)

    print(f"Sentence A: '{sentence_a}'")
    print(f"Sentence B: '{sentence_b}'")
    print(f"Sentence C: '{sentence_c}'")
    print()
    print(f"Similarity A vs B (related topics): {sim_ab:.4f}")
    print(f"Similarity A vs C (unrelated topics): {sim_ac:.4f}")
    print()

    if sim_ab > sim_ac:
        print("RESULT: A and B are more similar than A and C")
        print("The model understands MEANING not just words!")
    else:
        print("Unexpected result - check the model")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_embeddings()