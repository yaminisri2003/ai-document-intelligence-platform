# embeddings.py
# This module converts text into embeddings (lists of numbers).
#
# This is the bridge between human language and mathematics.
# Every chunk of text becomes 384 numbers that represent its meaning.
# Similar meanings produce similar numbers.
#
# We use sentence-transformers which runs entirely on your CPU.
# No API calls. No costs. No data leaving your machine.
# This is critical for privacy-sensitive domains like healthcare.

from sentence_transformers import SentenceTransformer
from app.utils.logger import get_logger

logger = get_logger(__name__)

# This is the model we use to generate embeddings.
# all-MiniLM-L6-v2 means:
# - all        = trained on all types of text
# - MiniLM     = a small but powerful architecture
# - L6         = 6 layers deep
# - v2         = version 2
# It produces 384 numbers per text and runs fast on CPU.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Converts text into numerical embeddings.

    Why a class?
    Loading the embedding model takes about 2-3 seconds.
    By wrapping it in a class and creating one instance,
    we load it once at startup and reuse it everywhere.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Load the embedding model.

        First time you run this:
        → Downloads the model (~80MB) from HuggingFace
        → Saves it to your local cache folder
        → Never downloads again after first time

        Every time after that:
        → Loads from local cache instantly
        """

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(
            "First run downloads ~80MB — "
            "this is a one-time download"
        )

        # Load the model — this is the slow step (2-3 seconds)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Get the size of embeddings this model produces
        # all-MiniLM-L6-v2 produces 384 numbers per text
        self.embedding_dim = self.model.get_embedding_dimension()

        logger.info(
            f"Embedding model ready | "
            f"model={model_name} | "
            f"dimensions={self.embedding_dim}"
        )

    def embed_text(self, text: str) -> list:
        """
        Convert a single piece of text into an embedding.

        Use this when embedding a user's question at query time.

        Args:
            text: Any string of text

        Returns:
            A list of 384 floating point numbers
            representing the meaning of the text
        """

        # encode() runs the text through the neural network
        # convert_to_numpy=True returns a numpy array
        # .tolist() converts it to a plain Python list
        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        ).tolist()

        logger.info(
            f"Text embedded | "
            f"text_length={len(text)} | "
            f"embedding_dimensions={len(embedding)}"
        )

        return embedding

    def embed_batch(self, texts: list) -> list:
        """
        Convert multiple texts into embeddings at once.

        Use this when embedding document chunks during ingestion.
        Processing all chunks together is much faster than
        embedding them one by one.

        Why faster?
        The model processes texts in parallel batches internally.
        10 texts together takes barely longer than 1 text alone.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embeddings — one per input text
            Each embedding is a list of 384 numbers
        """

        logger.info(f"Embedding batch of {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        ).tolist()

        logger.info(
            f"Batch embedding complete | "
            f"texts={len(texts)} | "
            f"dimensions={len(embeddings[0])}"
        )

        return embeddings


# Create one single instance to be shared across the project.
# The model loads once here — all other modules import this object.
# This prevents loading the model multiple times which would be slow.
embedding_model = EmbeddingModel()