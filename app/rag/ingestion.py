# ingestion.py
# This module handles the first step of RAG — document ingestion.
#
# The job of this file:
# Take a raw document file → extract clean text → split into chunks
#
# Why do we need this?
# LLMs cannot read a whole 50 page document at once.
# We split it into small pieces so we can later find
# and send only the most relevant pieces to the LLM.

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestionPipeline:
    """
    Loads documents and splits them into chunks.

    Supported file types: TXT, PDF, DOCX, Markdown

    The three steps this class performs:
    1. LOAD   → read the file and extract raw text
    2. CLEAN  → remove messy formatting artifacts
    3. CHUNK  → split into overlapping pieces
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the pipeline with chunking settings.

        Args:
            chunk_size: How many characters per chunk.
                       500 characters is roughly 100-125 words.
                       This is a good balance — not too small,
                       not too large.

            chunk_overlap: How many characters to share between
                          consecutive chunks.
                          100 characters of overlap prevents
                          information being lost at chunk boundaries.

        Example with chunk_size=20 and chunk_overlap=5:
            Text:    "The cat sat on the mat in the room"
            Chunk 1: "The cat sat on the mat"
            Chunk 2: "on the mat in the room"
                      ^^^^^^^^^^^ shared overlap
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter is the most intelligent
        # splitter available in LangChain.
        #
        # It tries to split text at natural boundaries in this order:
        # 1. Paragraph breaks (\n\n) — best split point
        # 2. Line breaks (\n)        — second choice
        # 3. Sentences (". ")        — third choice
        # 4. Words (" ")             — fourth choice
        # 5. Characters ("")         — last resort
        #
        # This means it always tries to keep related text together.
        # A paragraph about medication dosages stays in one chunk.
        # It only splits mid-sentence if absolutely necessary.

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        logger.info(
            f"DocumentIngestionPipeline ready | "
            f"chunk_size={chunk_size} | "
            f"chunk_overlap={chunk_overlap}"
        )

    def load_txt(self, file_path: Path) -> str:
        """
        Load a plain text or markdown file.
        This is the simplest loader — just read the file.
        """
        try:
            text = file_path.read_text(encoding="utf-8")
            logger.info(
                f"TXT loaded | "
                f"file={file_path.name} | "
                f"characters={len(text)}"
            )
            return text
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path.name} | {e}")
            raise

    def load_pdf(self, file_path: Path) -> str:
        """
        Load a PDF file using PyMuPDF.

        PDFs are complex — they store text with font information,
        positioning, and encoding. PyMuPDF extracts the raw text
        from each page and combines them.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            text_parts = []

            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    # Add page marker — useful for citations later
                    text_parts.append(
                        f"[Page {page_num + 1}]\n{page_text}"
                    )

            doc.close()
            full_text = "\n\n".join(text_parts)

            logger.info(
                f"PDF loaded | "
                f"file={file_path.name} | "
                f"pages={len(doc)} | "
                f"characters={len(full_text)}"
            )
            return full_text

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path.name} | {e}")
            raise

    def load_file(self, file_path: Path) -> str:
        """
        Load any supported file type.
        Routes to the correct loader based on file extension.
        """
        suffix = file_path.suffix.lower()

        # Map file extensions to their loader functions
        loaders = {
            ".txt": self.load_txt,
            ".md":  self.load_txt,
            ".pdf": self.load_pdf,
        }

        if suffix not in loaders:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: {list(loaders.keys())}"
            )

        return loaders[suffix](file_path)

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text before chunking.

        Why cleaning matters:
        PDF extraction often produces:
        - Extra spaces between words
        - Random line breaks in the middle of sentences
        - Multiple blank lines
        - Hyphenated words broken across lines

        We fix all of these before chunking so our chunks
        contain clean, readable text.
        """
        import re

        # Fix words broken across lines with a hyphen
        # Example: "medica-\ntion" becomes "medication"
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Collapse multiple spaces into one
        text = re.sub(r" {2,}", " ", text)

        # Collapse more than 2 consecutive newlines into 2
        # We keep double newlines because they mark paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove spaces at the start and end
        text = text.strip()

        return text

    def ingest(
        self,
        file_path: Path,
        domain: str,
    ) -> list:
        """
        Full pipeline: load → clean → chunk.

        This is the main method you call from outside this class.
        It runs all three steps and returns a list of chunks.

        Args:
            file_path: Path to the document file
            domain:    Which domain this document belongs to
                      e.g. "hr_enterprise", "healthcare", "legal"

        Returns:
            List of LangChain Document objects.
            Each Document has:
            - page_content: the chunk text
            - metadata:     source file, domain, chunk number
        """

        logger.info(
            f"Starting ingestion | "
            f"file={file_path.name} | "
            f"domain={domain}"
        )

        # Step 1: Load
        raw_text = self.load_file(file_path)

        # Step 2: Clean
        clean = self.clean_text(raw_text)

        # Step 3: Chunk
        # create_documents returns a list of Document objects
        # Each Document has page_content and metadata attributes
        chunks = self.splitter.create_documents(
            texts=[clean],
            metadatas=[{
                "source":    file_path.name,
                "domain":    domain,
                "file_type": file_path.suffix.lower(),
            }]
        )

        # Add chunk index to each chunk's metadata
        # This tells us which chunk number this is out of total
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(
            f"Ingestion complete | "
            f"file={file_path.name} | "
            f"chunks={len(chunks)} | "
            f"avg_size={sum(len(c.page_content) for c in chunks) // len(chunks)} chars"
        )

        return chunks