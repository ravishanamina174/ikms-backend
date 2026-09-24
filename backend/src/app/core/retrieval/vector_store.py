
"""Vector store wrapper for Pinecone integration with LangChain."""

from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from ..config import get_settings


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create a PineconeVectorStore instance configured from settings."""
    settings = get_settings()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    candidate_models = list(dict.fromkeys([
        settings.gemini_embedding_model_name,
        "gemini-embedding-001",
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "text-embedding-004",
    ]))

    last_error = None
    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=settings.gemini_api_key,
                output_dimensionality=settings.gemini_embedding_dimension,
            )
            return PineconeVectorStore(
                index=index,
                embedding=embeddings,
            )
        except Exception as exc:  # pragma: no cover - fallback resolution path
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("Could not create Gemini embedding model for Pinecone vector store.")
def get_retriever(k: int | None = None):
    """Get a Pinecone retriever instance.

    Args:
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        PineconeVectorStore instance configured as a retriever.
    """
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k

    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve documents from Pinecone for a given query.

    Args:
        query: Search query string.
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        List of Document objects with metadata (including page numbers).
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


def index_documents(file_path: Path) -> int:
    """Index a list of Document objects into the Pinecone vector store.

    Args:
        docs: Documents to embed and upsert into the vector index.

    Returns:
        The number of documents indexed.
    """
    loader = PyPDFLoader(str(file_path), mode="single")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    # Attach filename metadata to each chunk so downstream consumers can
    # report which file a chunk originated from. Preserve any existing
    # page/page_number metadata provided by the loader.
    filename = Path(file_path).name
    for doc in texts:
        try:
            doc.metadata["filename"] = filename
        except Exception:
            doc.metadata = {"filename": filename}

    vector_store = _get_vector_store()
    vector_store.add_documents(texts)
    return len(texts)