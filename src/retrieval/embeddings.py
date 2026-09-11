"""Shared embedding model singleton, reused by both the fixed NovaTech retriever and the
per-session uploaded-file store so the model is only ever loaded into memory once."""
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model
