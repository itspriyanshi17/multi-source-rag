"""Embeds document chunks with sentence-transformers and stores them in a persistent Chroma collection."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion.load_documents import load_and_chunk_documents

CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "database" / "chroma"
COLLECTION_NAME = "novatech_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_vector_store(chroma_dir: Path = CHROMA_DIR) -> int:
    chunks = load_and_chunk_documents()
    if not chunks:
        raise RuntimeError("No document chunks found — check data/documents/")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c.text for c in chunks], show_progress_bar=True).tolist()

    chroma_dir.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[f"{c.source}::{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    n = build_vector_store()
    print(f"Embedded and stored {n} chunks in Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
