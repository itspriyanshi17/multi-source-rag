"""Baseline dense retriever: embeds a query and fetches top-k chunks from Chroma."""
from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.retrieval.embeddings import get_embedding_model

CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "database" / "chroma"
COLLECTION_NAME = "novatech_docs"


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


class Retriever:
    def __init__(self, chroma_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        self._model = get_embedding_model()
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = client.get_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self._model.encode([query]).tolist()
        results = self._collection.query(query_embeddings=query_embedding, n_results=top_k)

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    distance=dist,
                )
            )
        return chunks


_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def warm_up() -> None:
    """Forces the embedding model and Chroma collection to load eagerly, instead of lazily on
    first query — call this once at process startup to avoid a slow first request."""
    _get_retriever()


CANDIDATE_POOL_SIZE = 20  # wide bi-encoder recall pool, narrowed by the cross-encoder re-ranker


def search_documents(query: str, top_k: int = 5) -> str:
    """Searches NovaTech's policy documents, FAQs, product manuals, and support tickets for
    passages relevant to a question. Use this for questions about return/shipping/warranty
    policies, how-to instructions, account/billing/loyalty program rules, store locations, or
    past support cases — anything that isn't a structured product-catalog lookup.

    Args:
        query: The natural-language question or topic to search for.
        top_k: Number of passages to return after re-ranking (default 5).

    Returns:
        A formatted string of the top matching passages with their source filenames.
    """
    from src.reranking.reranker import rerank  # lazy import to avoid a circular import

    candidates = _get_retriever().retrieve(query, top_k=CANDIDATE_POOL_SIZE)
    if not candidates:
        return "No matching documents found."
    ranked = rerank(query, candidates, top_n=top_k)
    parts = [f"[Source: {c.source}]\n{c.text}" for c in ranked]
    return "\n\n".join(parts)


if __name__ == "__main__":
    retriever = Retriever()
    query = "Can I return my laptop after 20 days if I already registered it?"
    for chunk in retriever.retrieve(query, top_k=3):
        print(f"[{chunk.source} #{chunk.chunk_index}] dist={chunk.distance:.3f}")
        print(chunk.text[:150], "...\n")
