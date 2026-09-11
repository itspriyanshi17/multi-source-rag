"""Cross-encoder re-ranking: re-scores a candidate set of chunks against the query jointly,
which is more accurate than bi-encoder cosine similarity but too slow to run over a whole corpus."""
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.retrieval.retriever import RetrievedChunk

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder: CrossEncoder | None = None


@dataclass
class RankedChunk:
    text: str
    source: str
    chunk_index: int
    score: float  # cross-encoder relevance score — higher is more relevant


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def warm_up() -> None:
    """Forces the cross-encoder to load eagerly — call this once at process startup to avoid a
    slow first request."""
    _get_cross_encoder()


def rerank(query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RankedChunk]:
    """Re-scores `candidates` against `query` with a cross-encoder and returns the top_n,
    sorted by relevance score descending (higher = more relevant)."""
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [
        RankedChunk(text=c.text, source=c.source, chunk_index=c.chunk_index, score=float(s))
        for c, s in scored[:top_n]
    ]
