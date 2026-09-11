"""Shows the bi-encoder top-5 vs. the cross-encoder re-ranked top-5 side by side, so the effect
of re-ranking is directly visible instead of assumed.

Run with: python -m src.reranking.compare_rerank
"""
from src.reranking.reranker import rerank
from src.retrieval.retriever import Retriever

TEST_QUERIES = [
    "If I registered my NovaBook Pro 15 laptop 20 days ago, is a refund still possible?",
    "My earbuds won't charge, what can I do?",
    "How long do I have to return a monitor with dead pixels?",
]

POOL_SIZE = 20
TOP_N = 5


def compare(query: str):
    retriever = Retriever()
    candidates = retriever.retrieve(query, top_k=POOL_SIZE)

    print(f"\n{'=' * 90}\nQUERY: {query}\n{'=' * 90}")

    print(f"\n--- BEFORE (bi-encoder top-{TOP_N}, ranked by vector distance — lower is better) ---")
    for rank, c in enumerate(candidates[:TOP_N], start=1):
        print(f"  {rank}. [{c.source}] dist={c.distance:.3f}")

    ranked = rerank(query, candidates, top_n=TOP_N)
    print(f"\n--- AFTER (cross-encoder top-{TOP_N}, ranked by relevance score — higher is better) ---")
    for rank, c in enumerate(ranked, start=1):
        print(f"  {rank}. [{c.source}] score={c.score:.3f}")


if __name__ == "__main__":
    for q in TEST_QUERIES:
        compare(q)
