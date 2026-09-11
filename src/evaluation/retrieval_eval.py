"""Quantifies retrieval quality (Hit Rate@5, MRR) against the labeled test set, computed both
before and after cross-encoder re-ranking, to put a real number on Phase 4's improvement.

Run with: python -m src.evaluation.retrieval_eval
"""
import json
from pathlib import Path

from src.reranking.reranker import rerank
from src.retrieval.retriever import Retriever

TEST_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "test_set.json"
POOL_SIZE = 20
TOP_K = 5


def _load_retrieval_cases() -> list[dict]:
    cases = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    return [c for c in cases if c["expected_sources"]]


def _hit_and_reciprocal_rank(ranked_sources: list[str], expected: list[str]) -> tuple[bool, float]:
    for i, source in enumerate(ranked_sources, start=1):
        if source in expected:
            return True, 1.0 / i
    return False, 0.0


def evaluate() -> None:
    cases = _load_retrieval_cases()
    retriever = Retriever()

    before_hits, before_rrs = [], []
    after_hits, after_rrs = [], []

    print(f"{'ID':<5} {'Hit(before)':<12} {'Hit(after)':<11} {'RR(before)':<11} {'RR(after)':<10} Question")
    for case in cases:
        candidates = retriever.retrieve(case["question"], top_k=POOL_SIZE)

        before_sources = [c.source for c in candidates[:TOP_K]]
        before_hit, before_rr = _hit_and_reciprocal_rank(before_sources, case["expected_sources"])
        before_hits.append(before_hit)
        before_rrs.append(before_rr)

        ranked = rerank(case["question"], candidates, top_n=TOP_K)
        after_sources = [c.source for c in ranked]
        after_hit, after_rr = _hit_and_reciprocal_rank(after_sources, case["expected_sources"])
        after_hits.append(after_hit)
        after_rrs.append(after_rr)

        print(
            f"{case['id']:<5} {str(before_hit):<12} {str(after_hit):<11} "
            f"{before_rr:<11.2f} {after_rr:<10.2f} {case['question'][:50]}"
        )

    n = len(cases)
    print(f"\nEvaluated {n} retrieval-relevant questions ({TOP_K=}, {POOL_SIZE=})\n")
    print(f"{'Metric':<20} {'Before rerank':<15} {'After rerank':<15}")
    print(f"{'Hit Rate@' + str(TOP_K):<20} {sum(before_hits) / n:<15.3f} {sum(after_hits) / n:<15.3f}")
    print(f"{'MRR':<20} {sum(before_rrs) / n:<15.3f} {sum(after_rrs) / n:<15.3f}")


if __name__ == "__main__":
    evaluate()
