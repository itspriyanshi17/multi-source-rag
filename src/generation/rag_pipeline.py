"""Ties retrieval and generation into a single baseline RAG call."""
from dataclasses import dataclass

from src.generation.generator import generate_answer
from src.retrieval.retriever import RetrievedChunk, Retriever

_retriever: Retriever | None = None


@dataclass
class RagResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def answer_question(question: str, top_k: int = 5) -> RagResult:
    retriever = get_retriever()
    chunks = retriever.retrieve(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return RagResult(question=question, answer=answer, chunks=chunks)


if __name__ == "__main__":
    result = answer_question("What is the return window for accessories?")
    print("Q:", result.question)
    print("\nA:", result.answer)
    print("\nRetrieved from:")
    for c in result.chunks:
        print(f"  - {c.source} (dist={c.distance:.3f})")
